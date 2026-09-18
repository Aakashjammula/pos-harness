"""/auth/* and /credentials/* endpoints.

Kept out of cli/server.py, which is already large and owns the realtime
transport rather than account management."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, EmailStr, Field

from pos import config
from pos.auth.deps import (
    REFRESH_COOKIE,
    clear_auth_cookies,
    require_user_id,
    set_auth_cookies,
)
from pos.auth.mailer import send_magic_link_email
from pos.auth.passwords import hash_password, verify_password
from pos.auth.store import EmailTaken, UserStore
from pos.auth.tokens import (
    MAGIC_LINK_TOKEN_TTL,
    REFRESH_TOKEN_TTL,
    create_access_token,
    hash_magic_link_token,
    hash_refresh_token,
    new_magic_link_token,
    new_refresh_token,
)

MAGIC_LINK_REQUEST_COOLDOWN = timedelta(seconds=60)


class MagicLinkRequest(BaseModel):
    email: EmailStr


class MagicLinkVerify(BaseModel):
    token: str

# Which request fields belong to which provider, and what environment
# variable each one becomes. Mirrors the mapping cli/server.py used for
# the old per-connection key_token flow.
PROVIDER_FIELDS: dict[str, dict[str, str]] = {
    "local": {"local_api_key": "LOCAL_API_KEY", "local_base_url": "LOCAL_BASE_URL"},
    "openai": {"openai_api_key": "OPENAI_API_KEY"},
    "azure": {
        "azure_api_key": "AZURE_OPENAI_API_KEY",
        "azure_endpoint": "AZURE_OPENAI_ENDPOINT",
        "azure_deployment": "AZURE_OPENAI_DEPLOYMENT",
    },
    "anthropic": {"anthropic_api_key": "ANTHROPIC_API_KEY"},
    "gemini": {"gemini_api_key": "GOOGLE_API_KEY"},
    "bedrock": {
        "bedrock_access_key_id": "AWS_ACCESS_KEY_ID",
        "bedrock_secret_access_key": "AWS_SECRET_ACCESS_KEY",
        "bedrock_region": "AWS_REGION",
    },
    "openrouter": {"openrouter_api_key": "OPENROUTER_API_KEY"},
    "tavily": {"tavily_api_key": "TAVILY_API_KEY"},
}


class Credentials(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


def build_auth_router(users: UserStore) -> APIRouter:
    router = APIRouter()

    def _issue(response: Response, user_id: str) -> None:
        raw_refresh, refresh_hash = new_refresh_token()
        users.store_refresh_token(user_id, refresh_hash, datetime.now(UTC) + REFRESH_TOKEN_TTL)
        set_auth_cookies(response, create_access_token(user_id), raw_refresh)

    @router.post("/auth/signup")
    async def signup(body: Credentials, response: Response):
        try:
            user = users.create_user(body.email, hash_password(body.password))
        except EmailTaken as e:
            raise HTTPException(status_code=409, detail="email already registered") from e
        _issue(response, user["id"])
        return {"id": user["id"], "email": user["email"]}

    @router.post("/auth/login")
    async def login(body: Credentials, response: Response):
        user = users.get_user_by_email(body.email)
        # Same 401 for an unknown email, a magic-link-only account (no
        # password set), and a wrong password: distinguishing any of
        # these tells an attacker which addresses are registered.
        if (
            user is None
            or user["password_hash"] is None
            or not verify_password(body.password, user["password_hash"])
        ):
            raise HTTPException(status_code=401, detail="invalid email or password")
        _issue(response, user["id"])
        return {"id": user["id"], "email": user["email"]}

    @router.post("/auth/magic-link/request")
    async def request_magic_link(body: MagicLinkRequest):
        email = body.email.lower()
        # Always 200 regardless of throttle/send outcome -- the caller
        # can't distinguish "already has one outstanding" from "just sent
        # a new one" from "email is known", any of which would leak
        # whether that address has an account.
        if not users.recent_magic_link_request(email, within=MAGIC_LINK_REQUEST_COOLDOWN):
            raw, token_hash = new_magic_link_token()
            users.store_magic_link_token(email, token_hash, datetime.now(UTC) + MAGIC_LINK_TOKEN_TTL)
            link_url = f"{config.FRONTEND_URL}/magic-link?token={raw}"
            try:
                await send_magic_link_email(email, link_url)
            except Exception as e:
                print(f"  magic-link email send failed for {email}: {e}")
        return {"ok": True}

    @router.post("/auth/magic-link/verify")
    async def verify_magic_link(body: MagicLinkVerify, response: Response):
        email = users.consume_magic_link_token(hash_magic_link_token(body.token))
        if email is None:
            raise HTTPException(status_code=401, detail="invalid or expired link")
        user = users.get_or_create_user_by_email(email)
        _issue(response, user["id"])
        return {"id": user["id"], "email": user["email"]}

    @router.post("/auth/logout")
    async def logout(request: Request, response: Response):
        raw = request.cookies.get(REFRESH_COOKIE)
        if raw:
            users.consume_refresh_token(hash_refresh_token(raw))
        clear_auth_cookies(response)
        return {"ok": True}

    @router.post("/auth/refresh")
    async def refresh(request: Request, response: Response):
        raw = request.cookies.get(REFRESH_COOKIE)
        token_hash = hash_refresh_token(raw) if raw else None
        user_id = users.consume_refresh_token(token_hash) if token_hash else None
        if user_id is None:
            # Presenting an already-revoked token means someone is replaying a
            # stolen or stale one. Which of the two it is can't be told apart,
            # so drop the whole family and make everyone sign in again.
            replayed_by = users.user_for_revoked_token(token_hash) if token_hash else None
            if replayed_by is not None:
                users.revoke_all_refresh_tokens(replayed_by)
                print(f"  refresh token replay detected for user {replayed_by} -- all tokens revoked")
            clear_auth_cookies(response)
            raise HTTPException(status_code=401, detail="invalid refresh token")
        _issue(response, user_id)
        return {"ok": True}

    @router.get("/auth/me")
    async def me(user_id: str = Depends(require_user_id)):
        user = users.get_user_by_id(user_id)
        if user is None:
            raise HTTPException(status_code=401, detail="not authenticated")
        return user

    @router.get("/credentials")
    async def list_credentials(user_id: str = Depends(require_user_id)):
        return {"configured": users.list_credential_providers(user_id)}

    @router.put("/credentials/{provider}")
    async def put_credential(
        provider: str, body: dict, user_id: str = Depends(require_user_id)
    ):
        fields = PROVIDER_FIELDS.get(provider)
        if fields is None:
            raise HTTPException(status_code=404, detail=f"unknown provider {provider!r}")
        payload = {
            env_name: str(body[field]).strip()
            for field, env_name in fields.items()
            if body.get(field) and str(body[field]).strip()
        }
        if not payload:
            raise HTTPException(status_code=422, detail="no credential fields provided")
        users.save_credential(user_id, provider, payload)
        return {"saved": provider}

    @router.delete("/credentials/{provider}")
    async def delete_credential(provider: str, user_id: str = Depends(require_user_id)):
        if not users.delete_credential(user_id, provider):
            raise HTTPException(status_code=404, detail="no stored credential for that provider")
        return {"deleted": provider}

    return router
