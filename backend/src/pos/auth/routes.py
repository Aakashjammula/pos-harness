"""/auth/* and /credentials/* endpoints.

Kept out of cli/server.py, which is already large and owns the realtime
transport rather than account management."""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, EmailStr, Field

from pos import config
from pos.auth.deps import (
    REFRESH_COOKIE,
    Auth,
    clear_auth_cookies,
    set_auth_cookies,
)
from pos.auth.mailer import send_magic_link_email
from pos.auth.passwords import DUMMY_HASH, hash_password, verify_password
from pos.auth.store import EmailTaken, UsernameTaken, UserStore
from pos.auth.tokens import (
    MAGIC_LINK_TOKEN_TTL,
    REFRESH_TOKEN_TTL,
    create_access_token,
    hash_magic_link_token,
    hash_refresh_token,
    new_magic_link_token,
    new_refresh_token,
)
from pos.llm.model_listing import ModelListError, list_models, supported_providers
from pos.net_policy import UnsafeUrl, validate_credential_urls
from pos.tools import all_tools, get_tool
from pos.tools import credential_fields as tool_credential_fields

MAGIC_LINK_REQUEST_COOLDOWN = timedelta(seconds=60)


class MagicLinkRequest(BaseModel):
    email: EmailStr


class MagicLinkVerify(BaseModel):
    token: str

# Which request fields belong to which provider, and what environment
# variable each one becomes. Mirrors the mapping cli/server.py used for
# the old per-connection key_token flow.
_LLM_PROVIDER_FIELDS: dict[str, dict[str, str]] = {
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
}

# Tools that need a key (Tavily, ...) declare their own fields in pos.tools, so a
# new tool's credential form works here without editing this file.
PROVIDER_FIELDS: dict[str, dict[str, str]] = {**_LLM_PROVIDER_FIELDS, **tool_credential_fields()}


# Credential fields that are settings, not secrets: safe to show back to their owner.
_PUBLIC_ENV = {"LOCAL_BASE_URL", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT", "AWS_REGION"}

# The one secret per provider that identifies it, shown masked ("••••abcd") so a person can tell
# WHICH key is saved without it ever being sent back. Short values are not hinted at all: the
# last four characters of a short secret are too large a share of it.
_HINT_ENV = {
    "local": "LOCAL_API_KEY", "openai": "OPENAI_API_KEY", "azure": "AZURE_OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY", "gemini": "GOOGLE_API_KEY", "bedrock": "AWS_ACCESS_KEY_ID",
    "openrouter": "OPENROUTER_API_KEY", "tavily": "TAVILY_API_KEY",
}
_MIN_HINT_LENGTH = 12


def _secret_hint(provider: str, saved: dict) -> str | None:
    secret = saved.get(_HINT_ENV.get(provider, ""), "")
    return f"••••{secret[-4:]}" if len(secret) >= _MIN_HINT_LENGTH else None


class Credentials(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class ProfileUpdate(BaseModel):
    """Same rules as signup. The email is deliberately not editable here: changing the
    address an account is tied to must be proven, and that flow does not exist yet."""

    name: str | None = Field(default=None, min_length=1, max_length=80)
    username: str | None = Field(default=None, min_length=3, max_length=32, pattern=r"^[a-zA-Z0-9_-]+$")


class AccountDelete(BaseModel):
    confirm_email: str
    password: str | None = None


class PasswordChange(BaseModel):
    """`current_password` is required when the account already has a password;
    an account that signs in only by email link may set its first without one."""

    current_password: str | None = None
    new_password: str = Field(min_length=8)


class SignupBody(BaseModel):
    """Signup asks for more than login does, and its password is
    optional: leaving it blank creates a magic-link-only account, which
    the store has always allowed (password_hash is NULLable) but no
    endpoint could reach until now."""

    name: str = Field(min_length=1, max_length=80)
    username: str = Field(min_length=3, max_length=32, pattern=r"^[a-zA-Z0-9_-]+$")
    email: EmailStr
    password: str | None = Field(default=None, min_length=8)


def build_auth_router(users: UserStore, auth: Auth | None = None) -> APIRouter:
    router = APIRouter()
    auth = auth or Auth(users)
    require_user_id = auth.require_user_id

    def _issue(request: Request, response: Response, user_id: str, session_id: str | None = None) -> None:
        """Start (or, on refresh, continue) a login session and set its cookies."""
        if session_id is None:
            session_id = users.create_auth_session(
                user_id, request.headers.get("user-agent"), request.client.host if request.client else None
            )
        raw_refresh, refresh_hash = new_refresh_token()
        users.store_refresh_token(user_id, refresh_hash, datetime.now(UTC) + REFRESH_TOKEN_TTL, session_id)
        set_auth_cookies(response, create_access_token(user_id, session_id), raw_refresh)

    @router.post("/auth/signup")
    async def signup(body: SignupBody, request: Request, response: Response):
        try:
            user = users.create_user(
                body.email,
                hash_password(body.password) if body.password else None,
                name=body.name.strip(),
                username=body.username,
            )
        except EmailTaken as e:
            raise HTTPException(status_code=409, detail="email already registered") from e
        except UsernameTaken as e:
            raise HTTPException(status_code=409, detail="username already taken") from e

        # With a password they are signed in here and now. Without one
        # there is nothing to authenticate against, so the account exists
        # but the session does not: send the link and let them in from
        # their inbox. Never issue cookies on this branch -- doing so
        # would make an unverified email a logged-in session.
        if body.password is None:
            sent = await _send_magic_link(body.email.lower())
            result = {"id": user["id"], "email": user["email"], "magic_link_sent": sent}
            if not sent:
                # The account exists but no email went out, so the person has no
                # way in yet. Say so instead of showing "check your inbox".
                result["message"] = (
                    "Your account was created, but the sign-in email couldn't be sent. "
                    "Try \"email me a link\" on the sign-in page in a minute."
                )
            return result

        _issue(request, response, user["id"])
        return {"id": user["id"], "email": user["email"], "magic_link_sent": False}

    @router.post("/auth/login")
    async def login(body: Credentials, request: Request, response: Response):
        user = users.get_user_by_email(body.email)
        # Same 401 for an unknown email, a magic-link-only account (no
        # password set), and a wrong password: distinguishing any of
        # these tells an attacker which addresses are registered. The
        # same goes for how long it takes, so exactly one password
        # verification always runs -- against a dummy hash when there is
        # no real one -- and the dummy can never authenticate.
        real_hash = user["password_hash"] if user else None
        password_ok = verify_password(body.password, real_hash or DUMMY_HASH)
        if user is None or real_hash is None or not password_ok:
            raise HTTPException(status_code=401, detail="invalid email or password")
        _issue(request, response, user["id"])
        return {"id": user["id"], "email": user["email"]}

    async def _send_magic_link(email: str) -> bool:
        """Mint, store and mail one link, honouring the cooldown. Shared
        by the request endpoint and by a passwordless signup, so the
        throttle covers both -- signing up repeatedly must not be a way
        around the rate limit on the request endpoint.

        True when a link is on its way (freshly sent, or one already
        outstanding inside the cooldown); False when sending failed."""
        if users.recent_magic_link_request(email, within=MAGIC_LINK_REQUEST_COOLDOWN):
            return True
        raw, token_hash = new_magic_link_token()
        users.store_magic_link_token(email, token_hash, datetime.now(UTC) + MAGIC_LINK_TOKEN_TTL)
        link_url = f"{config.FRONTEND_URL}/magic-link?token={raw}"
        try:
            await send_magic_link_email(email, link_url)
        except Exception as e:
            print(f"  magic-link email send failed for {email}: {e}")
            return False
        return True

    @router.post("/auth/magic-link/request")
    async def request_magic_link(body: MagicLinkRequest):
        # Always 200 regardless of throttle/send outcome -- the caller
        # can't distinguish "already has one outstanding" from "just sent
        # a new one" from "email is known", any of which would leak
        # whether that address has an account.
        await _send_magic_link(body.email.lower())
        return {"ok": True}

    @router.post("/auth/magic-link/verify")
    async def verify_magic_link(body: MagicLinkVerify, request: Request, response: Response):
        email = users.consume_magic_link_token(hash_magic_link_token(body.token))
        if email is None:
            raise HTTPException(status_code=401, detail="invalid or expired link")
        user = users.get_or_create_user_by_email(email)
        record = users.get_user_by_id(user["id"])
        if not record["email_verified"]:
            # Proving the address for the first time. If this browser is already
            # signed in as this very account, its holder is just verifying it. Anyone
            # else proving it may be the real owner meeting an account somebody
            # created for their address in advance (pre-hijacking): they get a
            # clean account and the creator keeps nothing.
            holder = auth.user_id_from_request(request) == user["id"]
            if not holder:
                users.reset_unverified_account(user["id"])
            users.mark_email_verified(user["id"])
            if holder:
                return {"id": user["id"], "email": user["email"]}
        _issue(request, response, user["id"])
        return {"id": user["id"], "email": user["email"]}

    @router.patch("/auth/me")
    async def update_profile(body: ProfileUpdate, user_id: str = Depends(require_user_id)):
        changes = body.model_dump(exclude_none=True)
        if changes:
            if "name" in changes:
                changes["name"] = changes["name"].strip()
            try:
                users.update_profile(user_id, **changes)
            except UsernameTaken as e:
                raise HTTPException(status_code=409, detail="username already taken") from e
        return users.get_user_by_id(user_id)

    @router.post("/auth/account/delete")
    async def delete_account(body: AccountDelete, response: Response, user_id: str = Depends(require_user_id)):
        """Permanent. Needs the email typed out (so a stray click cannot do it) and, when
        the account has a password, that password (so a borrowed session cannot do it)."""
        user = users.get_user_by_id(user_id)
        if body.confirm_email.strip().lower() != user["email"]:
            raise HTTPException(status_code=400, detail="the email you typed does not match this account")
        current = users.get_password_hash(user_id)
        if current is not None and (not body.password or not verify_password(body.password, current)):
            raise HTTPException(status_code=401, detail="password is incorrect")
        users.delete_user(user_id)
        clear_auth_cookies(response)
        return {"deleted": True}

    @router.put("/auth/password")
    async def change_password(body: PasswordChange, request: Request, user_id: str = Depends(require_user_id)):
        current = users.get_password_hash(user_id)
        if current is not None and (
            not body.current_password or not verify_password(body.current_password, current)
        ):
            raise HTTPException(status_code=401, detail="current password is incorrect")
        users.set_password_hash(user_id, hash_password(body.new_password))
        # A password change is a security boundary (OWASP): whoever else was signed in,
        # on a device the person may no longer control, is signed out.
        users.revoke_all_auth_sessions(user_id, except_id=auth.session_id_from_request(request))
        return {"ok": True}

    @router.post("/auth/verify-email/request")
    async def request_email_verification(user_id: str = Depends(require_user_id)):
        """Mail this account's address a link; opening it in this browser marks the
        address verified without touching anything else."""
        user = users.get_user_by_id(user_id)
        if user["email_verified"]:
            return {"sent": False, "already_verified": True}
        return {"sent": await _send_magic_link(user["email"])}

    @router.post("/auth/logout")
    async def logout(request: Request, response: Response):
        """End this device's login session server-side, so a copied access
        token stops working too -- not just the browser's cookies."""
        user_id = auth.user_id_from_request(request)
        session_id = auth.session_id_from_request(request)
        if user_id and session_id:
            users.revoke_auth_session(user_id, session_id)
        else:
            raw = request.cookies.get(REFRESH_COOKIE)
            if raw:   # signed in only by a refresh cookie (access token expired): end that session
                found = users.consume_refresh_token_with_session(hash_refresh_token(raw))
                if found and found[1]:
                    users.revoke_auth_session(found[0], found[1])
        clear_auth_cookies(response)
        return {"ok": True}

    @router.post("/auth/refresh")
    async def refresh(request: Request, response: Response):
        raw = request.cookies.get(REFRESH_COOKIE)
        token_hash = hash_refresh_token(raw) if raw else None
        found = users.consume_refresh_token_with_session(token_hash) if token_hash else None
        if found is None:
            # Presenting an already-rotated token means someone is replaying a
            # stolen or stale one. Which of the two it is can't be told apart,
            # so drop the whole family and make everyone sign in again. (A token
            # of a REVOKED session is deleted, not rotated, so it lands here as
            # "unknown" and does not trigger this.)
            replayed_by = users.user_for_revoked_token(token_hash) if token_hash else None
            if replayed_by is not None:
                users.revoke_all_refresh_tokens(replayed_by)
                print(f"  refresh token replay detected for user {replayed_by} -- all tokens revoked")
            clear_auth_cookies(response)
            raise HTTPException(status_code=401, detail="invalid refresh token")
        user_id, session_id = found
        if session_id is None:   # issued before login sessions existed: adopt it into one
            _issue(request, response, user_id)
            return {"ok": True}
        if not users.auth_session_active(session_id, user_id):
            clear_auth_cookies(response)
            raise HTTPException(status_code=401, detail="session ended")
        users.touch_auth_session(session_id)
        _issue(request, response, user_id, session_id)
        return {"ok": True}

    @router.get("/auth/sessions")
    async def list_sessions(request: Request, user_id: str = Depends(require_user_id)):
        """This user's signed-in devices. Only their own, never anyone else's."""
        current = auth.session_id_from_request(request)
        return {
            "sessions": [
                {
                    "id": s["id"],
                    "user_agent": s["user_agent"],
                    "ip": s["ip"],
                    "created_at": s["created_at"],
                    "last_seen_at": s["last_seen_at"],
                    "current": s["id"] == current,
                }
                for s in users.list_auth_sessions(user_id)
            ]
        }

    @router.delete("/auth/sessions/{session_id}")
    async def revoke_session(
        session_id: str, request: Request, response: Response, user_id: str = Depends(require_user_id)
    ):
        if not users.revoke_auth_session(user_id, session_id):
            raise HTTPException(status_code=404, detail="session not found")   # also for someone else's: no probing
        if session_id == auth.session_id_from_request(request):
            clear_auth_cookies(response)
        return {"revoked": session_id}

    @router.post("/auth/sessions/revoke-others")
    async def revoke_other_sessions(request: Request, user_id: str = Depends(require_user_id)):
        return {"revoked": users.revoke_all_auth_sessions(user_id, except_id=auth.session_id_from_request(request))}

    @router.get("/auth/me")
    async def me(user_id: str = Depends(require_user_id)):
        user = users.get_user_by_id(user_id)
        if user is None:
            raise HTTPException(status_code=401, detail="not authenticated")
        return user

    @router.get("/credentials")
    async def list_credentials(user_id: str = Depends(require_user_id)):
        configured = users.list_credential_providers(user_id)
        # Settings is meant to show what is already saved. Keys are never sent back, but the
        # non-secret parts (server URL, Azure endpoint/deployment, AWS region) are, so the
        # form can fill them in instead of looking empty after a reload.
        public: dict[str, dict[str, str]] = {}
        hints: dict[str, str] = {}
        for provider in configured:
            saved = users.get_credential(user_id, provider) or {}
            shown = {k: v for k, v in saved.items() if k in _PUBLIC_ENV}
            if shown:
                public[provider] = shown
            hint = _secret_hint(provider, saved)
            if hint:
                hints[provider] = hint
        return {"configured": configured, "public": public, "hints": hints}

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
        try:   # resolves DNS, which can block: keep it off the event loop
            await asyncio.get_running_loop().run_in_executor(None, validate_credential_urls, payload)
        except UnsafeUrl as e:
            raise HTTPException(status_code=422, detail=str(e)) from None
        users.save_credential(user_id, provider, payload)
        return {"saved": provider}

    @router.get("/credentials/{provider}/models")
    async def list_provider_models(provider: str, user_id: str = Depends(require_user_id)):
        """The models this user's credential (their saved key, else one the
        server has set) can actually use, asked of the provider itself. The key
        stays server-side; only ids and labels are returned."""
        if provider not in supported_providers():
            raise HTTPException(status_code=404, detail=f"{provider!r} has no models to list")
        stored = users.get_credential(user_id, provider)
        # A key the operator set on the server counts too: those users never saved
        # their own, but the provider is still theirs to use.
        server_has_key = any(os.environ.get(var) for var in PROVIDER_FIELDS.get(provider, {}).values())
        if stored is None and not server_has_key:
            raise HTTPException(status_code=404, detail="no stored credential for that provider")
        env = {**os.environ, **(stored or {})}   # the user's own key wins over the server's
        loop = asyncio.get_running_loop()
        try:
            models = await loop.run_in_executor(None, list_models, provider, env)
        except ModelListError as e:
            raise HTTPException(status_code=502, detail=str(e)) from None
        return {"models": models}

    def _tool_env(user_id: str, spec) -> dict[str, str]:
        creds = users.get_credential(user_id, spec.credential_provider) if spec.credential_provider else None
        return {**os.environ, **(creds or {})}

    @router.get("/tools")
    async def list_tools(user_id: str = Depends(require_user_id)):
        """Every tool with this user's state: whether its key is present
        (`configured`, from their saved key or the server's), their own switch
        (`enabled`), and whether it will actually be bound (`active`)."""
        settings = users.get_tool_settings(user_id)
        out = []
        for spec in all_tools():
            configured = spec.available(_tool_env(user_id, spec))
            wanted = settings.get(spec.id, spec.default_enabled)
            out.append({
                "id": spec.id,
                "label": spec.label,
                "description": spec.description,
                "requires_key": spec.requires_key,
                "credential_provider": spec.credential_provider,
                "credential_fields": list(spec.credential_fields),
                "configured": configured,
                "enabled": wanted,
                "active": wanted and configured,
            })
        return {"tools": out}

    @router.put("/tools/{tool_id}")
    async def set_tool(tool_id: str, body: dict, user_id: str = Depends(require_user_id)):
        spec = get_tool(tool_id)
        if spec is None:
            raise HTTPException(status_code=404, detail=f"unknown tool {tool_id!r}")
        enabled = bool(body.get("enabled"))
        if enabled and not spec.available(_tool_env(user_id, spec)):
            raise HTTPException(status_code=409, detail="add its API key first")
        users.set_tool_enabled(user_id, tool_id, enabled)
        return {"id": tool_id, "enabled": enabled}

    @router.delete("/credentials/{provider}")
    async def delete_credential(provider: str, user_id: str = Depends(require_user_id)):
        if not users.delete_credential(user_id, provider):
            raise HTTPException(status_code=404, detail="no stored credential for that provider")
        return {"deleted": provider}

    return router
