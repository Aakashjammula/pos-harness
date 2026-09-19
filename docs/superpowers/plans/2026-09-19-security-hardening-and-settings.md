# Security Hardening and Settings Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the audited security findings (High to Low, except rate limiting and input-size limits, which the user deferred) and add a Settings page where a signed-in user manages their account, security and active sessions.

**Architecture:** Access tokens become bound to a server-side `auth_sessions` row (checked on every request), which makes logout real, gives a list of a user's devices, and enables "sign out everywhere". A first email verification wipes anything an unverified account holder set up (closes pre-hijacking). User-supplied URLs go through one SSRF policy module. Origin checks, security headers, log privacy and fail-fast config validation harden the rest. A tabbed Settings shell hosts Account, Security, Model & voice and Tools.

**Tech Stack:** FastAPI, psycopg, PyJWT, argon2, cryptography (AESGCM); Next.js 16 / React / TypeScript; Docker Compose.

**Spec:** the audit report in the working session (findings H1-H2, M1-M7, L1-L6) plus the user's instructions: *"dont want to keep rate limit and input token limit and all that ... do all h to l security fix ... a setting button and setting page ... user should be able to see all his sessions and only he is allowed there"*. Research inputs: OWASP Session Management, Authentication, SSRF Prevention and Email Validation cheat sheets; account-settings UX guidance (profile, security, privacy, separate danger zone, per-section save).

## Global Constraints

- **Out of scope (user deferred):** rate limiting / lockout, input-size and token limits, concurrency caps. Do not add them.
- Backend tests only against a `*_test` database (`tests/conftest.py` enforces it). Never run tests against `pos`.
- Live checks may create ONE throwaway `@example.com` account in the real DB and must delete it by email.
- Secrets travel in headers or the credential store, never in URLs, logs or error text.
- Backend lint: `uv run ruff check src tests` clean (line length 120). Frontend: `npx tsc --noEmit` and `npx eslint src` clean; no `setState` synchronously inside effects (derive state instead).
- Commit trailer: `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`. Work on branch `feat/security-and-settings`.
- Existing behaviours that must keep working: refresh-token rotation with replay detection, magic links, per-user isolation of sessions/credentials/tools.

## Decisions (recorded so they are not re-litigated)

1. **H1:** password signup stays usable immediately, but the first time the address is verified (magic link) an *unverified* account is reset: password cleared, all login sessions revoked, saved credentials, tool settings and chats deleted. Verified accounts are untouched.
2. **H2:** only `http`/`https`; always block link-local/metadata (169.254.0.0/16, fe80::/10), unspecified and multicast; private/loopback only when `ALLOW_PRIVATE_LLM_URLS=true` (default false; `.env.example` and the local `.env` set it true for LM Studio). Azure endpoint must be public `https`. No redirects. DNS is resolved and checked at save time and again at use time (rebinding TOCTOU is documented, not fully closed).
3. **Signup email enumeration** (409 "already registered") is an accepted UX trade-off and is left as is; login *timing* is fixed.
4. **Login sessions** (devices) are what the Settings "sessions" list shows; chat history stays in the sidebar.
5. No `__Host-` cookie prefix (the frontend never reads cookies, but tests and the CLI name them); `Secure` is enforced by config validation instead.

## File Structure

- `backend/src/pos/settings_check.py` — startup validation (Task 1)
- `backend/src/pos/net_policy.py` — SSRF policy (Task 5)
- `backend/src/pos/auth/deps.py` — `Auth` class replacing module functions (Task 2)
- `backend/src/pos/auth/store.py`, `db.py` — sessions, verification, profile, deletion (Tasks 2-4, 11)
- `backend/src/pos/auth/routes.py` — sessions/password/profile/export/delete routes (Tasks 2-4, 11)
- `backend/src/pos/http_security.py` — origin check, headers, validation-error sanitiser (Tasks 6-7)
- `frontend/src/components/SettingsShell.tsx`, `AccountTab.tsx`, `SecurityTab.tsx`, `GuestGuard.tsx`, `lib/account.ts`, `hooks/useAccount.ts` (Tasks 10, 12)

---

### Task 1: Config and startup validation (L5, L3 settings)
**Files:** Create `backend/src/pos/settings_check.py`, `backend/tests/test_settings_check.py`; modify `backend/src/pos/config.py`, `backend/src/pos/cli/server.py` (`run()`).
**Interfaces — Produces:** `config.ALLOW_PRIVATE_LLM_URLS: bool`, `config.ENABLE_API_DOCS: bool`, `config.LOG_CONVERSATIONS: bool`; `check_settings(env: Mapping[str,str]) -> list[str]` returns fatal problems; `run()` exits non-zero listing them.
Rules: `JWT_SECRET` >= 32 chars; `ENCRYPTION_KEY` base64 decoding to exactly 32 bytes; if any `CORS_ORIGINS` entry starts with `https://` then `COOKIE_SECURE` must be true (fatal); if an origin is `http://` and not localhost/127.0.0.1 and `COOKIE_SECURE` is false → warning printed, not fatal.
- [ ] Failing tests for each rule (good config → `[]`; short secret; bad key length; https origin without Secure). - [ ] Implement, wire into `run()` before `create_app`. - [ ] Suite green. - [ ] Commit.

### Task 2: Server-side login sessions (M5) and the `Auth` object
**Files:** `db.py` (`auth_sessions` table; `ALTER TABLE refresh_tokens ADD COLUMN IF NOT EXISTS auth_session_id UUID REFERENCES auth_sessions(id) ON DELETE CASCADE`), `store.py`, `tokens.py` (`create_access_token(user_id, sid)`, `decode_access_token -> (user_id, sid) | None`), `deps.py`, `routes.py`, `cli/server.py`; tests `test_auth_sessions.py`.
**Interfaces — Produces:**
`UserStore.create_auth_session(user_id, user_agent, ip) -> str`, `.auth_session_active(session_id, user_id) -> bool`, `.touch_auth_session(session_id)`, `.list_auth_sessions(user_id) -> list[dict]` (id, created_at, last_seen_at, user_agent, ip), `.revoke_auth_session(user_id, session_id) -> bool`, `.revoke_all_auth_sessions(user_id, except_id=None) -> int`.
`Auth(users)` with `user_id_from_request(conn) -> str|None`, `session_id_from_request(conn) -> str|None`, `async require_user_id(request) -> str`. `build_auth_router(users, auth=None)`; `create_app` builds one `Auth` and uses `Depends(auth.require_user_id)` and `auth.user_id_from_request(websocket)`.
Routes: `GET /auth/sessions` (adds `current: bool`), `DELETE /auth/sessions/{id}` (own only, else 404), `POST /auth/sessions/revoke-others`. Login, signup, magic-link verify create a session (user agent from header, IP from `request.client.host`); the JWT carries `sid`; refresh keeps the same session, updates `last_seen_at`, and rejects a revoked session; logout revokes the current session; replay detection revokes all. An access token without `sid` (issued before this change) is rejected; a refresh token row without a session gets one created on first refresh.
- [ ] Tests: revoked session's access token is 401 immediately; logout kills the access token; list shows only own sessions (user B cannot list or revoke A's → 404); revoke-others keeps current; refresh keeps the session id; legacy refresh row upgrades. - [ ] Implement. - [ ] Whole suite (update tests that decode tokens). - [ ] Commit.

### Task 3: Email verification and pre-hijack fix (H1)
**Files:** `db.py` (`ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified_at TIMESTAMPTZ`), `store.py`, `routes.py`; tests `test_email_verification.py`.
**Interfaces — Produces:** `UserStore.mark_email_verified(user_id) -> bool` (True if it changed), `.reset_unverified_account(user_id)` (clear password, revoke all auth sessions/refresh tokens, delete `api_credentials`, `user_tool_settings`, `sessions`), user dicts gain `email_verified: bool`; `POST /auth/verify-email/request` (auth required) re-sends the link.
Behaviour: `verify_magic_link`: existing unverified user → `reset_unverified_account` then mark verified then issue a fresh session; existing verified → just sign in; new email → create already-verified. A magic-link-only signup is verified when its link is consumed. `/auth/me` returns `email_verified`.
- [ ] Failing test of the attack: signup(a@x, password P) + a saved credential + a chat → magic link for a@x consumed → password P no longer logs in, the old session cookie is 401, the credential and chat are gone, the new session works. - [ ] Test verified accounts keep everything. - [ ] Implement. - [ ] Suite. - [ ] Commit.

### Task 4: Password change and constant-time login (M2 timing, M5)
**Files:** `routes.py`, `passwords.py`; tests `test_password.py`.
**Interfaces — Produces:** `PUT /auth/password` body `{current_password?: str, new_password: str (>=8)}`: passwordless accounts may set one without `current_password`; accounts with a password must supply it (wrong → 401); on success all *other* sessions are revoked. `verify_password` unknown/passwordless path runs `_hasher.verify(_DUMMY_HASH, password)` so login time no longer reveals whether an email exists.
- [ ] Tests (set, change, wrong current 401, other sessions revoked, dummy hash used for unknown user via monkeypatch spy). - [ ] Implement. - [ ] Commit.

### Task 5: SSRF policy (H2)
**Files:** Create `backend/src/pos/net_policy.py`, `tests/test_net_policy.py`; modify `auth/routes.py` (validate on save), `llm/model_listing.py` (validate + `allow_redirects=False`), `llm/providers/local.py` (validate in `resolve`, pass `http_client=httpx.Client(follow_redirects=False)`), `llm/providers/azure.py` (public https only).
**Interfaces — Produces:** `check_url(url: str, *, allow_private: bool, https_only: bool = False) -> str` returns the normalised URL or raises `UnsafeUrl(reason)`; resolves the host (all A/AAAA records) and rejects if ANY address is blocked. `PUT /credentials/local` and `/azure` return 422 with the reason.
Tests use IP literals and a monkeypatched resolver: `169.254.169.254` always blocked; `127.0.0.1`, `10.0.0.5`, `192.168.1.5`, `[::1]` blocked unless `allow_private`; a hostname resolving to a private IP blocked; `file://`, `ftp://`, missing host blocked; a public IP allowed; a redirecting server is not followed (listing returns the 3xx as an error).
- [ ] Failing tests. - [ ] Implement. - [ ] Update `.env.example` (`ALLOW_PRIVATE_LLM_URLS=true` with an explanatory comment) and set it in the local `.env`; compose passes it through. - [ ] Suite. - [ ] Commit.

### Task 6: Origin checks (M4)
**Files:** Create `backend/src/pos/http_security.py`; modify `cli/server.py`; tests `test_origin_check.py`.
Behaviour: a middleware rejects (403) any non-GET/HEAD/OPTIONS request whose `Origin` header is present and not in `CORS_ORIGINS`; `/ws` closes with 1008 before `accept()` when `Origin` is present and not allowed. A request with no `Origin` (CLI, curl, tests) passes.
- [ ] Tests: foreign-origin POST 403, allowed-origin POST ok, no-origin ok, foreign-origin WS refused, allowed WS ok. - [ ] Implement. - [ ] Commit.

### Task 7: Security headers, docs, validation errors, cache (M3, L3, L6)
**Files:** `http_security.py`, `cli/server.py`; tests `test_security_headers.py`.
Every API response gets `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, `X-Frame-Options: DENY`, `Content-Security-Policy: default-src 'none'; frame-ancestors 'none'`, and `Cache-Control: no-store` for `/auth/*`, `/credentials*`, `/tools*`, `/sessions*`; `Strict-Transport-Security` only when `COOKIE_SECURE`. `FastAPI(docs_url=None, redoc_url=None, openapi_url=None)` unless `ENABLE_API_DOCS`. A `RequestValidationError` handler returns `{"detail": [{"loc","msg","type"}]}` with **no `input`/`ctx`** (so a rejected password is never echoed).
- [ ] Tests (headers present, docs 404 by default and 200 when enabled, 422 body contains no submitted value). - [ ] Implement; check the frontend still reads `detail` strings it relies on. - [ ] Commit.

### Task 8: Conversation logging off by default (L4)
**Files:** `config.py`, `agent.py`; tests `test_agent.py` (capsys).
Every `print` in `agent.py` that includes user/bot text (`USER:`, `BOT:`, `[LLM] triggered "…"`, `[STT] dropped … text=`, `[TTS] triggered … "preview"`) prints only the metadata (lengths/timings) unless `LOG_CONVERSATIONS` is true.
- [ ] Test that with the default a distinctive phrase never reaches stdout and with `LOG_CONVERSATIONS=true` it does. - [ ] Implement. - [ ] Commit.

### Task 9: Credential encryption hardening (L5)
**Files:** `auth/crypto.py`, `auth/store.py`; tests `test_crypto.py`.
`encrypt_payload(payload, aad: bytes)` / `decrypt_payload(ct, nonce, aad)` bind ciphertext to `f"{user_id}:{provider}"`; decryption tries the AAD form, then legacy no-AAD (rows written before this change), and re-encrypts legacy rows with AAD on read. `ENCRYPTION_KEY_PREVIOUS` (comma-separated) is tried for decryption only, so a key can be rotated.
- [ ] Tests: round trip; a row copied to another user/provider fails to decrypt; legacy row still readable then upgraded; previous key decrypts; wrong key raises. - [ ] Implement. - [ ] Commit.

### Task 10: Frontend security (M3, M6)
**Files:** `frontend/next.config.ts`, create `components/GuestGuard.tsx`, modify `app/login/page.tsx`, `app/signup/page.tsx`, `components/AuthGuard.tsx`, sign-out handler.
`next.config.ts` `headers()` for all routes: CSP (`default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self' <API origin> <WS origin>; frame-ancestors 'none'; base-uri 'self'; form-action 'self'; object-src 'none'` with the origins taken from `NEXT_PUBLIC_API_URL`/`NEXT_PUBLIC_WS_URL`), `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: strict-origin-when-cross-origin` (`no-referrer` for `/magic-link`), `Permissions-Policy: microphone=(self), camera=(), geolocation=()`, `Strict-Transport-Security` in production.
`GuestGuard`: on `/login` and `/signup`, `fetchMe()`; a signed-in user is `router.replace("/")`; never wraps `/magic-link`. `AuthGuard` re-checks on `pageshow` with `event.persisted` (back/forward cache) and after sign-out uses `router.replace("/login")`.
- [ ] Implement. - [ ] `tsc`/`eslint`. - [ ] Verify headers on the built container with `curl -I`. - [ ] Commit.

### Task 11: Account endpoints (Settings backend)
**Files:** `routes.py`, `store.py`; tests `test_account.py`.
`PATCH /auth/me` `{name?, username?}` (same validation as signup; username taken → 409); `GET /auth/export` returns `{"user": {...}, "sessions": [{... "turns": [...]}]}` (never credentials, never the password hash; `Content-Disposition: attachment`); `DELETE /auth/account` body `{confirm_email: str, password?: str}` — the email must match; if the account has a password it must be supplied and correct; deletes the user (FKs cascade) and clears the cookies.
- [ ] Tests incl. isolation (export contains only the caller's chats) and the wrong-confirmation cases. - [ ] Implement. - [ ] Commit.

### Task 12: Settings shell and tabs (frontend)
**Files:** Create `SettingsShell.tsx`, `AccountTab.tsx`, `SecurityTab.tsx`, `lib/account.ts`, `hooks/useAccount.ts`; modify `SettingsPanel.tsx` and `ToolsPanel.tsx` (add `embedded?: boolean` to drop their own header/`h-screen`), `Sidebar.tsx` (Settings button beside the email/Log out), `ChatPanel.tsx` (model chip → `#/settings/model`, Tools → `#/settings/tools`), `app/page.tsx` (routes `#/settings`, `#/settings/account|security|model|tools`; `#/tools` maps to the tools tab).
Tabs: **Account** (name, username, email + verified badge and "Send verification email", per-section Save, "Export my data", danger zone with typed-email confirmation and password if set) · **Security** (set/change password; **Active sessions**: browser/OS parsed from the user agent, IP, last active, "This device" badge, per-row Revoke, "Sign out other devices") · **Model & voice** (existing) · **Tools** (existing).
- [ ] Implement with the derive-don't-set-state hook pattern. - [ ] `tsc`/`eslint`. - [ ] Commit.

### Task 13: Dependencies, images and compose (M7, L3, L6)
**Files:** `frontend/Dockerfile` (`node:20-alpine` → `node:24-alpine`, all three stages), `backend/pyproject.toml`/`uv.lock` (`langchain>=1.3.9`), `backend/src/pos/tts/kokoro.py` (`revision=` pin from the cached snapshot `adf39fcf901ff5f6bf576421e1570114a78669a4`), `docker-compose.yml` (`${BIND_ADDRESS:-127.0.0.1}:8000:8000` and `:3000:3000`; pass `ALLOW_PRIVATE_LLM_URLS`, `ENABLE_API_DOCS`, `LOG_CONVERSATIONS`), `.dockerignore` (`.env`, `.env.*` except `.env.example`), `.env.example` (comment that the default Postgres password is for loopback dev only).
- [ ] `uv lock --upgrade-package langchain` and full suite green. - [ ] Frontend builds under Node 24. - [ ] Re-run `pip-audit` (expect langchain cleared; nltk remains, documented). - [ ] Commit.

### Task 14: Docs
**Files:** `README.md`. Document the new env vars, sessions/Settings, the SSRF policy and its DNS-rebinding limit, the accepted signup-enumeration trade-off, and what is deliberately not done (rate/size limits).

### Task 15: Verify, gate, finish
- [ ] Rebuild both images; live checks with one throwaway account covering: login pages redirect when signed in (HTTP-level: `/auth/me`), sessions list/revoke/kill-token, logout invalidates the access token, pre-hijack scenario, SSRF blocked for `169.254.169.254` and (with the flag off) `127.0.0.1`, origin check on REST and WS, headers on API and frontend, docs off, 422 has no echoed input, timing gap closed.
- [ ] Full backend suite, ruff, tsc, eslint green; `git status` clean.
- [ ] State plainly what was not visually verified (no browser tool).
- [ ] Use superpowers:finishing-a-development-branch.

---

## Self-Review

- **Coverage:** H1 → T3; H2 → T5; M2 timing → T4 (signup enumeration recorded as accepted); M3 → T7 + T10; M4 → T6; M5 → T1/T2/T4; M6 → T10; M7 → T13; L3 → T7/T13; L4 → T8; L5 → T1/T9; L6 → T7/T13; Settings page + sessions list → T2, T11, T12. Rate limits and size limits intentionally absent.
- **Placeholders:** file lists, interfaces, rules and test cases are concrete; exact code is written per task while implementing because each task edits code that must be read first.
- **Consistency:** `Auth`, `auth_sessions`, `sid`, `email_verified`, and the route paths are named identically wherever used.
