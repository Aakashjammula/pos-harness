# User accounts, auth, and per-user encrypted API keys

## Goal

Today the app is single-tenant: any browser can list/resume/delete any
session (`GET /sessions`, `DELETE /sessions/{id}` have no ownership
check), and provider API keys are typed in fresh every connection,
held in an in-memory 60-second single-use token, and thrown away
(`POST /session-keys`, see `server.py`'s docstring). This spec adds
real user accounts so that:

- Each user only sees and can act on their own chat sessions.
- Each user can save their provider API keys once (encrypted at
  rest) instead of retyping them every time they connect.
- The system is ready to scale to many concurrent users without a
  redesign (standard normalized schema, no per-tenant schema/DB
  sprawl).

## Non-goals

- No OAuth/SSO in this pass — email+password only (the user
  explicitly chose this over OAuth or "both").
- No admin/roles/permissions system — every authenticated user has
  the same capabilities, scoped to their own data.
- No org/team/workspace concept — one user owns their own sessions
  and credentials directly, not through an intermediate tenant entity.
  (This is a multi-*user* app, not a multi-*tenant* SaaS — see
  "Why a shared schema, not multi-tenancy" below.)
- Not migrating existing session/turn rows to a user — there is no
  real user data in this environment yet (confirmed with the user
  earlier in this project). New rows require a `user_id`; if any
  pre-auth rows exist in a dev database they're simply orphaned/unused
  going forward, not backfilled.

## Auth design

**Password hashing: Argon2id.** OWASP's Password Storage Cheat
Sheet lists Argon2id as its primary recommendation over bcrypt —
it's memory-hard (resists GPU/ASIC cracking better) and exposes two
independently tunable costs (time, memory) instead of bcrypt's single
exponential factor. Minimum params per OWASP: memory=19 MiB,
iterations=2, parallelism=1 (`argon2-cffi`'s own defaults meet or
exceed this). No bcrypt fallback needed — this is a fresh user table,
not a migration from an existing bcrypt store.

**Tokens: short-lived JWT access token + long-lived refresh token.**
- Access token: ~15 minutes, signed with `JWT_SECRET` (env var, never
  hardcoded), delivered as an **httpOnly, SameSite=Strict, Secure**
  cookie — never `localStorage`/`sessionStorage`, which is readable by
  any injected script (XSS). This is the standard 2026 guidance across
  every FastAPI+Next.js auth writeup reviewed.
- Refresh token: long-lived (e.g. 30 days), also httpOnly cookie.
  Stored **hashed** (SHA-256) in the `refresh_tokens` table so it can
  be looked up and revoked (logout, "log out everywhere") without
  ever keeping the raw token server-side — same principle as never
  storing plaintext passwords.
- `POST /auth/refresh` exchanges a valid refresh token for a new
  access token (and rotates the refresh token — one-time-use, old one
  marked `revoked_at`, standard refresh-token-rotation practice to
  detect token theft).
- Every existing endpoint (`/sessions`, `/sessions/{id}`, `/ws`, the
  new credentials endpoints) resolves the current user from the
  verified access token's claim — **never** from a `user_id` in the
  request body/query string, per OWASP's tenant-isolation guidance
  ("resolve identity at the API boundary from a non-spoofable
  source").

## Why a shared schema, not multi-tenancy

Multi-tenant SaaS design (schema-per-tenant, DB-per-tenant, or
row-level tenant isolation with `tenant_id`) solves a different
problem: isolating *organizations* that might each have many users,
compliance/data-residency requirements, or need to move to dedicated
infrastructure. This app has one flat pool of individual users, each
owning their own data directly — the standard, lowest-overhead
pattern for that (confirmed against current Postgres multi-tenancy
guidance) is a **shared schema with a `user_id` foreign key on every
owned table**, which is what's below. If org/team support is ever
needed later, a `user_id` column is trivially promotable to
`owner_id` on a join table — nothing here forecloses that.

## Why not LangChain/LangGraph's Postgres checkpointer

LangChain's `PostgresSaver`/`AsyncPostgresSaver` (LangGraph's
checkpointer) persists full agent **graph execution state** per
`thread_id`/`checkpoint_id` — it's designed for resuming a
`create_agent` graph mid-run, not for a chat-history UI (session
list, per-turn display, titles, deletion). This codebase already
made and documented this exact call: `LangChainLlm` hand-rolls its
own tool-execution loop specifically to avoid `create_agent`'s
`AgentState` + checkpointer, "which would duplicate the history
`Agent` already tracks in `self.conversation`" (see
`backend/src/pos/llm/langchain_llm.py` and the root README's "LLM:
LangChain + tool calling" section). The existing `sessions`/`turns`
tables are the intentional lightweight alternative — this spec keeps
that model and only adds `user_id` scoping to it, rather than
introducing a second, differently-shaped persistence mechanism.

## Database schema

```
┌───────────────────────────────┐
│ users                         │
├───────────────────────────────┤
│ id             UUID PK        │
│ email          TEXT UNIQUE NN │
│ password_hash  TEXT NN        │
│ created_at     TIMESTAMPTZ NN │
│ updated_at     TIMESTAMPTZ NN │
└───────────────────────────────┘
        │ 1
        ├───────────────────────────────────────────┐
        │ N                                          │ N
┌───────────────────────────────┐   ┌───────────────────────────────┐
│ refresh_tokens                │   │ api_credentials                │
├───────────────────────────────┤   ├───────────────────────────────┤
│ id             UUID PK        │   │ id             UUID PK         │
│ user_id        FK -> users NN │   │ user_id        FK -> users NN  │
│ token_hash     TEXT NN        │   │ provider       TEXT NN         │
│ expires_at     TIMESTAMPTZ NN │   │ encrypted_payload BYTEA NN     │
│ revoked_at     TIMESTAMPTZ    │   │ nonce          BYTEA NN        │
│ created_at     TIMESTAMPTZ NN │   │ created_at     TIMESTAMPTZ NN  │
└───────────────────────────────┘   │ updated_at     TIMESTAMPTZ NN  │
                                     │ UNIQUE (user_id, provider)     │
        │ 1                         └───────────────────────────────┘
        │
        │ N
┌───────────────────────────────┐
│ sessions                      │  (existing table + 1 new column)
├───────────────────────────────┤
│ id             TEXT PK        │
│ user_id        FK -> users NN │  <- NEW
│ created_at     TIMESTAMPTZ NN │
│ mode           TEXT NN        │
│ tts_engine     TEXT           │
│ llm_model      TEXT NN        │
│ title          TEXT           │
└───────────────────────────────┘
        │ 1
        │
        │ N
┌───────────────────────────────┐
│ turns                         │  (unchanged — already scoped
├───────────────────────────────┤   transitively via session_id)
│ id             BIGSERIAL PK   │
│ session_id     FK -> sessions │
│ role           TEXT NN        │
│ text           TEXT NN        │
│ usage_json     JSONB          │
│ created_at     TIMESTAMPTZ NN │
└───────────────────────────────┘
```

**Normalization notes:**
- `users` is 3NF: no column depends on anything but the key.
- `api_credentials` stores one row per `(user_id, provider)`, not one
  row per field (e.g. Azure's key+endpoint+deployment). Splitting to
  one-row-per-field would be "more normalized" on paper but those
  fields are never read/written independently — they're one atomic
  credential set per provider, decrypted together into one env-var
  dict (see `server.py`'s existing `session_keys` handler, which
  already groups them this way). `encrypted_payload` holds an
  AES-GCM ciphertext of that provider's JSON fields; `nonce` is the
  per-encryption random IV (GCM requires a unique nonce per
  encryption with the same key — stored alongside, it's not secret).
- `turns` needs no `user_id` of its own — ownership is transitive
  through `session_id → sessions.user_id`, adding a duplicate column
  would just be a denormalized cache with no read pattern that needs
  it (every turn is always fetched by session, never queried
  cross-session by user directly).
- `refresh_tokens.token_hash` stores `sha256(raw_token)`, never the
  raw token — same reasoning as `users.password_hash`: a DB leak
  shouldn't hand out usable credentials.

## Encryption for `api_credentials`

- `ENCRYPTION_KEY` env var: 32 random bytes (base64-encoded in the
  env, decoded at startup), separate from `JWT_SECRET`.
- On save: serialize that provider's fields to JSON, encrypt with
  AES-256-GCM using `ENCRYPTION_KEY` and a fresh random 12-byte nonce,
  store ciphertext + nonce.
- On use (building the LLM client for a `/ws` connection): decrypt
  in-memory only, merge into the env-overrides dict exactly like
  today's `key_token` flow already does — nothing about how the
  decrypted values reach `LangChainLlm`/provider classes changes.
- Losing `ENCRYPTION_KEY` makes all stored credentials
  unrecoverable (expected/acceptable — same tradeoff as any envelope
  encryption scheme; back it up like any other production secret).

## API changes

New:
- `POST /auth/signup` — email + password, returns the same cookies as login.
- `POST /auth/login` — sets access + refresh cookies.
- `POST /auth/logout` — revokes the refresh token, clears cookies.
- `POST /auth/refresh` — rotates the refresh token, issues a new access token.
- `GET /credentials` — list saved providers (which are configured, never the secret values).
- `PUT /credentials/{provider}` — save/update that provider's encrypted fields.
- `DELETE /credentials/{provider}` — remove a saved credential.

Changed:
- `GET /sessions`, `GET/DELETE /sessions/{id}` — filtered/scoped to
  the authenticated user; 404 (not 403) for another user's session id,
  to avoid confirming it exists.
- `WS /ws` — requires a valid access token (cookie sent automatically
  by the browser on the same origin); new sessions are stamped with
  the connecting user's id. `key_token` query param and
  `POST /session-keys` are removed — replaced by reading the user's
  saved `api_credentials` server-side instead of a per-connection
  typed override. (A user can still leave a provider's credentials
  unset to fall back to the server's own environment defaults, same
  as today's "leave blank" behavior.)
- `GET /options` — unchanged (not user-specific: engine/voice/model
  lists are global).

## Frontend changes

- New `/login` and `/signup` pages.
- A root layout auth guard: unauthenticated users are redirected to
  `/login`; the existing chat UI mounts only once authenticated.
- Settings panel: the per-provider fields gain a real **Save**
  button (`PUT /credentials/{provider}`) and a **Remove** button,
  replacing the current "retype every connect, never persisted"
  copy/behavior. A saved provider shows as configured without the
  raw value ever being sent back down.
- A simple account menu (email + Logout) somewhere in the shell
  (sidebar footer is the natural spot, matching the ChatGPT-style
  layout already in place).

## Testing

- `backend/tests/test_auth.py`: signup/login/refresh/logout, wrong
  password, duplicate email, expired/invalid tokens, refresh-token
  rotation and revocation.
- `backend/tests/test_credentials.py`: save/list/delete, encryption
  round-trip, one user can never read another's credentials.
- `backend/tests/test_server.py`: extend existing session-listing
  tests to assert cross-user isolation (user A never sees user B's
  sessions).
- Frontend: manual verification (no test framework added, per this
  project's existing YAGNI stance — matches how the rest of the
  frontend was scoped).

## Migration path

Since there's no real data to preserve, the Postgres schema changes
land as new `CREATE TABLE`/`ALTER TABLE ... ADD COLUMN` statements in
`storage.py`'s existing startup migration block (same pattern already
used there for the `title` column), not a separate migration tool —
consistent with how this module already handles schema evolution.
`sessions.user_id` is added `NOT NULL` with no default, which is safe
here specifically because the table is expected to be empty at
upgrade time in every environment this runs in today.
