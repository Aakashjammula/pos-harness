# Multi-user platform: accounts, encrypted credentials, context compaction, guardrails

## Scope of this document

This is the architecture document for turning `pos-harness` from a
single-tenant local tool into a multi-user application. It covers five
phases. Phases 1-3 are meant to be executed as one implementation plan;
phases 4 and 5 each get their own plan later, but their **database
requirements are included here on purpose** so the schema is built once
and doesn't need a painful migration when they land.

| Phase | What | Plan |
|---|---|---|
| 1 | Storage foundation (connection pool, indexes, schema) | first plan |
| 2 | Auth (accounts, login, sessions scoped per user) | first plan |
| 3 | Per-user encrypted API credentials | first plan |
| 4 | Context compaction (replace `HISTORY_TURNS=3` truncation) | later |
| 5 | Guardrails (PII detection, per-user call limits) | later |

## Goal

Today the app is single-tenant: any browser can list, resume, or delete
any session (`GET /sessions`, `DELETE /sessions/{id}` have no ownership
check), and provider API keys are typed in fresh every connection, held
in an in-memory 60-second single-use token, then thrown away
(`POST /session-keys`, see `server.py`'s module docstring). This
document defines the path to:

- Each user seeing and acting only on their own chat sessions.
- Each user saving provider API keys once, encrypted at rest.
- The bot remembering a whole conversation instead of the last three
  exchanges, without unbounded token cost.
- Per-user safety and spend controls (PII handling, call limits) that
  the user can tune from the UI.
- A storage layer that doesn't serialize every database call behind one
  connection.

## Non-goals

- No OAuth/SSO in this pass — email+password only (explicitly chosen
  over OAuth or both).
- No admin/roles/permissions system — every authenticated user has the
  same capabilities, scoped to their own data.
- No org/team/workspace entity — a user owns their data directly. See
  "Why a shared schema, not multi-tenancy".
- No data migration of existing sessions/turns — there is no real user
  data in any environment this runs in today. New rows require a
  `user_id`; any pre-auth rows are left orphaned, not backfilled.
- Not adopting LangGraph / `create_agent`. See "Why we keep the
  hand-rolled agent loop" — this decision is revisited in every phase
  below and the answer stays the same for the same reason.

---

## Phase 1: Storage foundation

### 1a. Connection pooling (the real scalability ceiling)

`storage.py` currently holds **one** `psycopg.connect()` for the whole
process and wraps every single method — `create_session`, `add_turn`,
`set_title`, `list_sessions`, `get_session`, `delete_session` — in
`with self._lock:` on one `threading.Lock`. Every database call from
every concurrent WebSocket session is therefore serialized: one
statement at a time, process-wide.

That design was inherited from the SQLite version, where it was
correct (SQLite does not do concurrent writers well). On Postgres it is
purely a bottleneck — Postgres is built for concurrent connections, and
this pattern means adding user accounts would not actually buy
concurrency. With N simultaneous voice sessions, every turn insert,
title update, and sidebar load queues behind the same mutex.

**Change:** replace the single connection + lock with
`psycopg_pool.ConnectionPool` (add the `psycopg-pool` dependency). Each
call borrows a connection, uses it, returns it. The `threading.Lock`
disappears entirely — the pool is already thread-safe, and there is no
shared mutable state left to guard.

Pool sizing: `min_size=2`, `max_size` from a `DB_POOL_MAX_SIZE` env var
defaulting to 10. Keep it well under Postgres' own `max_connections`
(100 by default) since each app instance holds its own pool.

### 1b. Indexes

- `sessions(user_id, created_at DESC)` — the sidebar query is "this
  user's sessions, newest first"; without this composite index it
  degrades as the table grows across all users.
- `refresh_tokens(expires_at)` — so a future cleanup job can find
  expired rows without a full scan.
- `usage_events(user_id, kind, created_at DESC)` — the rate-limit check
  runs on the hot path (see Phase 5).
- `session_summaries(session_id, created_at DESC)` — "latest summary
  for this session" is read at the start of every turn.
- `turns(session_id)` already exists and stays.

### 1c. Schema created up front

All tables and columns in the "Database schema" section below are
created in Phase 1, including the ones only Phase 4 and 5 populate
(`turns.token_count`, `turns.pii_flags`, `session_summaries`,
`usage_events`, `user_settings`). They are nullable/empty until those
phases land. Creating them now means one schema pass instead of three.

---

## Phase 2: Auth

**Password hashing: Argon2id.** OWASP's Password Storage Cheat Sheet
lists Argon2id first (memory-hard, resists GPU/ASIC cracking better than
bcrypt, and exposes independently tunable time and memory costs instead
of bcrypt's single factor). OWASP minimum: memory 19 MiB, iterations 2,
parallelism 1 — `argon2-cffi`'s defaults meet or exceed this. No bcrypt
fallback path is needed; this is a fresh user table, not a migration.

**Tokens: short-lived JWT access token + rotating refresh token.**

- Access token: 15 minutes, signed with `JWT_SECRET` (env var, never
  hardcoded), delivered as an **httpOnly, SameSite=Strict, Secure**
  cookie. Never `localStorage`/`sessionStorage` — those are readable by
  any injected script.
- Refresh token: 30 days, also an httpOnly cookie, stored **hashed**
  (SHA-256) in `refresh_tokens` so it can be revoked without the server
  ever holding a usable copy — same principle as `password_hash`.
- `POST /auth/refresh` rotates: the presented refresh token is marked
  `revoked_at` and a new one issued. Reuse of an already-revoked token
  is the standard signal of token theft — log it and revoke that user's
  whole token family.
- Every protected endpoint resolves the user from the **verified access
  token claim only**, never from a `user_id` in a request body or query
  string. This is the specific mistake OWASP's tenant-isolation guidance
  calls out.

**Session ownership.** `sessions.user_id` is set from the authenticated
user at creation. `GET /sessions` filters by it. `GET`/`DELETE
/sessions/{id}` return **404, not 403**, for a session belonging to
someone else — a 403 confirms the id exists.

---

## Phase 3: Per-user encrypted API credentials

Replaces the current ephemeral `POST /session-keys` + `key_token` flow.

- `ENCRYPTION_KEY` env var: 32 random bytes, base64-encoded in the
  environment, decoded at startup. Separate from `JWT_SECRET`.
- On save: serialize that provider's fields to JSON, encrypt with
  **AES-256-GCM** using a fresh random 12-byte nonce, store ciphertext
  and nonce. GCM requires a unique nonce per encryption under the same
  key; the nonce is not secret and is stored alongside.
- On use: decrypt in memory only when building the LLM client for a
  connection, and merge into the env-override dict exactly the way the
  existing `key_token` path already does. Nothing downstream of
  `LangChainLlm` changes.
- The API never returns a stored secret. `GET /credentials` returns
  only which providers are configured.
- Losing `ENCRYPTION_KEY` makes stored credentials unrecoverable. That
  is the expected tradeoff for envelope encryption — back it up like
  any other production secret.

**Why persisted-and-encrypted rather than the current never-stored
model:** this matches what comparable products do (LibreChat encrypts
per-user keys server-side; Open WebUI relies on database-level
encryption). The current zero-persistence design is stricter but only
works for a single local user; it does not survive the move to
accounts, where retyping every key every connection is the wrong
tradeoff.

---

## Phase 4: Context compaction

### The actual problem

`config.HISTORY_TURNS = 3` means `Agent` sends only the last three
user/assistant turn-pairs to the LLM. The bot forgets everything older.
The problem to solve is **not** context overflow — it is premature
amnesia, plus the cost of naively fixing it by sending everything.

### How the three reference implementations do it

| | Trigger | Mechanism |
|---|---|---|
| **Anthropic** (server-side, beta `compact-2026-01-12`) | `trigger: {type: "input_tokens", value: 150000}`, minimum 50,000 | API emits a `compaction` block; all blocks before it are dropped on subsequent requests. `pause_after_compaction` allows injecting content first. Compaction's own token cost appears in `usage.iterations`, **not** in top-level `input_tokens` |
| **OpenAI** (Agents SDK / Responses API) | `StaticCompactionPolicy(threshold=8000)`, auto, or forced `run_compaction({"force": True})` | Summarize into a structured carry-forward. Explicit guidance: *compact at meaningful workflow boundaries, not after every turn* |
| **LangChain** (`SummarizationMiddleware`) | `trigger=("tokens", N)` / `("messages", N)` / `("fraction", 0.8)`, AND/OR combinations | Summarize old, `keep=("messages", 20)` verbatim, keep AI/Tool message pairs together, retry 3× via `Runnable.with_retry`, and **never** substitute a synthetic summary on failure |

All three agree on the shape: **summary of old + N recent verbatim**,
triggered on tokens rather than turn count, with compaction's own cost
accounted for separately.

### Why none of them can be used directly here

1. **Provider-native compaction covers 2 of 7 providers and not the
   default one.** Anthropic's works only on Claude models, OpenAI's only
   on OpenAI. This app's default provider is local LM Studio, with
   seven providers behind `LlmBase`. Application-level compaction is the
   only mechanism that works for all of them. Anthropic passthrough
   remains possible later as a per-provider optimization, not as the
   foundation.
2. **`SummarizationMiddleware` requires `create_agent()`.** Every
   LangChain middleware does. See "Why we keep the hand-rolled agent
   loop".

So: port the *design*, not the code.

### Design

**Defaults follow LangChain's**, and every one of them is
user-overridable via `user_settings` (NULL there = use the global
default from `config.py`):

| Setting | Default | Source |
|---|---|---|
| `compact_trigger_fraction` | `0.8` | LangChain's `("fraction", 0.8)` example; Deep Agents uses 0.85 |
| `compact_keep_messages` | `20` | LangChain's `keep=("messages", 20)` default |

- **Trigger on a fraction of the model's context window**, not a fixed
  token count — the window varies enormously across the seven providers
  (the app already reads it live for the local backend and has a table +
  env override for the rest, exposed as `usage["context_window"]`).
- **Keep the last `compact_keep_messages` messages verbatim**, summarize
  everything older into a rolling summary.
- **Never fabricate a summary.** If the summarization call fails after
  retries, fall back to the current truncation behavior and log it —
  silently inventing a summary corrupts the conversation.

**Compaction runs off the turn path.** This is the constraint that
makes this project different from every reference implementation above.
TTFA is currently ~3.3s with `llm_ttft` about 65% of it; inserting a
synchronous summarization call into the turn path would add seconds of
dead air to a live voice conversation. Compaction is therefore performed
by a background worker triggered while the user is speaking or during
idle, never between "user stopped talking" and "bot starts talking".
This is the same reasoning behind OpenAI's "compact at boundaries, not
every turn" and their guidance to call `run_compaction()` manually for
fast turn-taking.

**Storage.** `turns` remains the full verbatim record — the UI keeps
showing every message, which is what "maintain all messages" requires.
A summary is a *derived, rebuildable* artifact in `session_summaries`,
append-only, each row recording `covers_through_turn_id`. Building the
LLM message list is then: latest summary, plus every turn after
`covers_through_turn_id`.

**Token accounting.** `turns.token_count` is populated per turn so the
context size is a cheap `SUM`, not a re-tokenization of the whole
history every turn. The summarizer's own input/output tokens are
recorded as a `usage_events` row of kind `model_call` — Anthropic
separates this in `usage.iterations` precisely because it is easy to
under-report, and the same trap applies to our own summarizer calls.

---

## Phase 5: Guardrails (PII + per-user call limits)

LangChain ships exactly the middleware wanted here — PII detection, tool
call limit, model call limit — and **all of them require
`create_agent()`**. Same conclusion as compaction: port the design and
the defaults, implement at the points this codebase already has.

### 5a. PII detection

LangChain's PII middleware config and defaults:

| Option | Default | Values |
|---|---|---|
| `strategy` | `redact` | `redact` (`[REDACTED_{TYPE}]`), `mask` (`****-****-****-1234`), `hash` (deterministic), `block` (raise) |
| `apply_to_input` | `true` | check user messages |
| `apply_to_output` | `false` | check model responses |
| `apply_to_tool_results` | `false` | check tool output |

Built-in detectors cover **email, credit card, IP, MAC address, URL**
only. Their own documentation shows SSN as a *custom* detector example.

**Two findings that matter for this project specifically:**

1. **The built-in detector set is not sufficient here.** This is a
   healthcare context (the repo's own MCP configuration includes ICD-10
   and NPI registry servers). SSN, MRN, DOB, and patient names are the
   PII/PHI that matters and none are built in — custom detectors are
   required regardless of which framework is used, which removes most of
   the "just use the middleware" advantage.
2. **Voice input defeats regex detectors.** PII arrives here as an STT
   transcript. A spoken social security number transcribes as words
   ("four five six...") or with spaces, not as `\d{3}-\d{2}-\d{4}`.
   Detectors written for typed input will silently miss it. Any detector
   added here must handle spelled-out digits, or the feature gives false
   assurance. This is the single biggest risk in this phase and should
   be validated against real STT output before the feature is claimed to
   work.

**Design:**
- Detection runs at two points already present in the pipeline: on the
  STT transcript before it is sent to the LLM (`apply_to_input`), and
  optionally on the reply before TTS (`apply_to_output`).
- `pii_strategy` per user, default `redact`, with `off` as an explicit
  additional value (the local-only single-user case).
- **Do not redact what is stored.** `turns.text` keeps the user's own
  words — redacting a user's own history from themselves is the wrong
  tradeoff, and the verbatim record is what the UI and compaction both
  need. What is redacted is what leaves the process toward a third-party
  LLM provider. `turns.pii_flags` records *which types* were detected
  for audit, without duplicating the content.

### 5b. Per-user tool and model call limits

LangChain's middleware config and defaults:

| Middleware | Options | Default `exit_behavior` |
|---|---|---|
| `ToolCallLimitMiddleware` | `tool_name` (optional, else global), `thread_limit`, `run_limit` (at least one required — no numeric default) | `continue` — blocks the call with an error message, agent keeps going |
| `ModelCallLimitMiddleware` | `thread_limit`, `run_limit` | `end` — graceful termination |

**What already exists here:** `LangChainLlm`'s hand-rolled tool loop
already enforces `max_tool_rounds=3`, which is exactly LangChain's
`run_limit` for tools. What is missing is the **persisted, per-user**
limit (their `thread_limit`, which requires a checkpointer in LangGraph
— for us it is a database query).

**Design:**
- Enforcement point: the existing tool-execution loop in
  `langchain_llm.py` (tools) and the LLM call site (model calls).
- Limits are per-user, per rolling hour, read from `user_settings` with
  a global default in `config.py`. LangChain provides no numeric
  defaults, so these are ours to choose; proposed starting values are
  `tool_calls_per_hour=100`, `model_calls_per_hour=500`,
  `max_cost_usd_per_day=5.00`, all tunable per user and all nullable
  meaning "no limit".
- Behavior on limit: follow LangChain's `continue` default for tools —
  block the call, return a tool-result error telling the model the tool
  is rate-limited, let it answer without it. For a voice agent this is
  clearly right: the bot explains instead of going silent. Model-call
  and cost limits end the turn with a spoken explanation.
- Counting source: `usage_events`, one row per model call and per tool
  call, indexed on `(user_id, kind, created_at DESC)`. The limit check
  is an indexed range count over the last hour.

### 5c. User-facing configuration

All Phase 4 and 5 settings are exposed in the existing Settings panel
under a new section, reading and writing `user_settings` through
`GET/PUT /settings`. Every field shows the inherited default when unset.
This is the "user can change it, defaults follow LangChain" requirement:
defaults live in `config.py`, overrides live per user in the database,
and the resolution order is always *user setting → global default*.

---

## Database schema

`NN` = NOT NULL. Every `user_id` foreign key is `ON DELETE CASCADE`, so
deleting a user removes their sessions, turns, credentials, tokens, and
usage rows. `turns.session_id` and `session_summaries.session_id` are
also `ON DELETE CASCADE`, which lets `delete_session` drop its manual
two-statement delete.

```
                        ┌──────────────────────────────┐
                        │ users                        │
                        ├──────────────────────────────┤
                        │ id            UUID PK        │
                        │ email         TEXT UNIQUE NN │
                        │ password_hash TEXT NN        │
                        │ created_at    TIMESTAMPTZ NN │
                        │ updated_at    TIMESTAMPTZ NN │
                        └──────────────────────────────┘
                          │ 1      │ 1       │ 1      │ 1
        ┌─────────────────┘        │         │        └──────────────┐
        │ 1                        │ N       │ N                     │ N
┌──────────────────────────┐  ┌──────────────────────┐  ┌───────────────────────────┐
│ user_settings            │  │ refresh_tokens       │  │ api_credentials           │
├──────────────────────────┤  ├──────────────────────┤  ├───────────────────────────┤
│ user_id   UUID PK/FK     │  │ id        UUID PK    │  │ id        UUID PK         │
│ compact_trigger_fraction │  │ user_id   FK NN      │  │ user_id   FK NN           │
│ compact_keep_messages    │  │ token_hash TEXT NN   │  │ provider  TEXT NN         │
│ pii_strategy             │  │ expires_at TSTZ NN   │  │ encrypted_payload BYTEA NN│
│ pii_apply_to_output      │  │ revoked_at TSTZ      │  │ nonce     BYTEA NN        │
│ tool_calls_per_hour      │  │ created_at TSTZ NN   │  │ created_at TSTZ NN        │
│ model_calls_per_hour     │  └──────────────────────┘  │ updated_at TSTZ NN        │
│ max_cost_usd_per_day     │   IX (expires_at)          │ UNIQUE (user_id, provider)│
│ updated_at TIMESTAMPTZ NN│   IX (user_id)             └───────────────────────────┘
└──────────────────────────┘
 (all setting columns NULL
  = inherit config.py default)

┌───────────────────────────────┐        ┌──────────────────────────────────┐
│ sessions                      │        │ usage_events         (Phase 5)   │
├───────────────────────────────┤        ├──────────────────────────────────┤
│ id            TEXT PK         │        │ id           BIGSERIAL PK        │
│ user_id       FK -> users NN  │ <-NEW  │ user_id      FK -> users NN      │
│ created_at    TIMESTAMPTZ NN  │        │ session_id   FK -> sessions      │
│ mode          TEXT NN         │        │ kind         TEXT NN             │
│ tts_engine    TEXT            │        │   'model_call' | 'tool_call'     │
│ llm_model     TEXT NN         │        │ name         TEXT                │
│ title         TEXT            │        │ status       TEXT NN             │
└───────────────────────────────┘        │   'ok' | 'error' | 'blocked'     │
 IX (user_id, created_at DESC)  <-NEW    │ input_tokens  INT                │
        │ 1                 │ 1          │ output_tokens INT                │
        │                   │            │ cost_usd     NUMERIC(12,6)       │
        │ N                 │ N          │ created_at   TIMESTAMPTZ NN      │
┌───────────────────────────┐ │          └──────────────────────────────────┘
│ turns                     │ │           IX (user_id, kind, created_at DESC)
├───────────────────────────┤ │
│ id          BIGSERIAL PK  │ │          ┌──────────────────────────────────┐
│ session_id  FK NN         │ └───────>  │ session_summaries    (Phase 4)   │
│ role        TEXT NN       │            ├──────────────────────────────────┤
│ text        TEXT NN       │            │ id          BIGSERIAL PK         │
│ usage_json  JSONB         │            │ session_id  FK -> sessions NN    │
│ created_at  TIMESTAMPTZ NN│            │ summary_text TEXT NN             │
│ token_count INT   (Ph. 4) │ <-NEW      │ covers_through_turn_id BIGINT NN │
│ pii_flags   JSONB (Ph. 5) │ <-NEW      │ token_count INT NN               │
└───────────────────────────┘            │ model       TEXT NN              │
 IX (session_id)                         │ created_at  TIMESTAMPTZ NN       │
                                          └──────────────────────────────────┘
                                           IX (session_id, created_at DESC)
```

### Normalization notes and deliberate tradeoffs

- **`users` / `user_settings` split (1:1).** Identity is read on every
  authenticated request; settings are read once per session start and
  will keep growing a column per new knob. Keeping them apart avoids
  widening the hot table and separates two different change rates. A
  typed 1:1 table is chosen over a generic key/value settings table so
  defaults and types stay explicit.
- **`api_credentials` is one row per `(user_id, provider)`, not one per
  field.** Azure's key+endpoint+deployment are never read or written
  independently — they are one atomic credential set, decrypted together
  into one env dict, exactly as `server.py`'s existing `session_keys`
  handler already groups them. One AES-GCM ciphertext per provider.
- **`turns` has no `user_id`.** Ownership is transitive via
  `session_id → sessions.user_id`, and no query asks for a user's turns
  across sessions. A duplicate column would be a denormalized cache
  nothing reads.
- **`usage_events` *does* carry `user_id`.** Unlike `turns`, its hot
  query *is* cross-session: "how many tool calls has this user made in
  the last hour", evaluated on the tool-execution path. Joining through
  `sessions` for every tool call is exactly what this denormalization
  exists to avoid. The asymmetry with `turns` is intentional, and the
  rule behind it is: denormalize where a hot cross-session query exists,
  not otherwise.
- **`usage_events` vs `turns.usage_json` overlap.** These are different
  granularities: `usage_json` is a per-assistant-turn snapshot the UI
  renders; `usage_events` is a per-call ledger used for enforcement and
  audit (one turn with three tool rounds produces one `turns` row and
  several `usage_events` rows). This is a read-model/ledger split, not
  accidental duplication — but it *is* partial duplication and is worth
  revisiting if the UI ever needs to render from the ledger directly.
- **`session_summaries` is append-only** rather than one mutable summary
  per session, so each compaction is auditable and the message list can
  be rebuilt at any point in the session's history.
- **Counting from `usage_events` vs a rollup counter table.** A rolling
  hour count over an indexed `(user_id, kind, created_at DESC)` range is
  cheap at this application's scale, and the same rows double as the
  audit trail. If volume ever makes the count expensive, add a
  fixed-window rollup table then — don't build it preemptively.
- **`refresh_tokens.token_hash` stores `sha256(raw_token)`.** Same
  reasoning as `password_hash`: a database leak must not yield usable
  credentials.

---

## Why a shared schema, not multi-tenancy

Multi-tenant patterns (schema-per-tenant, database-per-tenant, or
row-level `tenant_id` isolation) solve a different problem: isolating
*organizations*, compliance/data-residency boundaries, or moving large
tenants to dedicated infrastructure. This app has one flat pool of
individual users who each own their data directly. For that, current
Postgres guidance is a **shared schema with a `user_id` foreign key on
every owned table** — which is what the schema above is. If org/team
support is ever needed, `user_id` promotes to an `owner_id` on a
membership join table; nothing here forecloses it.

---

## Why we keep the hand-rolled agent loop

This question recurs in phases 4 and 5 because the features wanted —
summarization, PII detection, tool call limits, model call limits — all
exist as LangChain middleware, and **every LangChain middleware requires
`create_agent()`**, which compiles to a LangGraph `StateGraph`. The
accumulation is a fair argument for reconsidering, so the reasoning is
recorded here rather than re-litigated per phase.

The codebase already made this call: `LangChainLlm` hand-rolls the
tool-execution loop specifically to avoid `create_agent`'s `AgentState`
plus checkpointer, "which would duplicate the history `Agent` already
tracks in `self.conversation`". Beyond that, this is not a
request/response agent:

- **Barge-in requires mid-stream cancellation.** `LlmBase.stream(messages,
  cancel)` must abandon an in-flight generation the moment VAD detects
  the user talking over the reply.
- **Tokens stream into a TTS chunker**, grouped at sentence boundaries
  (`FIRST_CHUNK_CHARS` / `MAX_CHUNK_CHARS`) so synthesis of one sentence
  overlaps generation of the next.
- **One stateless `LlmBase` is shared across all concurrent sessions**,
  with per-connection history owned by `Agent` — this is what keeps
  memory flat as session count grows.
- Voice/text mode, wake-word gating, and push-to-talk all branch outside
  a standard tool loop.

LangChain's own guidance is to use `create_agent` when the workflow fits
a standard model-and-tools loop, and to hand-roll or drop to direct
LangGraph when the loop's shape itself must differ — which is this case.

**The condition that would change this answer** is narrow and testable:
if `create_agent` can be shown to support token-level streaming into an
external consumer *with* mid-generation cancellation, the middleware
ecosystem becomes worth the migration. That is a spike, not an
assumption, and it is out of scope here. Until it is run, phases 4 and 5
port the middleware *designs and defaults* rather than adopting the
runtime.

Likewise, LangGraph's `PostgresSaver` checkpointer is not adopted: it
persists serialized graph execution state per `thread_id`/`checkpoint_id`
as opaque `BYTEA` blobs, for resuming a graph — not human-readable rows
a session-history UI can list. Even after a hypothetical migration,
`sessions`/`turns` would still be needed for the UI, so the checkpointer
would add a second persistence mechanism rather than replace one.

---

## API changes

**New:**

| Endpoint | Purpose |
|---|---|
| `POST /auth/signup` | email + password; sets the same cookies as login |
| `POST /auth/login` | sets access + refresh cookies |
| `POST /auth/logout` | revokes the refresh token, clears cookies |
| `POST /auth/refresh` | rotates the refresh token, issues a new access token |
| `GET /auth/me` | the current user (email, id) for the account menu |
| `GET /credentials` | which providers are configured — never the values |
| `PUT /credentials/{provider}` | save/update that provider's encrypted fields |
| `DELETE /credentials/{provider}` | remove a stored credential |
| `GET /settings` | **Phase 4/5 only.** Resolved settings (user override or inherited default) |
| `PUT /settings` | **Phase 4/5 only.** Update this user's overrides; `null` clears one back to the default |

The two `/settings` endpoints ship with the phase whose settings they expose —
`user_settings` is created in Phase 1 so the schema is built once, but an API
that configures nothing would be dead code until then.

**Changed:**

- `GET /sessions`, `GET`/`DELETE /sessions/{id}` — scoped to the
  authenticated user; 404 for another user's id.
- `WS /ws` — requires a valid access token (the browser sends the cookie
  automatically on the same origin); new sessions are stamped with the
  connecting user's id. The `key_token` query parameter and
  `POST /session-keys` are **removed**, replaced by reading the user's
  stored `api_credentials` server-side. Leaving a provider unconfigured
  still falls back to the server's own environment variables, matching
  today's "leave blank" behavior.
- `GET /options` — unchanged; engine/voice/model lists are global, not
  per user.

---

## Frontend changes

- `/login` and `/signup` pages.
- A root auth guard: unauthenticated users are redirected to `/login`;
  the chat UI mounts only once authenticated.
- Settings panel: provider fields gain real **Save** and **Remove**
  buttons (`PUT`/`DELETE /credentials/{provider}`), replacing the
  current "retype every connect, never persisted" copy and behavior. A
  configured provider shows as configured without the value ever being
  sent back to the browser.
- A new Settings section for Phase 4/5 knobs (compaction thresholds, PII
  strategy, call limits), each showing its inherited default when unset.
- An account menu (email + Logout) in the sidebar footer.

---

## Testing

- `backend/tests/test_auth.py` — signup, login, refresh, logout, wrong
  password, duplicate email, expired/invalid token, refresh rotation,
  reuse of a revoked token.
- `backend/tests/test_credentials.py` — save/list/delete, encryption
  round-trip, and that one user can never read another's credentials.
- `backend/tests/test_server.py` — extend existing session tests to
  assert cross-user isolation (user A never sees or deletes user B's
  sessions) and that unauthenticated requests are rejected.
- `backend/tests/test_storage.py` — extend for the pool; assert
  concurrent writes from multiple threads all land.
- Phase 4/5 tests are defined with their own plans; the PII detector
  tests must include **STT-shaped input** (spelled-out digits, no
  punctuation), not just typed formats.
- Frontend: manual verification, consistent with this project's existing
  stance (no frontend test framework).

---

## Migration path

There is no data to preserve, so schema changes land as
`CREATE TABLE`/`ALTER TABLE ... ADD COLUMN` statements in `storage.py`'s
existing startup block — the same pattern already used for the `title`
column — rather than introducing a migration tool. `sessions.user_id` is
added `NOT NULL` with no default, which is safe only because the table
is empty in every environment this runs in today; that assumption should
be re-checked before running against any database that has real rows.

A migration tool (Alembic) becomes worth adding the first time a schema
change has to preserve existing production rows. That is explicitly not
now, and the startup-DDL pattern should not outlive the first real
deployment.
