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

**Text mode is the primary mode; voice is secondary.** Compaction is
therefore designed the ordinary way — synchronously, at a turn boundary,
as all three reference implementations do. A multi-second pause before a
reply is normal in text chat and is what every chat product already does
behind a typing indicator.

**Voice mode gets one extra accommodation.** TTFA there is currently
~3.3s with `llm_ttft` about 65% of it, so a synchronous summarization
call inserted between "user stopped talking" and "bot starts talking"
would add seconds of dead air. In voice mode only, compaction is deferred
to a background worker triggered while the user is speaking or during
idle. This is a mode-specific scheduling detail, not a different
mechanism — the trigger, the prompt, the `keep` window, and the stored
result are identical. It is the same reasoning behind OpenAI's "compact
at boundaries, not every turn" and their guidance to call
`run_compaction()` manually when turn-taking is fast.

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
  user's message before it is sent to the LLM (`apply_to_input`), and
  optionally on the reply before it is returned or spoken
  (`apply_to_output`).
- **Off by default is not the design — off by *choice* is.** The whole
  feature is gated by `pii_enabled` per user (see "Configurability"
  below), and each individual detector can be enabled or disabled on its
  own.
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

### 5c. Configurability (every guardrail is opt-out and tunable)

Nothing in phases 4 and 5 is mandatory or fixed. Each feature has an
explicit on/off switch and tunable parameters, all per user, all editable
from the Settings UI. The resolution order is always **user setting →
global default from `config.py`**, and `NULL` in the database means
"inherit", never "disabled" — so raising a default later reaches every
user who has not deliberately overridden it.

**Master switches** (`user_settings`, one boolean each):

| Switch | Default | Effect when off |
|---|---|---|
| `compaction_enabled` | `true` | Fall back to the current fixed-window truncation |
| `pii_enabled` | `false` | No detection, no redaction, no `pii_flags` written |
| `rate_limits_enabled` | `false` | No counting, no blocking |

PII and rate limiting default to **off** because both change what the
model receives or whether a request runs at all, and a self-hosted
single-user install should not silently acquire either. Compaction
defaults to **on** because it strictly improves on the truncation it
replaces.

**Tunable parameters** (`user_settings`, `NULL` = inherit):

| Setting | Default | Source of the default |
|---|---|---|
| `compact_trigger_fraction` | `0.8` | LangChain `("fraction", 0.8)` |
| `compact_keep_messages` | `20` | LangChain `keep=("messages", 20)` |
| `pii_strategy` | `redact` | LangChain PII middleware default |
| `pii_apply_to_output` | `false` | LangChain `apply_to_output=False` |
| `tool_calls_per_hour` | `100` | ours — LangChain ships no numeric default |
| `model_calls_per_hour` | `500` | ours |
| `max_cost_usd_per_day` | `5.00` | ours |

**Per-detector customization** (`user_pii_rules`, one row per detector a
user has customized): each row carries `pii_type`, `enabled`, an optional
`strategy` overriding the user's global `pii_strategy` for that type
alone, and an optional `pattern` holding a custom regex. A user with no
rows gets the built-in detector set at their global strategy. This is a
table rather than more columns because "a set of detectors, each with its
own settings" is a repeating group — exactly what a child table is for —
and because custom patterns are open-ended.

The built-in detector set is seeded as rows on first use rather than
hardcoded, so enabling, disabling, restyling, or adding a detector are
all the same operation against the same table.

**Deliberately not built:** per-tool call limits (LangChain's
`tool_name` parameter). A global tool-call ceiling covers the stated
need; per-tool ceilings are speculative until someone asks for one. The
`usage_events.name` column already records which tool was called, so
adding them later is a query change, not a migration.

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
│ compaction_enabled  BOOL │  │ user_id   FK NN      │  │ user_id   FK NN           │
│ pii_enabled         BOOL │  │ token_hash TEXT NN   │  │ provider  TEXT NN         │
│ rate_limits_enabled BOOL │  │ expires_at TSTZ NN   │  │ encrypted_payload BYTEA NN│
│ compact_trigger_fraction │  │ revoked_at TSTZ      │  │ nonce     BYTEA NN        │
│ compact_keep_messages    │  │ created_at TSTZ NN   │  │ created_at TSTZ NN        │
│ pii_strategy             │  └──────────────────────┘  │ updated_at TSTZ NN        │
│ pii_apply_to_output      │   IX (expires_at)          │ UNIQUE (user_id, provider)│
│ tool_calls_per_hour      │   IX (user_id)             └───────────────────────────┘
│ model_calls_per_hour     │
│ max_cost_usd_per_day     │  ┌──────────────────────────────┐
│ updated_at TIMESTAMPTZ NN│  │ user_pii_rules     (Phase 5) │
└──────────────────────────┘  ├──────────────────────────────┤
 (every column NULL           │ id        BIGSERIAL PK       │
  = inherit config.py         │ user_id   FK -> users NN     │
  default; NULL never         │ pii_type  TEXT NN            │
  means "disabled")           │ enabled   BOOLEAN NN         │
        │ 1                   │ strategy  TEXT               │
        └──────────────────>  │ pattern   TEXT               │
                          N   │ updated_at TIMESTAMPTZ NN    │
                              │ UNIQUE (user_id, pii_type)   │
                              └──────────────────────────────┘

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
- **`user_pii_rules` is a child table, not more `user_settings`
  columns.** Per-detector settings are a repeating group — the same three
  attributes (enabled, strategy, pattern) for an open-ended set of
  detector types. Flattening that into columns would mean a schema change
  every time a detector is added, and custom patterns make the set
  genuinely unbounded.
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

**This argument is weaker than it first looks, and the weakening should
be recorded.** The most compelling reasons to stay hand-rolled were
voice-specific — barge-in needing mid-generation cancellation, and tokens
streaming into a sentence-boundary TTS chunker (`FIRST_CHUNK_CHARS` /
`MAX_CHUNK_CHARS`) so synthesis of one sentence overlaps generation of the
next. With **text as the primary mode**, those become constraints on a
secondary path rather than on the product's main flow.

What still holds independently of mode:

- **One stateless `LlmBase` is shared across all concurrent sessions**,
  with per-connection history owned by `Agent`. This is what keeps memory
  flat as session count grows, and it is the opposite of `create_agent`'s
  model, where state lives in the graph.
- **The checkpointer would duplicate `turns`.** `LangChainLlm` hand-rolls
  its loop specifically to avoid `create_agent`'s `AgentState` plus
  checkpointer, "which would duplicate the history `Agent` already tracks
  in `self.conversation`". That remains true, and the UI still needs
  readable rows the checkpointer's `BYTEA` blobs cannot provide.
- **Migration cost against a working pipeline.** Rewriting the agent
  runtime is a large change with no user-visible benefit on its own; the
  benefit is only unlocked afterwards, via middleware.

So the recommendation stands for phases 4 and 5 — port the middleware
designs and defaults — but it now stands mostly on migration cost rather
than on technical impossibility, and it should be re-examined rather than
inherited.

**The spike that would settle it** is narrow and worth running before
phase 5, not after: can `create_agent` stream tokens to an external
consumer *with* mid-generation cancellation, and can its middleware stack
be driven per-user (per-user PII rules, per-user limits) rather than
configured once at graph construction? The second half matters as much as
the first: this design requires every guardrail to be user-configurable,
and middleware instantiated at `create_agent()` time is the wrong shape
for per-user settings unless it can read them from runtime context. If
both answers are yes, adopting the runtime likely beats porting four
middleware by hand.

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
