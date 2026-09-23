"""Usage statistics, aggregated from the turns already stored per chat.

Nothing extra is recorded for this -- every number here comes from the
`turns` and `sessions` rows that `/chat/stream` already writes, plus the
`usage_ledger` rows left behind by chats that have since been deleted. The
provider billed for those too, so leaving them out would make this page
disagree with the bill by however much has been deleted.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from pos import db

# How far back each range goes, as a SQLite datetime modifier. None = everything.
RANGES: dict[str, str | None] = {"all": None, "30d": "-30 days", "7d": "-7 days"}


def _usage_rows(conn, since: str | None) -> list[dict]:
    """Assistant turns in range, with their usage JSON already parsed."""
    sql = """SELECT model, usage_json, created_at FROM turns
             WHERE role = 'assistant' AND usage_json IS NOT NULL"""
    params: tuple = ()
    if since:
        sql += " AND created_at >= datetime('now', ?)"
        params = (since,)

    rows = []
    for r in conn.execute(sql, params).fetchall():
        try:
            usage = json.loads(r["usage_json"])
        except (TypeError, ValueError):
            continue
        rows.append({"model": r["model"], "usage": usage, "created_at": r["created_at"]})
    return rows


def _ledger_rows(conn, since: str | None) -> list[dict]:
    """Deleted chats' usage in range, shaped like `_usage_rows` output.

    The ledger holds one row per day and model rather than per turn, so its
    `turns` count is carried through rather than being one per row.
    """
    sql = "SELECT day, model, turns, input_tokens, output_tokens, total_tokens, cache_read, cache_creation, cost FROM usage_ledger"
    params: tuple = ()
    if since:
        sql += " WHERE day >= date('now', ?)"
        params = (since,)
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def summary(range_key: str = "all") -> dict:
    """Aggregate usage for one time range.

    Args:
        range_key: One of `RANGES` ("all", "30d", "7d"). Unknown values
            fall back to "all".

    Returns:
        Totals, a per-day series (for the heatmap and bar chart), a
        per-model breakdown, and the peak hour of day.
    """
    since = RANGES.get(range_key, None)
    conn = db.connect()
    try:
        rows = _usage_rows(conn, since)
        ledger = _ledger_rows(conn, since)

        session_sql = "SELECT COUNT(*) FROM sessions"
        session_params: tuple = ()
        if since:
            session_sql += " WHERE created_at >= datetime('now', ?)"
            session_params = (since,)
        sessions = conn.execute(session_sql, session_params).fetchone()[0]
    finally:
        conn.close()

    by_day: dict[str, dict] = defaultdict(lambda: {"input": 0, "output": 0, "messages": 0, "cost": 0.0})
    by_model: dict[str, dict] = defaultdict(lambda: {"input": 0, "output": 0, "messages": 0, "cost": 0.0})
    hours: Counter = Counter()
    totals = {"input": 0, "output": 0, "total": 0, "cost": 0.0, "cache_read": 0, "cache_creation": 0}

    for row in rows:
        u = row["usage"]
        day = (row["created_at"] or "")[:10]
        hour = (row["created_at"] or "")[11:13]
        model = row["model"] or "unknown"

        inp = u.get("input_tokens") or 0
        out = u.get("output_tokens") or 0
        cost = u.get("cost_usd") or 0.0

        totals["input"] += inp
        totals["output"] += out
        totals["total"] += u.get("total_tokens") or 0
        totals["cost"] += cost
        totals["cache_read"] += u.get("cache_read") or 0
        totals["cache_creation"] += u.get("cache_creation") or 0

        for bucket in (by_day[day], by_model[model]):
            bucket["input"] += inp
            bucket["output"] += out
            bucket["messages"] += 1
            bucket["cost"] += cost

        if hour.isdigit():
            hours[int(hour)] += 1

    # Deleted chats. Counted in the totals, the day series and the per-model
    # breakdown -- the money was spent -- but they contribute no hour, since
    # the ledger keeps only the day.
    deleted = {"turns": 0, "input": 0, "output": 0, "cost": 0.0}
    for row in ledger:
        inp, out = row["input_tokens"], row["output_tokens"]
        model = row["model"] or "unknown"

        totals["input"] += inp
        totals["output"] += out
        totals["total"] += row["total_tokens"]
        totals["cost"] += row["cost"]
        totals["cache_read"] += row["cache_read"]
        totals["cache_creation"] += row["cache_creation"]

        for bucket in (by_day[row["day"]], by_model[model]):
            bucket["input"] += inp
            bucket["output"] += out
            bucket["messages"] += row["turns"]
            bucket["cost"] += row["cost"]

        deleted["turns"] += row["turns"]
        deleted["input"] += inp
        deleted["output"] += out
        deleted["cost"] += row["cost"]

    model_total = sum(m["input"] + m["output"] for m in by_model.values()) or 1
    models = sorted(
        (
            {
                "model": name,
                **vals,
                "share": (vals["input"] + vals["output"]) / model_total,
            }
            for name, vals in by_model.items()
        ),
        key=lambda m: m["share"],
        reverse=True,
    )

    return {
        "range": range_key if range_key in RANGES else "all",
        "sessions": sessions,
        "messages": len(rows) + deleted["turns"],
        # Shown as its own line, so a total that includes chats you can no
        # longer open is explained rather than mysterious.
        "deleted": deleted,
        "active_days": len(by_day),
        "peak_hour": hours.most_common(1)[0][0] if hours else None,
        "favorite_model": models[0]["model"] if models else None,
        "totals": totals,
        "by_day": [{"day": d, **v} for d, v in sorted(by_day.items())],
        "by_model": models,
    }
