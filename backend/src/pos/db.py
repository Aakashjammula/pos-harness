"""Postgres connection pool shared by every store in the process.

Replaces the single connection + global lock the SQLite-era SessionStore
used: that serialized every query process-wide, which Postgres has no
reason to do."""

from __future__ import annotations

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from pos import config


def create_pool(dsn: str | None = None, max_size: int | None = None) -> ConnectionPool:
    return ConnectionPool(
        conninfo=dsn or config.DATABASE_URL,
        min_size=2,
        max_size=max_size or config.DB_POOL_MAX_SIZE,
        kwargs={"row_factory": dict_row, "autocommit": True},
        open=True,
    )
