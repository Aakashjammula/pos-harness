"""Test database safety.

Several test modules run `TRUNCATE users, sessions, ...` against the database
in TEST_DATABASE_URL. Pointing that at the real development database wipes
every account and chat in it, so this module (imported before any test
module) does two things:

  * defaults TEST_DATABASE_URL to a dedicated `pos_test` database, and
  * aborts the run if the configured database name does not end in `_test`.

Create it once:  docker compose exec postgres psql -U pos -d postgres -c 'CREATE DATABASE pos_test'
(compose publishes Postgres on host port 5433 by default).
"""

import os
from urllib.parse import urlparse

import pytest

_DEFAULT = "postgresql://pos:pos@localhost:5433/pos_test"
os.environ.setdefault("TEST_DATABASE_URL", _DEFAULT)

_db_name = urlparse(os.environ["TEST_DATABASE_URL"]).path.lstrip("/")
if not _db_name.endswith("_test"):
    raise pytest.UsageError(
        f"refusing to run: TEST_DATABASE_URL points at database {_db_name!r}, and these tests "
        "TRUNCATE tables. Use a database whose name ends in '_test' (default: pos_test)."
    )


@pytest.fixture(autouse=True)
def _network_policy_defaults(monkeypatch):
    """Tests use made-up hostnames ("llm.test"); resolve them all to a public address and
    allow private literals. Tests of the policy itself pass their own resolver and flags."""
    from pos import config

    monkeypatch.setattr("pos.net_policy._resolve", lambda host, port: ["93.184.216.34"])
    monkeypatch.setattr(config, "ALLOW_PRIVATE_LLM_URLS", True)
