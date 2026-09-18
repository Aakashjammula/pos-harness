import os
import threading

from pos.db import create_pool

_DSN = os.environ.get("TEST_DATABASE_URL", "postgresql://pos:pos@localhost:5432/pos")


def test_pool_runs_concurrent_queries():
    pool = create_pool(_DSN, max_size=5)
    results = []

    def query():
        with pool.connection() as conn:
            row = conn.execute("SELECT 1 AS n").fetchone()
            results.append(row["n"])

    threads = [threading.Thread(target=query) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == [1] * 20
    pool.close()


def test_pool_uses_dict_rows():
    pool = create_pool(_DSN, max_size=2)
    with pool.connection() as conn:
        row = conn.execute("SELECT 42 AS answer").fetchone()
    assert row == {"answer": 42}
    pool.close()
