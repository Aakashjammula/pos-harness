import os

from pos.storage import SessionStore

# Requires a running Postgres reachable at TEST_DATABASE_URL (defaults to the
# docker-compose "postgres" service exposed on localhost) -- see
# docker-compose.yml. Each test truncates both tables first for isolation,
# since a real Postgres instance persists across test runs unlike the old
# sqlite3 ":memory:" store.
_DSN = os.environ.get("TEST_DATABASE_URL", "postgresql://pos:pos@localhost:5432/pos")


def _store():
    store = SessionStore(_DSN)
    with store._lock:
        store._conn.execute("TRUNCATE turns, sessions RESTART IDENTITY CASCADE")
        store._conn.commit()
    return store


def test_create_and_list_sessions():
    store = _store()
    store.create_session("s1", mode="voice", tts_engine="kokoro", llm_model="lfm2.5-230m")
    store.create_session("s2", mode="text", tts_engine=None, llm_model="gpt-4o-mini")

    sessions = store.list_sessions()

    assert [s["id"] for s in sessions] == ["s2", "s1"]  # newest first
    assert sessions[0]["mode"] == "text"
    assert sessions[0]["tts_engine"] is None
    assert sessions[1]["tts_engine"] == "kokoro"
    assert "created_at" in sessions[0]


def test_list_sessions_respects_limit():
    store = _store()
    for i in range(5):
        store.create_session(f"s{i}", mode="voice", tts_engine="kokoro", llm_model="m")

    assert len(store.list_sessions(limit=2)) == 2


def test_add_turn_and_get_session():
    store = _store()
    store.create_session("s1", mode="voice", tts_engine="kokoro", llm_model="m")
    store.add_turn("s1", "user", "hello")
    store.add_turn("s1", "assistant", "hi there", usage={"input_tokens": 5, "output_tokens": 3})

    result = store.get_session("s1")

    assert result["session"]["id"] == "s1"
    assert len(result["turns"]) == 2
    assert result["turns"][0] == {"role": "user", "text": "hello", "usage": None}
    assert result["turns"][1] == {
        "role": "assistant", "text": "hi there",
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }


def test_get_session_returns_turn_count_in_list_sessions():
    store = _store()
    store.create_session("s1", mode="voice", tts_engine="kokoro", llm_model="m")
    store.add_turn("s1", "user", "hello")
    store.add_turn("s1", "assistant", "hi")

    sessions = store.list_sessions()

    assert sessions[0]["turn_count"] == 2


def test_get_session_unknown_id_returns_none():
    store = _store()

    assert store.get_session("does-not-exist") is None


def test_add_turn_without_usage_stores_none():
    store = _store()
    store.create_session("s1", mode="voice", tts_engine="kokoro", llm_model="m")
    store.add_turn("s1", "user", "hello")

    result = store.get_session("s1")

    assert result["turns"][0]["usage"] is None


def test_new_session_has_no_title_until_set():
    store = _store()
    store.create_session("s1", mode="voice", tts_engine="kokoro", llm_model="m")

    assert store.get_session("s1")["session"]["title"] is None
    assert store.list_sessions()[0]["title"] is None


def test_set_title_updates_session_and_list():
    store = _store()
    store.create_session("s1", mode="voice", tts_engine="kokoro", llm_model="m")

    store.set_title("s1", "Weekend trip planning")

    assert store.get_session("s1")["session"]["title"] == "Weekend trip planning"
    assert store.list_sessions()[0]["title"] == "Weekend trip planning"


def test_set_mode_updates_session_and_list():
    store = _store()
    store.create_session("s1", mode="voice", tts_engine="kokoro", llm_model="m")

    store.set_mode("s1", "text")

    assert store.get_session("s1")["session"]["mode"] == "text"
    assert store.list_sessions()[0]["mode"] == "text"


def test_delete_session_removes_it_and_its_turns():
    store = _store()
    store.create_session("s1", mode="voice", tts_engine="kokoro", llm_model="m")
    store.add_turn("s1", "user", "hello")

    deleted = store.delete_session("s1")

    assert deleted is True
    assert store.get_session("s1") is None
    assert store.list_sessions() == []


def test_delete_session_unknown_id_returns_false():
    store = _store()

    assert store.delete_session("does-not-exist") is False
