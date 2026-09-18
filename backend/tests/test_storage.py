import os

from pos.db import create_pool, init_schema
from pos.storage import SessionStore

_DSN = os.environ.get("TEST_DATABASE_URL", "postgresql://pos:pos@localhost:5432/pos")


def _store():
    """A SessionStore on a clean database, plus one user that owns
    everything the test creates."""
    pool = create_pool(_DSN, max_size=5)
    init_schema(pool)
    with pool.connection() as conn:
        conn.execute("TRUNCATE users, sessions, turns RESTART IDENTITY CASCADE")
        user = conn.execute(
            "INSERT INTO users (email, password_hash) VALUES ('owner@test', 'x') RETURNING id"
        ).fetchone()
    store = SessionStore(pool)
    store.test_user_id = str(user["id"])
    return store


def test_create_and_list_sessions():
    store = _store()
    store.create_session("s1", store.test_user_id, mode="voice", tts_engine="kokoro", llm_model="lfm2.5-230m")
    store.create_session("s2", store.test_user_id, mode="text", tts_engine=None, llm_model="gpt-4o-mini")

    sessions = store.list_sessions(store.test_user_id)

    assert [s["id"] for s in sessions] == ["s2", "s1"]
    assert sessions[0]["mode"] == "text"
    assert sessions[0]["tts_engine"] is None
    assert sessions[1]["tts_engine"] == "kokoro"
    assert "created_at" in sessions[0]


def test_list_sessions_respects_limit():
    store = _store()
    for i in range(5):
        store.create_session(f"s{i}", store.test_user_id, mode="voice", tts_engine="kokoro", llm_model="m")

    assert len(store.list_sessions(store.test_user_id, limit=2)) == 2


def test_add_turn_and_get_session():
    store = _store()
    store.create_session("s1", store.test_user_id, mode="voice", tts_engine="kokoro", llm_model="m")
    store.add_turn("s1", "user", "hello")
    store.add_turn("s1", "assistant", "hi there", usage={"input_tokens": 5, "output_tokens": 3})

    result = store.get_session("s1", store.test_user_id)

    assert result["session"]["id"] == "s1"
    assert len(result["turns"]) == 2
    assert result["turns"][0] == {"role": "user", "text": "hello", "usage": None}
    assert result["turns"][1] == {
        "role": "assistant", "text": "hi there",
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }


def test_get_session_returns_turn_count_in_list_sessions():
    store = _store()
    store.create_session("s1", store.test_user_id, mode="voice", tts_engine="kokoro", llm_model="m")
    store.add_turn("s1", "user", "hello")
    store.add_turn("s1", "assistant", "hi")

    sessions = store.list_sessions(store.test_user_id)

    assert sessions[0]["turn_count"] == 2


def test_get_session_unknown_id_returns_none():
    store = _store()

    assert store.get_session("does-not-exist", store.test_user_id) is None


def test_add_turn_without_usage_stores_none():
    store = _store()
    store.create_session("s1", store.test_user_id, mode="voice", tts_engine="kokoro", llm_model="m")
    store.add_turn("s1", "user", "hello")

    result = store.get_session("s1", store.test_user_id)

    assert result["turns"][0]["usage"] is None


def test_new_session_has_no_title_until_set():
    store = _store()
    store.create_session("s1", store.test_user_id, mode="voice", tts_engine="kokoro", llm_model="m")

    assert store.get_session("s1", store.test_user_id)["session"]["title"] is None
    assert store.list_sessions(store.test_user_id)[0]["title"] is None


def test_set_title_updates_session_and_list():
    store = _store()
    store.create_session("s1", store.test_user_id, mode="voice", tts_engine="kokoro", llm_model="m")

    store.set_title("s1", "Weekend trip planning")

    assert store.get_session("s1", store.test_user_id)["session"]["title"] == "Weekend trip planning"
    assert store.list_sessions(store.test_user_id)[0]["title"] == "Weekend trip planning"


def test_set_mode_updates_session_and_list():
    store = _store()
    store.create_session("s1", store.test_user_id, mode="voice", tts_engine="kokoro", llm_model="m")

    store.set_mode("s1", "text")

    assert store.get_session("s1", store.test_user_id)["session"]["mode"] == "text"
    assert store.list_sessions(store.test_user_id)[0]["mode"] == "text"


def test_delete_session_removes_it_and_its_turns():
    store = _store()
    store.create_session("s1", store.test_user_id, mode="voice", tts_engine="kokoro", llm_model="m")
    store.add_turn("s1", "user", "hello")

    deleted = store.delete_session("s1", store.test_user_id)

    assert deleted is True
    assert store.get_session("s1", store.test_user_id) is None
    assert store.list_sessions(store.test_user_id) == []


def test_delete_session_unknown_id_returns_false():
    store = _store()

    assert store.delete_session("does-not-exist", store.test_user_id) is False


def test_one_user_cannot_see_or_delete_another_users_session():
    store = _store()
    with store._pool.connection() as conn:
        other = conn.execute(
            "INSERT INTO users (email, password_hash) VALUES ('other@test', 'x') RETURNING id"
        ).fetchone()
    other_id = str(other["id"])
    store.create_session("mine", store.test_user_id, mode="text", tts_engine=None, llm_model="m")

    assert store.list_sessions(other_id) == []
    assert store.get_session("mine", other_id) is None
    assert store.delete_session("mine", other_id) is False
    assert store.get_session("mine", store.test_user_id) is not None
