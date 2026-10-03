"""The file panel's endpoints: what they show, and what they refuse."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pos import files
from pos.app import app


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A folder to open, with a secret outside it and a few traps inside."""
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("print('hi')\n")
    (root / "README.md").write_text("# hi\n")
    (root / "index.html").write_text("<h1>hi</h1>")
    (root / ".env").write_text("SECRET=1\n")
    (root / "src" / ".env.local").write_text("SECRET=2\n")
    (root / "node_modules").mkdir()
    (root / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")

    outside = tmp_path / "outside.txt"
    outside.write_text("not yours\n")
    os.symlink(outside, root / "escape.txt")
    os.symlink(root / ".env", root / "env-link")
    return root


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.mark.parametrize("path", ["../outside.txt", "src/../../outside.txt", "/../outside.txt"])
def test_refuses_paths_that_climb_out(project: Path, path: str):
    with pytest.raises(files.FileAccessError) as e:
        files.read_text(str(project), path)
    assert e.value.status == 403


def test_refuses_a_symlink_that_points_out(project: Path):
    with pytest.raises(files.FileAccessError) as e:
        files.read_text(str(project), "escape.txt")
    assert e.value.status == 403


@pytest.mark.parametrize("path", [".env", "src/.env.local", "env-link"])
def test_refuses_env_files_however_named(project: Path, path: str):
    with pytest.raises(files.FileAccessError) as e:
        files.read_text(str(project), path)
    assert e.value.status == 403


def test_reads_a_file_with_or_without_a_leading_slash(project: Path):
    assert files.read_text(str(project), "src/app.py") == "print('hi')\n"
    assert files.read_text(str(project), "/src/app.py") == "print('hi')\n"


def test_refuses_binary_and_oversized_files(project: Path):
    with pytest.raises(files.FileAccessError) as e:
        files.read_text(str(project), "logo.png")
    assert e.value.status == 415

    (project / "big.log").write_bytes(b"x" * (files.MAX_VIEW_BYTES + 1))
    with pytest.raises(files.FileAccessError) as e:
        files.read_text(str(project), "big.log")
    assert e.value.status == 413


def test_lists_folders_first_and_marks_what_cannot_be_opened(project: Path):
    entries, truncated = files.list_dir(str(project))
    by_name = {e.name: e for e in entries}

    assert not truncated
    assert [e.type for e in entries[:3]] == ["dir", "dir", "file"]  # node_modules, src, then files
    assert by_name["node_modules"].heavy
    assert by_name[".env"].locked
    assert by_name["escape.txt"].locked
    assert not by_name["README.md"].locked


def test_folder_key_round_trips(project: Path):
    assert files.folder_from_key(files.folder_key(str(project))) == str(project)


def test_tree_and_file_endpoints(project: Path, client: TestClient):
    tree = client.get("/fs/tree", params={"folder": str(project)}).json()
    assert "README.md" in [e["name"] for e in tree["entries"]]

    sub = client.get("/fs/tree", params={"folder": str(project), "path": "src"}).json()
    assert [e["path"] for e in sub["entries"]] == ["src/.env.local", "src/app.py"]

    ok = client.get("/fs/file", params={"folder": str(project), "path": "README.md"})
    assert ok.json()["content"] == "# hi\n"

    assert client.get("/fs/file", params={"folder": str(project), "path": ".env"}).status_code == 403
    assert client.get("/fs/file", params={"folder": str(project), "path": "../outside.txt"}).status_code == 403
    assert client.get("/fs/file", params={"folder": "/no/such/folder", "path": "x"}).status_code == 400


def test_preview_is_sandboxed_and_the_app_is_not_loosened(project: Path, client: TestClient):
    key = files.folder_key(str(project))
    page = client.get(f"/fs/preview/{key}/index.html")

    assert page.status_code == 200
    assert page.text == "<h1>hi</h1>"
    assert page.headers["content-type"].startswith("text/html")
    csp = page.headers["content-security-policy"]
    assert csp.startswith("sandbox allow-scripts")
    assert "allow-same-origin" not in csp
    assert page.headers["x-frame-options"] == "SAMEORIGIN"

    assert client.get(f"/fs/preview/{key}/.env").status_code == 403
    assert client.get(f"/fs/preview/{key}/escape.txt").status_code == 403
    assert client.get("/fs/preview/not-a-key/index.html").status_code == 400

    # Everything else keeps the app's own strict policy.
    api = client.get("/fs/tree", params={"folder": str(project)})
    assert api.headers["x-frame-options"] == "DENY"
    assert "sandbox" not in api.headers["content-security-policy"]
