"""RealPathMiddleware: real paths inside the open folder become the file tools' own."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pos.agent import RealPathMiddleware


class _Request(SimpleNamespace):
    def override(self, **changes):
        return _Request(**{**self.__dict__, **changes})


@pytest.fixture
def root(tmp_path: Path) -> Path:
    folder = tmp_path / "test"
    folder.mkdir()
    return folder


def run(middleware: RealPathMiddleware, args: dict) -> dict:
    seen = {}
    middleware.wrap_tool_call(
        _Request(tool_call={"name": "read_file", "args": args, "id": "1"}),
        lambda request: seen.update(request.tool_call["args"]),
    )
    return seen


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("{root}/cloud/solar-system.html", "/cloud/solar-system.html"),
        ("{root}", "/"),
        ("{root}/", "/"),
        ("/cloud/solar-system.html", "/cloud/solar-system.html"),  # already the tools' own
        ("{root}-other/x.html", "{root}-other/x.html"),  # a sibling that merely starts the same
        ("/etc/passwd", "/etc/passwd"),  # outside: left for the tool to refuse
    ],
)
def test_rewrites_only_paths_inside_the_folder(root: Path, given: str, expected: str):
    middleware = RealPathMiddleware(str(root))
    out = run(middleware, {"file_path": given.format(root=root)})
    assert out["file_path"] == expected.format(root=root)


def test_leaves_other_arguments_alone(root: Path):
    middleware = RealPathMiddleware(str(root))
    out = run(middleware, {"path": f"{root}/src", "pattern": f"{root}/*.py", "content": str(root)})
    assert out == {"path": "/src", "pattern": f"{root}/*.py", "content": str(root)}
