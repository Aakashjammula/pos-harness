"""Read-only access to the open folder, for the file panel.

The agent reaches the folder through deepagents' sandboxed backend; the UI
needs its own way in to show a tree and open files. Same rule as the agent:
nothing outside the folder, and never a `.env`.

Containment is checked on the *resolved* path. `Path.resolve()` follows
symlinks, so a link inside the folder that points out of it resolves to
its real target and fails the check -- comparing the joined path as a
string would let it through.
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import os
from pathlib import Path

# Big enough for any source file worth reading in a browser; small enough
# that opening a log or a data dump can't freeze the page.
MAX_VIEW_BYTES = 2 * 1024 * 1024

# A listing this long is not something anyone scrolls; say it was cut.
MAX_ENTRIES = 2000

# Shown in the tree but greyed out: heavy, generated, rarely what you want.
HEAVY_DIRS = frozenset({
    ".git", "node_modules", ".venv", "venv", "__pycache__", ".next",
    ".pytest_cache", ".ruff_cache", ".mypy_cache", "dist", "build", ".cache",
})


class FileAccessError(Exception):
    """A request the panel must refuse. `status` is the HTTP code to send."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def is_secret(name: str) -> bool:
    """Whether a file holds secrets: `.env` and its variants (`.env.local`)."""
    return name == ".env" or name.startswith(".env.")


def resolve_inside(root: str, relative: str) -> Path:
    """The real path of `relative` within `root`, or a refusal.

    Args:
        root: The open folder.
        relative: A path inside it, as the UI sends it -- "src/app.py",
            "/src/app.py" (the agent's virtual style), or "" for the root.

    Returns:
        The resolved absolute path.

    Raises:
        FileAccessError: 403 if it lands outside `root` or names a secret,
            404 if it doesn't exist.
    """
    base = Path(root).resolve()
    target = (base / relative.lstrip("/\\")).resolve()
    if not target.is_relative_to(base):
        raise FileAccessError(403, "That path is outside the open folder.")
    if any(is_secret(part) for part in target.relative_to(base).parts):
        raise FileAccessError(403, ".env files are not shown.")
    if not target.exists():
        raise FileAccessError(404, "No such file or folder.")
    return target


@dataclasses.dataclass(frozen=True)
class Entry:
    """One row of the tree.

    Attributes:
        name: The file or folder name.
        path: Its path relative to the open folder, with forward slashes.
        type: "dir" or "file".
        size: Bytes, for files; None for folders.
        heavy: A folder that is usually generated (node_modules, .git...).
        locked: Listed so the tree is honest, but can't be opened -- a
            `.env`, or a link that points outside the folder.
    """

    name: str
    path: str
    type: str
    size: int | None
    heavy: bool = False
    locked: bool = False


def list_dir(root: str, relative: str = "") -> tuple[list[Entry], bool]:
    """One folder's entries, folders first, each group by name.

    Args:
        root: The open folder.
        relative: The folder to list, relative to `root`.

    Returns:
        The entries, and whether the list was cut at MAX_ENTRIES.

    Raises:
        FileAccessError: as resolve_inside, or 400 if it isn't a folder.
    """
    base = Path(root).resolve()
    folder = resolve_inside(root, relative)
    if not folder.is_dir():
        raise FileAccessError(400, "That is a file, not a folder.")

    entries: list[Entry] = []
    try:
        children = list(os.scandir(folder))
    except PermissionError as e:
        raise FileAccessError(403, "No permission to read that folder.") from e

    for child in children:
        path = Path(child.path)
        rel = path.relative_to(base).as_posix()
        try:
            real = path.resolve()
            escapes = not real.is_relative_to(base)
            is_dir = real.is_dir()
            size = None if is_dir else real.stat().st_size
        except OSError:  # a broken link, or one we may not stat
            escapes, is_dir, size = True, False, None
        entries.append(Entry(
            name=child.name,
            path=rel,
            type="dir" if is_dir else "file",
            size=size,
            heavy=is_dir and child.name in HEAVY_DIRS,
            locked=escapes or is_secret(child.name),
        ))

    entries.sort(key=lambda e: (e.type != "dir", e.name.lower()))
    return entries[:MAX_ENTRIES], len(entries) > MAX_ENTRIES


def read_text(root: str, relative: str) -> str:
    """A file's contents, for the viewer.

    Raises:
        FileAccessError: as resolve_inside; 400 for a folder; 413 past
            MAX_VIEW_BYTES; 415 for anything that isn't UTF-8 text.
    """
    target = resolve_inside(root, relative)
    if target.is_dir():
        raise FileAccessError(400, "That is a folder.")
    size = target.stat().st_size
    if size > MAX_VIEW_BYTES:
        raise FileAccessError(413, f"Too large to show ({size / 1024 / 1024:.1f} MB).")
    data = target.read_bytes()
    # NUL bytes never appear in text; checking a prefix catches images and
    # archives without decoding the whole thing first.
    if b"\x00" in data[:8192]:
        raise FileAccessError(415, "Binary file — not shown.")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise FileAccessError(415, "Not UTF-8 text — not shown.") from e


def folder_key(folder: str) -> str:
    """The open folder as one URL path segment, for preview URLs.

    It has to live in the *path*, not the query string: a previewed page's
    relative links ("style.css") resolve against the path and would drop a
    query, so its CSS and images would never load.
    """
    return base64.urlsafe_b64encode(folder.encode()).decode().rstrip("=")


def folder_from_key(key: str) -> str:
    """Reverses folder_key.

    Raises:
        FileAccessError: 400 if `key` isn't one, or names no folder.
    """
    try:
        folder = base64.urlsafe_b64decode(key + "=" * (-len(key) % 4)).decode()
    except (binascii.Error, UnicodeDecodeError, ValueError) as e:
        raise FileAccessError(400, "Bad preview link.") from e
    if not Path(folder).is_dir():
        raise FileAccessError(400, "That folder no longer exists.")
    return folder
