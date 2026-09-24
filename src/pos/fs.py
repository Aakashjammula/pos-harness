"""Opens the real native OS folder dialog.

The browser's File System Access API can hand back a folder's *name* but
never a real filesystem path -- that's deliberate, for security, and no
in-browser picker can work around it. But this is a local desktop app:
backend and browser run on the same machine, so the backend can pop the
native dialog itself and hand back a real path, which is what
FilesystemBackend actually needs.

Tk >= 8.6.3 already uses Windows' modern IFileDialog under the hood (the
same dialog Explorer shows), so this is a native picker there, not a dated
Tk widget -- but the process has to declare itself DPI-aware or Windows
bitmap-stretches the dialog on a scaled display and it looks blurry. macOS
Tk likewise calls the real Cocoa panel.

Linux is the exception: Tk draws its own themed widget there instead of
shelling out to the desktop's own dialog, so it looks dated and clashes
with the rest of the UI. `zenity` (GTK/GNOME) or `kdialog` (KDE) give the
real thing, so on Linux we try those first and only fall back to Tk if
neither is installed.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
from typing import Any


class PickerUnavailableError(RuntimeError):
    """Raised when there is no desktop to open a dialog on."""


_NO_LINUX_TOOL = object()


def _pick_folder_linux() -> str | None | object:
    """Tries the desktop's own dialog via zenity or kdialog.

    Returns:
        The chosen path; None if the dialog opened but was cancelled;
        `_NO_LINUX_TOOL` if neither tool is installed, so the caller can
        tell "cancelled" apart from "fall back to Tk".
    """
    if shutil.which("zenity"):
        cmd = ["zenity", "--file-selection", "--directory", "--title=Open folder"]
    elif shutil.which("kdialog"):
        cmd = ["kdialog", "--getexistingdirectory", "."]
    else:
        return _NO_LINUX_TOOL
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    return proc.stdout.strip() or None


def _fix_win_hidpi() -> None:
    """Declares this process DPI-aware, so Tk dialogs render crisply.

    Lifted from CPython's own idlelib/util.py (`fix_win_hidpi`), which is
    what IDLE and turtledemo use. `OleDLL` rather than `windll` is
    deliberate: it raises OSError on a failed HRESULT, hence the catch.
    Must run before any Tk operation.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        process_system_dpi_aware = 1  # Int required.
        ctypes.OleDLL("shcore").SetProcessDpiAwareness(process_system_dpi_aware)
    except (ImportError, AttributeError, OSError):
        pass


def pick_folder() -> str | None:
    """Opens the native folder dialog and blocks until it's closed.

    Runs on a separate thread from Tk's own, since FastAPI's sync route
    handlers already run in a threadpool -- this just gives Tk a clean,
    dedicated thread of its own rather than sharing whichever worker
    thread happened to pick up the request.

    Returns:
        The chosen absolute path, or None if the dialog was cancelled.

    Raises:
        PickerUnavailableError: When Tk is missing or there is no display --
            a container, or a machine reached over SSH. The caller turns
            this into a message telling you to set the folder another way.
    """
    if sys.platform.startswith("linux"):
        path = _pick_folder_linux()
        if path is not _NO_LINUX_TOOL:
            return path

    # Imported here, not at module scope: a slim container image has no Tk
    # libraries, and importing tkinter there raises ImportError, which at
    # module scope would take the whole server down rather than just this
    # one endpoint.
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as e:
        msg = "no desktop available to open a folder dialog on"
        raise PickerUnavailableError(msg) from e

    result: dict[str, str] = {}
    failure: dict[str, BaseException] = {}

    def run() -> None:
        _fix_win_hidpi()  # before any Tk call, per CPython's own comment
        try:
            _open(tk, filedialog, result)
        except Exception as e:  # noqa: BLE001 -- re-raised on the calling thread
            failure["error"] = e

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    if "error" in failure:
        msg = "could not open a folder dialog on this machine"
        raise PickerUnavailableError(msg) from failure["error"]
    return result.get("path")


def _open(tk: Any, filedialog: Any, result: dict[str, str]) -> None:
    """Shows the dialog and records what was chosen.

    Runs on its own thread; see pick_folder. `tk` and `filedialog` are
    passed in because they are imported there, lazily.

    Args:
        tk: The `tkinter` module.
        filedialog: The `tkinter.filedialog` module.
        result: Written to with key "path" if a folder was chosen.
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)  # otherwise it can open behind the browser
    path = filedialog.askdirectory(title="Open folder")
    root.destroy()
    if path:
        result["path"] = path
