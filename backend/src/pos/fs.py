"""Opens the real native OS folder dialog.

The browser's File System Access API can hand back a folder's *name* but
never a real filesystem path -- that's deliberate, for security, and no
in-browser picker can work around it. But this is a local desktop app:
backend and browser run on the same machine, so the backend can pop the
native dialog itself and hand back a real path, which is what
FilesystemBackend actually needs.

Tk >= 8.6.3 already uses Windows' modern IFileDialog under the hood (the
same dialog Explorer shows), so this is a native picker, not a dated Tk
widget -- but the process has to declare itself DPI-aware or Windows
bitmap-stretches the dialog on a scaled display and it looks blurry.
"""

from __future__ import annotations

import sys
import threading
import tkinter as tk
from tkinter import filedialog


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
    """
    result: dict[str, str] = {}

    def run() -> None:
        _fix_win_hidpi()  # before any Tk call, per CPython's own comment
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)  # otherwise it can open behind the browser
        path = filedialog.askdirectory(title="Open folder")
        root.destroy()
        if path:
            result["path"] = path

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    return result.get("path")
