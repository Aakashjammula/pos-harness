"""Builds the UI into the wheel.

The frontend is a Next.js app, but nothing Node-related is needed to *run*
this package -- only to build it. `next build` with `output: 'export'`
produces plain HTML/CSS/JS, which is copied to `src/pos/web/` and served by
FastAPI. So a user installs Python and nothing else.

This is the same arrangement JupyterLab and Streamlit use (a build hook
that runs npm during the wheel build) rather than committing build output
to the repo, which goes stale.

If `npm` or the frontend folder is missing, the hook leaves any existing
`src/pos/web/` alone and the package still builds -- it just serves no UI,
which is exactly what a development checkout wants.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
WEB_DIR = Path(__file__).resolve().parent / "src" / "pos" / "web"


class FrontendBuildHook(BuildHookInterface):
    """Runs `npm ci && npm run build`, then copies `out/` into the package."""

    PLUGIN_NAME = "frontend"

    def initialize(self, version: str, build_data: dict) -> None:
        """Builds the UI before the wheel's files are collected.

        Args:
            version: "standard" or "editable" -- an editable install runs
                from the checkout, where the dev server serves the UI, so
                there is nothing to bundle.
            build_data: Hatchling's build state. Unused.
        """
        if version == "editable":
            return
        if not FRONTEND_DIR.is_dir() or shutil.which("npm") is None:
            self.app.display_warning(
                "npm or frontend/ not found -- building without a bundled UI. "
                "The API will work; the app will serve no pages."
            )
            return

        npm = shutil.which("npm")
        # `npm ci` deletes node_modules first, which fails on Windows if a dev
        # server is holding a native module open -- so only use it for a clean
        # checkout, where it is both correct and the reproducible choice.
        install = ["ci"] if not (FRONTEND_DIR / "node_modules").is_dir() else ["install", "--no-audit", "--no-fund"]
        self.app.display_info(f"building the frontend (npm {install[0]} && npm run build)")
        subprocess.run([npm, *install], cwd=FRONTEND_DIR, check=True)
        subprocess.run([npm, "run", "build"], cwd=FRONTEND_DIR, check=True)

        out = FRONTEND_DIR / "out"
        if not out.is_dir():
            msg = f"expected {out} after `next build` -- is output: 'export' set in next.config.ts?"
            raise RuntimeError(msg)

        # Replace rather than merge, so a renamed or deleted page can't
        # linger from an earlier build.
        if WEB_DIR.exists():
            shutil.rmtree(WEB_DIR)
        shutil.copytree(out, WEB_DIR)
        self._bundled = True
        self.app.display_info(f"bundled the UI into {WEB_DIR.relative_to(Path.cwd())}")

    def finalize(self, version: str, build_data: dict, artifact_path: str) -> None:
        """Removes the bundled UI once it is inside the wheel.

        Left behind, `src/pos/web/` would make a source checkout look
        packaged: app.py would serve that stale build instead of enabling
        CORS for the Next dev server, and `npm run dev` changes would stop
        appearing. The wheel already has its own copy.

        Args:
            version: The build version, as passed to `initialize`.
            build_data: Hatchling's build state. Unused.
            artifact_path: The built wheel. Unused.
        """
        if getattr(self, "_bundled", False) and WEB_DIR.exists():
            shutil.rmtree(WEB_DIR)
            self.app.display_info("removed the bundled UI from the checkout")
