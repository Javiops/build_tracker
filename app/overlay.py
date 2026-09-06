"""In-game overlay: a frameless, always-on-top widget over the League client.

Wraps the /live panel in a native window (pywebview → EdgeWebView2), pinned
top-right like Porofessor/Blitz overlays. Starts the local server if it isn't
already running.

League must be in Borderless (or Windowed) display mode — exclusive Fullscreen
cannot be drawn over; that limitation is shared by every overlay app.

Run: .venv\\Scripts\\pythonw.exe -m app.overlay   (or start_overlay.bat)
Drag anywhere to move; the ✕ in the corner closes it.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
import time
from pathlib import Path

import httpx
import webview

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8000"
WIDTH, HEIGHT, MARGIN = 350, 540, 16


def server_up() -> bool:
    try:
        return httpx.get(f"{BASE}/api/status", timeout=1.5).status_code == 200
    except httpx.HTTPError:
        return False


def ensure_server() -> subprocess.Popen | None:
    if server_up():
        return None
    proc = subprocess.Popen(
        [str(ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "uvicorn", "app.main:app", "--port", "8000"],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    for _ in range(60):
        if server_up():
            return proc
        time.sleep(0.5)
    raise SystemExit("Could not start the advisor server on port 8000.")


class Api:
    def close(self) -> None:
        for w in webview.windows:
            w.destroy()


def main() -> None:
    owns_server = ensure_server()
    screen_w = ctypes.windll.user32.GetSystemMetrics(0) if sys.platform == "win32" else 1920
    window = webview.create_window(
        "Build Advisor",
        f"{BASE}/live?overlay=1",
        width=WIDTH,
        height=HEIGHT,
        x=screen_w - WIDTH - MARGIN,
        y=MARGIN,
        frameless=True,
        easy_drag=True,
        on_top=True,
        js_api=Api(),
    )
    try:
        webview.start()
    finally:
        if owns_server is not None:
            owns_server.terminate()


if __name__ == "__main__":
    main()
