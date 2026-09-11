"""Beta launcher: one process serving the advisor and drawing the overlay.

Unlike app/overlay.py (which spawns the venv's uvicorn as a subprocess), this
runs the FastAPI server in an in-process thread so it survives PyInstaller
freezing — the beta bundle has no venv or python.exe to spawn. Testers need no
Riot API key: the live path talks only to the League client's local API
(https://127.0.0.1:2999) and Data Dragon's public CDN.

Dev run:   .venv\\Scripts\\pythonw.exe -m app.advisor
Bundle:    BuildAdvisor.exe (see scripts/build_beta.ps1)

League must be in Borderless/Windowed display mode — exclusive Fullscreen
cannot be drawn over (same limitation as every overlay app).
"""

from __future__ import annotations

import ctypes
import io
import sys
import threading
import time
import webbrowser

import httpx

# PyInstaller --windowed apps have no console: sys.stdout/stderr are None and
# anything that touches them (uvicorn's log formatter calls .isatty()) crashes.
if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()

HOST = "127.0.0.1"
PORT = 8000
BASE = f"http://{HOST}:{PORT}"
WIDTH, HEIGHT, MARGIN = 350, 540, 16


def server_up() -> bool:
    # static route on purpose: /api/status scans the multi-GB events table on
    # the dev box and outlives any sane health-check timeout
    try:
        return httpx.get(f"{BASE}/live", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


def start_server() -> None:
    if server_up():
        return
    import uvicorn

    from app.main import app

    # log_config=None: uvicorn's default dictConfig builds console formatters
    # that break under --windowed even with the stdout shim above
    config = uvicorn.Config(app, host=HOST, port=PORT, log_level="warning", log_config=None)
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(60):
        if server_up():
            return
        time.sleep(0.5)
    raise SystemExit(f"Could not start the advisor server on port {PORT}.")


def main() -> None:
    start_server()
    try:
        import webview
    except ImportError:
        webview = None
    if webview is None:
        # no EdgeWebView2/pywebview: still usable as a browser panel
        webbrowser.open(f"{BASE}/live")
        while True:
            time.sleep(3600)

    class Api:
        def close(self) -> None:
            for w in webview.windows:
                w.destroy()

    screen_w = ctypes.windll.user32.GetSystemMetrics(0) if sys.platform == "win32" else 1920
    webview.create_window(
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
    webview.start()


if __name__ == "__main__":
    main()
