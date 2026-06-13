#!/usr/bin/env python3
"""CalorieCal backend entry point.

Starts two services:
  1. OpenNI depth server (port 8765) — tries to open the RGB-D camera.
     If the camera is absent or OpenNI fails, falls back to 2D silently.
  2. Action recognition server (port 8766) — FastAPI + WebSocket.

Usage (from project root):
    conda activate CalorieCalPose
    python backend/main.py

Swagger UI:   http://127.0.0.1:8766/docs
ReDoc:        http://127.0.0.1:8766/redoc
WebSocket:    ws://127.0.0.1:8766/ws
Health check: http://127.0.0.1:8766/health
Depth health: http://127.0.0.1:8765/health
"""

from __future__ import annotations

import atexit
import signal
import subprocess
import sys
import time
from pathlib import Path

# Make project root importable regardless of invocation directory.
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import urllib.request
import urllib.error

import uvicorn

from backend.action_server import DEFAULT_HOST, DEFAULT_PORT  # noqa: E402

DEPTH_SERVER_SCRIPT = _PROJECT_ROOT / "task5-SAM 3D Body尝试" / "openni_depth_server.py"
DEPTH_HOST = "127.0.0.1"
DEPTH_PORT = 8765
DEPTH_PROBE_TIMEOUT = 5.0   # seconds to wait for depth server to come up
DEPTH_PROBE_INTERVAL = 0.3  # seconds between health-check retries

_depth_proc: subprocess.Popen | None = None  # type: ignore[type-arg]


def _probe_depth_server() -> bool:
    """Return True if the depth server health endpoint responds OK."""
    try:
        with urllib.request.urlopen(
            f"http://{DEPTH_HOST}:{DEPTH_PORT}/health",
            timeout=2,
        ) as resp:
            return resp.status == 200
    except Exception:
        return False


def start_depth_server(python_exe: str) -> bool:
    """Launch openni_depth_server.py as a subprocess.

    Returns True if the server came up successfully, False if the camera
    is unavailable or startup failed (caller should continue in 2D mode).
    """
    global _depth_proc

    if not DEPTH_SERVER_SCRIPT.exists():
        print(f"[depth] script not found: {DEPTH_SERVER_SCRIPT}")
        return False

    # Already running?
    if _probe_depth_server():
        print(f"[depth] already running on port {DEPTH_PORT}")
        return True

    print(f"[depth] starting depth server on port {DEPTH_PORT} …")
    try:
        _depth_proc = subprocess.Popen(
            [
                python_exe,
                str(DEPTH_SERVER_SCRIPT),
                "--host", DEPTH_HOST,
                "--port", str(DEPTH_PORT),
            ],
            cwd=str(_PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        print(f"[depth] failed to launch subprocess: {exc}")
        return False

    # Wait for it to respond
    deadline = time.monotonic() + DEPTH_PROBE_TIMEOUT
    while time.monotonic() < deadline:
        time.sleep(DEPTH_PROBE_INTERVAL)
        if _depth_proc.poll() is not None:
            # Process already exited — camera likely not connected
            out = _depth_proc.stdout.read().decode(errors="replace") if _depth_proc.stdout else ""
            print(f"[depth] process exited early (camera not available?) — 2D mode\n  {out.strip()}")
            _depth_proc = None
            return False
        if _probe_depth_server():
            print(f"[depth] depth server ready  →  http://{DEPTH_HOST}:{DEPTH_PORT}/health")
            return True

    # Timed out
    print("[depth] timed out waiting for depth server — 2D mode")
    _stop_depth_server()
    return False


def _stop_depth_server() -> None:
    global _depth_proc
    if _depth_proc and _depth_proc.poll() is None:
        print("[depth] stopping depth server …")
        _depth_proc.terminate()
        try:
            _depth_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _depth_proc.kill()
    _depth_proc = None


def main() -> None:
    parser = argparse.ArgumentParser(description="CalorieCal backend")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload (dev)")
    parser.add_argument("--no-depth", action="store_true", help="Skip depth server, force 2D mode")
    args = parser.parse_args()

    # -- Depth server (optional, degrades gracefully) -----------------------
    depth_ok = False
    if not args.no_depth:
        depth_ok = start_depth_server(sys.executable)
        if not depth_ok:
            print("[depth] running in 2D mode (no depth data)")
    else:
        print("[depth] --no-depth flag set, skipping depth server")

    # Clean up depth process on exit
    atexit.register(_stop_depth_server)
    for sig in (signal.SIGINT, signal.SIGTERM):
        original = signal.getsignal(sig)
        def _handler(signum: int, frame: object, _orig: object = original) -> None:
            _stop_depth_server()
            if callable(_orig):
                _orig(signum, frame)  # type: ignore[call-arg]
            sys.exit(0)
        signal.signal(sig, _handler)

    # -- Action server -------------------------------------------------------
    print()
    print(f"  Swagger UI  →  http://{args.host}:{args.port}/docs")
    print(f"  ReDoc       →  http://{args.host}:{args.port}/redoc")
    print(f"  WebSocket   →  ws://{args.host}:{args.port}/ws")
    print(f"  Health      →  http://{args.host}:{args.port}/health")
    print(f"  Depth       →  {'http://' + DEPTH_HOST + ':' + str(DEPTH_PORT) + '/health' if depth_ok else '2D mode (no depth camera)'}")
    print()

    uvicorn.run(
        "backend.action_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
