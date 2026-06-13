#!/usr/bin/env python3
"""FastAPI + WebSocket action recognition server.

Runs on port 8766 alongside the existing OpenNI depth server (port 8765).

Protocol
--------
Client → server (each video frame):
    {
        "frame_index": 123,
        "timestamp_ms": 4567.8,
        "landmarks": {
            "LEFT_SHOULDER":  {"x": 0.42, "y": 0.31, "z": -0.05, "visibility": 0.98},
            "RIGHT_SHOULDER": { ... },
            ...
        }
    }

Server → client (same frame):
    {
        "frame_index": 123,
        "actions": [
            {"name": "squat",        "count": 3, "stage": "down",   "score": 0.84},
            {"name": "jumping_jack", "count": 0, "stage": "closed", "score": 0.09}
        ],
        "features": { ... },
        "latency_ms": 3.2,
        "buffer_size": 72
    }

Startup
-------
    conda activate CalorieCalPose
    python backend/action_server.py

Or with uvicorn directly:
    uvicorn backend.action_server:app --host 127.0.0.1 --port 8766 --reload
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Resolve project root so this script works whether invoked from the repo root
# or from inside the backend/ directory.
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from backend.action_engine.realtime_engine import RealtimeActionEngine  # noqa: E402

DEFAULT_SPEC_DIR = _PROJECT_ROOT / "backend" / "action_engine" / "action_specs"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8766

app = FastAPI(title="CalorieCal Action Server", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "service": "calorie-cal-action-server",
        "spec_dir": str(DEFAULT_SPEC_DIR),
        "time": time.time(),
    }


@app.websocket("/ws")
async def action_ws(websocket: WebSocket) -> None:
    await websocket.accept()

    log_dir = _PROJECT_ROOT / "datasets" / "pose" / "count_event_logs"
    engine = RealtimeActionEngine(
        spec_dir=DEFAULT_SPEC_DIR,
        view="front",
        buffer_sec=3.0,
        fps_estimate=30.0,
        event_log_dir=log_dir,
    )

    client = websocket.client
    print(f"[action_server] connected: {client}")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError as exc:
                await websocket.send_text(
                    json.dumps({"error": f"invalid json: {exc}"})
                )
                continue

            # Config message: {"type": "config", "weight_kg": 70, "height_cm": 175}
            if msg.get("type") == "config":
                weight_kg = msg.get("weight_kg")
                height_cm = msg.get("height_cm")
                engine.set_user_config(
                    weight_kg=float(weight_kg) if weight_kg is not None else None,
                    height_cm=float(height_cm) if height_cm is not None else None,
                )
                await websocket.send_text(json.dumps({
                    "type": "config_ack",
                    "weight_kg": engine._weight_kg,
                }))
                continue

            # Debug recording control
            if msg.get("type") == "debug_start":
                if msg.get("reset"):
                    engine.reset()
                engine.start_debug_recording()
                await websocket.send_text(json.dumps({
                    "type": "debug_started",
                    "reset": bool(msg.get("reset")),
                }))
                continue

            if msg.get("type") == "debug_stop":
                debug_data = engine.stop_debug_recording()
                await websocket.send_text(json.dumps({
                    "type": "debug_result",
                    "data": debug_data,
                    "count": len(debug_data),
                }))
                continue

            if msg.get("type") == "dual_frame":
                front = msg.get("front")
                side = msg.get("side")
                if not front and not side:
                    await websocket.send_text(
                        json.dumps({"error": "invalid dual_frame format"})
                    )
                    continue
                timestamp_ms = (
                    (front or {}).get("timestamp_ms")
                    or (side or {}).get("timestamp_ms")
                )
                result = engine.push_dual_frame(
                    landmarks_front=(front or {}).get("landmarks"),
                    landmarks_side=(side or {}).get("landmarks"),
                    timestamp_ms=timestamp_ms,
                )
                frame_index = msg.get("frame_index")
                if frame_index is not None:
                    result["frame_index"] = frame_index
                await websocket.send_text(json.dumps(result, ensure_ascii=False))
                continue

            landmarks = msg.get("landmarks")
            if not landmarks:
                await websocket.send_text(
                    json.dumps({"error": "missing landmarks field"})
                )
                continue

            timestamp_ms = msg.get("timestamp_ms")
            result = engine.push_frame(landmarks, timestamp_ms=timestamp_ms)

            frame_index = msg.get("frame_index")
            if frame_index is not None:
                result["frame_index"] = frame_index

            await websocket.send_text(json.dumps(result, ensure_ascii=False))

    except WebSocketDisconnect:
        print(f"[action_server] disconnected: {client}")
    except Exception as exc:
        print(f"[action_server] error: {exc}")
        try:
            await websocket.send_text(json.dumps({"error": str(exc)}))
        except Exception:
            pass


if __name__ == "__main__":
    # Use backend/main.py as the intended entry point.
    from backend.main import main
    main()
