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
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
DEFAULT_LOG_DIR = _PROJECT_ROOT / "datasets" / "pose" / "count_event_logs"
ASYNC_VIEW_SYNC_WINDOW_MS = 150.0


@dataclass
class ViewFrame:
    view: str
    landmarks: dict[str, Any]
    timestamp_ms: float
    perf_timestamp_ms: float | None
    received_at_ms: float


@dataclass
class AsyncActionSession:
    session_id: str
    engine: RealtimeActionEngine
    latest_front: ViewFrame | None = None
    latest_side: ViewFrame | None = None
    view_dts: dict[str, deque[float]] = field(default_factory=lambda: {
        "front": deque(maxlen=30),
        "side": deque(maxlen=30),
    })
    subscribers: list[WebSocket] = field(default_factory=list)


_SESSIONS: dict[str, AsyncActionSession] = {}

app = FastAPI(title="CalorieCal Action Server", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _new_engine() -> RealtimeActionEngine:
    return RealtimeActionEngine(
        spec_dir=DEFAULT_SPEC_DIR,
        view="front",
        buffer_sec=3.0,
        fps_estimate=30.0,
        event_log_dir=DEFAULT_LOG_DIR,
    )


def _get_session(session_id: str) -> AsyncActionSession:
    sid = (session_id or "default").strip() or "default"
    session = _SESSIONS.get(sid)
    if session is None:
        session = AsyncActionSession(session_id=sid, engine=_new_engine())
        _SESSIONS[sid] = session
    return session


def _remember_view_frame(
    session: AsyncActionSession,
    frame: ViewFrame,
) -> None:
    prev = session.latest_front if frame.view == "front" else session.latest_side
    if prev is not None:
        dt = frame.timestamp_ms - prev.timestamp_ms
        if 10.0 < dt < 1000.0:
            session.view_dts[frame.view].append(dt)
    if frame.view == "front":
        session.latest_front = frame
    else:
        session.latest_side = frame


def _view_fps(session: AsyncActionSession, view: str) -> float | None:
    dts = session.view_dts.get(view)
    if not dts:
        return None
    avg = sum(dts) / len(dts)
    return round(1000.0 / avg, 2) if avg > 0 else None


def _frame_age_ms(frame: ViewFrame | None, now_ms: float) -> float | None:
    if frame is None:
        return None
    return round(now_ms - frame.timestamp_ms, 1)


def _is_recent(frame: ViewFrame | None, now_ms: float) -> bool:
    age = _frame_age_ms(frame, now_ms)
    return age is not None and 0 <= age <= ASYNC_VIEW_SYNC_WINDOW_MS


def _session_view_status(
    session: AsyncActionSession,
    now_ms: float,
) -> dict[str, Any]:
    return {
        "session_id": session.session_id,
        "sync_window_ms": ASYNC_VIEW_SYNC_WINDOW_MS,
        "front": {
            "present": session.latest_front is not None,
            "age_ms": _frame_age_ms(session.latest_front, now_ms),
            "fps": _view_fps(session, "front"),
        },
        "side": {
            "present": session.latest_side is not None,
            "age_ms": _frame_age_ms(session.latest_side, now_ms),
            "fps": _view_fps(session, "side"),
        },
    }


def _subscribe(session: AsyncActionSession, websocket: WebSocket) -> None:
    if websocket not in session.subscribers:
        session.subscribers.append(websocket)


def _unsubscribe_all(websocket: WebSocket) -> None:
    for session in _SESSIONS.values():
        if websocket in session.subscribers:
            session.subscribers.remove(websocket)


async def _broadcast_session_result(
    session: AsyncActionSession,
    result: dict[str, Any],
) -> None:
    if not session.subscribers:
        return
    payload = json.dumps(result, ensure_ascii=False)
    stale: list[WebSocket] = []
    for subscriber in list(session.subscribers):
        try:
            await subscriber.send_text(payload)
        except Exception:
            stale.append(subscriber)
    for subscriber in stale:
        if subscriber in session.subscribers:
            session.subscribers.remove(subscriber)


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

    engine = _new_engine()
    subscribed_session: AsyncActionSession | None = None

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
                session_id = msg.get("session_id")
                if session_id:
                    session = _get_session(str(session_id))
                    session.engine.set_user_config(
                        weight_kg=float(weight_kg) if weight_kg is not None else None,
                        height_cm=float(height_cm) if height_cm is not None else None,
                    )
                await websocket.send_text(json.dumps({
                    "type": "config_ack",
                    "weight_kg": engine._weight_kg,
                }))
                continue

            if msg.get("type") == "subscribe_session":
                session_id = str(msg.get("session_id") or "default")
                subscribed_session = _get_session(session_id)
                _subscribe(subscribed_session, websocket)
                await websocket.send_text(json.dumps({
                    "type": "session_subscribed",
                    "session_id": subscribed_session.session_id,
                }))
                continue

            # Debug recording control
            if msg.get("type") == "debug_start":
                session_id = msg.get("session_id")
                target_engine = _get_session(str(session_id)).engine if session_id else engine
                if msg.get("reset"):
                    target_engine.reset()
                target_engine.start_debug_recording()
                await websocket.send_text(json.dumps({
                    "type": "debug_started",
                    "reset": bool(msg.get("reset")),
                    "session_id": session_id,
                }))
                continue

            if msg.get("type") == "debug_stop":
                session_id = msg.get("session_id")
                target_engine = _get_session(str(session_id)).engine if session_id else engine
                debug_data = target_engine.stop_debug_recording()
                await websocket.send_text(json.dumps({
                    "type": "debug_result",
                    "data": debug_data,
                    "count": len(debug_data),
                    "session_id": session_id,
                }))
                continue

            if msg.get("type") == "view_frame":
                session_id = str(msg.get("session_id") or "default")
                view = str(msg.get("view") or "").lower()
                if view not in {"front", "side"}:
                    await websocket.send_text(
                        json.dumps({"error": "view_frame view must be front or side"})
                    )
                    continue
                landmarks = msg.get("landmarks")
                if not landmarks:
                    await websocket.send_text(
                        json.dumps({"error": "view_frame missing landmarks"})
                    )
                    continue
                timestamp_ms = msg.get("timestamp_ms") or (time.time() * 1000)
                session = _get_session(session_id)
                frame = ViewFrame(
                    view=view,
                    landmarks=landmarks,
                    timestamp_ms=float(timestamp_ms),
                    perf_timestamp_ms=(
                        float(msg["perf_timestamp_ms"])
                        if msg.get("perf_timestamp_ms") is not None
                        else None
                    ),
                    received_at_ms=time.time() * 1000,
                )
                _remember_view_frame(session, frame)

                if view == "front":
                    side = session.latest_side
                    side_landmarks = (
                        side.landmarks
                        if _is_recent(side, frame.timestamp_ms)
                        else None
                    )
                    result = session.engine.push_dual_frame(
                        landmarks_front=frame.landmarks,
                        landmarks_side=side_landmarks,
                        timestamp_ms=frame.timestamp_ms,
                    )
                    result["type"] = "session_result"
                    result["session_id"] = session.session_id
                    result["source_view"] = view
                    result["async_view_status"] = _session_view_status(
                        session,
                        frame.timestamp_ms,
                    )
                    await websocket.send_text(json.dumps(result, ensure_ascii=False))
                    await _broadcast_session_result(session, result)
                else:
                    ack = {
                        "type": "view_frame_ack",
                        "session_id": session.session_id,
                        "view": view,
                        "async_view_status": _session_view_status(
                            session,
                            frame.timestamp_ms,
                        ),
                    }
                    await websocket.send_text(json.dumps(ack, ensure_ascii=False))
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
        _unsubscribe_all(websocket)
    except Exception as exc:
        print(f"[action_server] error: {exc}")
        _unsubscribe_all(websocket)
        try:
            await websocket.send_text(json.dumps({"error": str(exc)}))
        except Exception:
            pass


if __name__ == "__main__":
    # Use backend/main.py as the intended entry point.
    from backend.main import main
    main()
