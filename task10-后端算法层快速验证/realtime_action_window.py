#!/usr/bin/env python3
"""Realtime webcam validation window for Task10 action specs.

This is a validation tool, not the final product API. It runs MediaPipe Pose,
extracts the same compact features as video_pose_trajectory.py, evaluates the
current rolling buffer with the draft action specs, and displays action stage,
score, count, latency, and gate reasons.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import offline_action_state_machine as action_sm
import video_pose_trajectory as pose_tools


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_SPEC_DIR = PROJECT_ROOT / "backend" / "action_engine" / "action_specs"
DEFAULT_LOG_DIR = SCRIPT_DIR / "datasets" / "pose" / "realtime_logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run realtime Task10 action validation from a webcam.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--view", choices=["front", "side", "diagonal"], default="front")
    parser.add_argument(
        "--backend",
        choices=["auto", "solutions", "tasks"],
        default="auto",
        help="Pose backend. Use tasks if mediapipe.solutions is unavailable.",
    )
    parser.add_argument(
        "--model-asset-path",
        type=Path,
        default=SCRIPT_DIR / "models" / "pose_landmarker_lite.task",
        help="MediaPipe Tasks Pose Landmarker .task model path.",
    )
    parser.add_argument("--spec-dir", type=Path, default=DEFAULT_SPEC_DIR)
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--visibility-threshold", type=float, default=0.5)
    parser.add_argument("--buffer-sec", type=float, default=3.0, help="Rolling feature buffer length.")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--camera-fourcc", default="MJPG", help="Preferred camera FOURCC. Use empty string to skip.")
    parser.add_argument("--read-fail-limit", type=int, default=30, help="Stop after this many consecutive camera read failures.")
    parser.add_argument("--resize-width", type=int, default=960, help="Resize frames before pose extraction and display.")
    parser.add_argument("--mirror", action="store_true", help="Mirror webcam frames before processing.")
    parser.add_argument("--record-log", action="store_true", help="Write per-frame realtime JSONL log.")
    parser.add_argument("--record-video", action="store_true", help="Write debug overlay video.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for index in range(1000):
        suffix = f"_{stamp}" if index == 0 else f"_{stamp}_{index:03d}"
        candidate = path.with_name(f"{path.stem}{suffix}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Failed to create unique path for {path}")


def safe_float(value: Any) -> float | None:
    return action_sm.safe_float(value)


def landmarks_to_dict(pose_landmark_list: list[Any] | None, width: int, height: int) -> dict[str, dict[str, float]]:
    landmarks: dict[str, dict[str, float]] = {}
    if not pose_landmark_list:
        return landmarks
    for i, lm in enumerate(pose_landmark_list[: len(pose_tools.LANDMARK_NAMES)]):
        name = pose_tools.LANDMARK_NAMES[i]
        visibility = getattr(lm, "visibility", None)
        if visibility is None:
            visibility = getattr(lm, "presence", 1.0)
        landmarks[name] = {
            "index": i,
            "x": float(lm.x),
            "y": float(lm.y),
            "z": float(lm.z),
            "visibility": float(visibility),
            "presence": float(getattr(lm, "presence", visibility)),
            "px": float(lm.x * width),
            "py": float(lm.y * height),
        }
    return landmarks


def row_from_features(
    frame_index: int,
    timestamp_sec: float,
    pose_detected: bool,
    features: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "frame_index": frame_index,
        "timestamp_sec": timestamp_sec,
        "pose_detected": pose_detected,
    }
    for name in pose_tools.CSV_FEATURES:
        row[name] = features.get(name)
    return row


def rolling_summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    stats = action_sm.feature_stats(rows)
    detected = sum(1 for row in rows if row.get("pose_detected"))
    visibility_values = [safe_float(row.get("mean_visibility")) for row in rows]
    visibility_values = [x for x in visibility_values if x is not None]
    core_values = [safe_float(row.get("core_visible_ratio")) for row in rows]
    core_values = [x for x in core_values if x is not None]
    return {
        **stats,
        "detected_ratio": detected / len(rows),
        "mean_visibility": sum(visibility_values) / len(visibility_values) if visibility_values else None,
        "mean_core_visible_ratio": sum(core_values) / len(core_values) if core_values else None,
    }


def action_stage_names(action: str) -> tuple[str, str, int, int]:
    if action == "squat":
        return ("stand", "down", 2, 2)
    if action == "jumping_jack":
        return ("closed", "open", 1, 1)
    return ("low", "high", 1, 1)


class OnlineCounter:
    def __init__(self, action: str, spec: dict[str, Any]) -> None:
        self.action = action
        low_stage, high_stage, default_high, default_low = action_stage_names(action)
        params = spec.get("frame_state_machine", {})
        if action == "squat":
            self.min_high_hold = int(params.get("min_down_hold_frames", default_high))
            self.min_low_hold = int(params.get("min_return_to_stand_frames", default_low))
        elif action == "jumping_jack":
            self.min_high_hold = int(params.get("min_open_hold_frames", default_high))
            self.min_low_hold = int(params.get("min_return_to_closed_frames", default_low))
        else:
            self.min_high_hold = default_high
            self.min_low_hold = default_low
        self.low_stage = low_stage
        self.high_stage = high_stage
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.seen_high = False
        self.high_run = 0
        self.low_run = 0
        self.last_count_timestamp = None

    def update(self, stage: str, can_count: bool, timestamp_sec: float) -> bool:
        counted = False
        if stage == self.high_stage:
            self.high_run += 1
            self.low_run = 0
            if self.high_run >= self.min_high_hold:
                self.seen_high = True
        elif stage == self.low_stage:
            self.low_run += 1
            self.high_run = 0
            if self.seen_high and self.low_run >= self.min_low_hold:
                if can_count:
                    self.count += 1
                    self.last_count_timestamp = timestamp_sec
                    counted = True
                self.seen_high = False
        else:
            self.high_run = 0
            self.low_run = 0
        return counted


class FpsMeter:
    def __init__(self) -> None:
        self.values: deque[float] = deque(maxlen=30)
        self.last_time = None

    def update(self) -> float | None:
        now = time.perf_counter()
        if self.last_time is not None:
            dt = now - self.last_time
            if dt > 0:
                self.values.append(1.0 / dt)
        self.last_time = now
        if not self.values:
            return None
        return sum(self.values) / len(self.values)


def score_color(score: float | None, gate_ok: bool, supported: bool) -> tuple[int, int, int]:
    if not supported:
        return (170, 170, 170)
    if not gate_ok:
        return (70, 130, 255)
    if score is not None and score >= 0.70:
        return (60, 220, 80)
    if score is not None and score >= 0.35:
        return (0, 220, 255)
    return (255, 210, 80)


def quality_hint(pose_detected: bool, features: dict[str, Any]) -> str:
    if not pose_detected:
        return "no pose detected"

    missing = []
    if safe_float(features.get("mean_knee_angle")) is None:
        missing.append("knees")
    if safe_float(features.get("foot_spread_shoulder_ratio")) is None:
        missing.append("feet")
    if safe_float(features.get("hip_height_above_ankles_torso")) is None:
        missing.append("hips+ankles")
    if missing:
        return "missing lower-body evidence: " + ", ".join(missing)

    mean_visibility = safe_float(features.get("mean_visibility"))
    core_visible = safe_float(features.get("core_visible_ratio"))
    if mean_visibility is not None and mean_visibility < 0.85:
        return f"low landmark visibility: {mean_visibility:.2f}"
    if core_visible is not None and core_visible < 0.85:
        return f"low core visibility: {core_visible:.2f}"
    return ""


def put_text(cv2: Any, frame: Any, text: str, x: int, y: int, color: tuple[int, int, int], scale: float = 0.55) -> None:
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_panel(
    cv2: Any,
    frame: Any,
    results: list[dict[str, Any]],
    fps: float | None,
    latency_ms: float,
    view: str,
    backend_name: str,
    hint: str,
) -> None:
    h, w = frame.shape[:2]
    panel_h = 132
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    fps_text = "n/a" if fps is None else f"{fps:.1f}"
    put_text(
        cv2,
        frame,
        f"Task10 realtime | view={view} | fps={fps_text} | latency={latency_ms:.1f}ms | {backend_name}",
        14,
        26,
        (235, 235, 235),
        0.55,
    )

    y = 58
    for result in results[:3]:
        action = result["action"]
        stage = result["stage"]
        score = result["score"]
        score_text = "n/a" if score is None else f"{score:.2f}"
        count = result["count"]
        supported = result["supported"]
        gate_ok = result["summary_rules_passed"]
        color = score_color(score, gate_ok, supported)
        if not supported:
            status = "unsupported"
        elif gate_ok:
            status = "gate-ok"
        else:
            status = "gated"
        put_text(
            cv2,
            frame,
            f"{action}: {status} stage={stage} score={score_text} count={count}",
            14,
            y,
            color,
            0.55,
        )
        y += 27

    reasons = [result["reason"] for result in results if result.get("reason") and not result["summary_rules_passed"]]
    if hint:
        put_text(cv2, frame, f"hint: {hint}", 14, min(panel_h - 14, h - 12), (210, 210, 210), 0.45)
    elif reasons:
        reason = " ".join(str(reasons[0]).split())
        if len(reason) > 105:
            reason = reason[:102] + "..."
        put_text(cv2, frame, f"reason: {reason}", 14, min(panel_h - 14, h - 12), (210, 210, 210), 0.45)


def evaluate_actions(
    rows: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    view: str,
    counters: dict[str, OnlineCounter],
    timestamp_sec: float,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    summary_row = rolling_summary_row(rows)
    results = []
    for spec in specs:
        action = str(spec.get("action_name"))
        supported, summary_passed, reason = action_sm.summary_rules_pass(summary_row, spec, view)
        scores, stages, rolling_cycles = action_sm.run_state_machine(rows, spec)
        score = next((x for x in reversed(scores) if x is not None), None)
        stage = next((x for x in reversed(stages) if x != "unknown"), stages[-1] if stages else "unknown")
        counter = counters[action]
        counted = counter.update(stage, bool(supported and summary_passed), timestamp_sec)
        results.append(
            {
                "action": action,
                "supported": supported,
                "summary_rules_passed": summary_passed,
                "reason": reason,
                "stage": stage,
                "score": score,
                "rolling_cycles": rolling_cycles,
                "count": counter.count,
                "counted": counted,
            }
        )
    return results


def write_log_line(log_file: Any, item: dict[str, Any]) -> None:
    if log_file is None:
        return
    log_file.write(json.dumps(item, ensure_ascii=False) + "\n")
    log_file.flush()


def main() -> int:
    args = parse_args()
    cv2, mp, _np = pose_tools.import_runtime_deps()
    specs = action_sm.load_specs(args.spec_dir)
    counters = {str(spec.get("action_name")): OnlineCounter(str(spec.get("action_name")), spec) for spec in specs}

    cap = cv2.VideoCapture(args.camera)
    if args.camera_fourcc:
        fourcc_text = args.camera_fourcc[:4].ljust(4)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc_text))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open camera index {args.camera}")

    estimator, backend_name = pose_tools.create_pose_estimator(mp, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = unique_path(args.output_dir / f"realtime_{stamp}.jsonl")
    video_path = unique_path(args.output_dir / f"realtime_overlay_{stamp}.mp4")
    log_file = log_path.open("w", encoding="utf-8") if args.record_log else None
    video_writer = None

    frame_rows: deque[dict[str, Any]] = deque()
    prev_landmarks: dict[str, dict[str, float]] | None = None
    fps_meter = FpsMeter()
    frame_index = 0
    started_at = time.perf_counter()
    last_timestamp_ms = -1
    consecutive_read_failures = 0
    pose_detected_frames = 0
    knee_valid_frames = 0
    foot_valid_frames = 0
    hip_height_valid_frames = 0

    print("Realtime window started.")
    print("Press q or ESC to quit. Press r to reset counts.")
    if args.record_log:
        print(f"Log: {log_path}")
    if args.record_video:
        print(f"Video: {video_path}")

    try:
        with estimator as pose_estimator:
            while True:
                ok, frame = cap.read()
                if not ok:
                    consecutive_read_failures += 1
                    if consecutive_read_failures >= args.read_fail_limit:
                        print(
                            "Camera read failed repeatedly: "
                            f"{consecutive_read_failures} consecutive failures. "
                            "Check camera index, USB connection, and other apps using /dev/video*."
                        )
                        break
                    time.sleep(0.05)
                    continue
                consecutive_read_failures = 0
                if args.max_frames is not None and frame_index >= args.max_frames:
                    break

                if args.mirror:
                    frame = cv2.flip(frame, 1)
                if args.resize_width and frame.shape[1] > args.resize_width:
                    frame = pose_tools.resize_frame(cv2, frame, args.resize_width)

                start = time.perf_counter()
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp_sec = time.perf_counter() - started_at
                timestamp_ms = max(int(timestamp_sec * 1000.0), last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms
                pose_landmark_list = pose_estimator.process(rgb, timestamp_ms)
                landmarks = landmarks_to_dict(pose_landmark_list, w, h)
                pose_detected = bool(landmarks)
                features = (
                    pose_tools.compute_features(landmarks, prev_landmarks, args.visibility_threshold)
                    if landmarks
                    else {}
                )
                hint = quality_hint(pose_detected, features)
                if pose_detected:
                    pose_detected_frames += 1
                if safe_float(features.get("mean_knee_angle")) is not None:
                    knee_valid_frames += 1
                if safe_float(features.get("foot_spread_shoulder_ratio")) is not None:
                    foot_valid_frames += 1
                if safe_float(features.get("hip_height_above_ankles_torso")) is not None:
                    hip_height_valid_frames += 1
                row = row_from_features(frame_index, timestamp_sec, pose_detected, features)
                frame_rows.append(row)
                while frame_rows and timestamp_sec - float(frame_rows[0]["timestamp_sec"]) > args.buffer_sec:
                    frame_rows.popleft()

                results = evaluate_actions(list(frame_rows), specs, args.view, counters, timestamp_sec)
                latency_ms = (time.perf_counter() - start) * 1000.0
                fps = fps_meter.update()

                pose_tools.draw_landmarks(cv2, frame, landmarks, args.visibility_threshold)
                draw_panel(cv2, frame, results, fps, latency_ms, args.view, backend_name, hint)

                if args.record_video:
                    if video_writer is None:
                        output_fps = fps if fps is not None and fps > 1 else 30.0
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(str(video_path), fourcc, output_fps, (w, h))
                    video_writer.write(frame)

                write_log_line(
                    log_file,
                    {
                        "frame_index": frame_index,
                        "timestamp_sec": timestamp_sec,
                        "pose_detected": pose_detected,
                        "latency_ms": latency_ms,
                        "fps": fps,
                        "quality_hint": hint,
                        "features": features,
                        "results": results,
                    },
                )

                cv2.imshow("Task10 Realtime Action Validation", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in {27, ord("q")}:
                    break
                if key == ord("r"):
                    for counter in counters.values():
                        counter.reset()
                    print("Counts reset.")

                prev_landmarks = landmarks if landmarks else prev_landmarks
                frame_index += 1
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if log_file is not None:
            log_file.close()
        cv2.destroyAllWindows()

    print("Done.")
    processed_frames = frame_index
    if processed_frames:
        print(
            "Quality summary: "
            f"pose={pose_detected_frames}/{processed_frames}, "
            f"knee_angle={knee_valid_frames}/{processed_frames}, "
            f"foot_spread={foot_valid_frames}/{processed_frames}, "
            f"hip_height={hip_height_valid_frames}/{processed_frames}"
        )
        if knee_valid_frames == 0 or foot_valid_frames == 0:
            print("Lower-body evidence was missing. Move the camera back/lower until the full body and feet are visible.")
    for action, counter in counters.items():
        print(f"{action}: count={counter.count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
