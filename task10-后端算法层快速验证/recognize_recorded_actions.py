#!/usr/bin/env python3
"""Recognize squat and jumping-jack counts from a recorded video.

The script accepts either a raw video file or an existing pose output directory.
For raw videos it first runs the same pose extraction pipeline used by
video_pose_trajectory.py, then evaluates the extracted features with rolling
windows so one clip can contain multiple actions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import offline_action_state_machine as action_sm
import video_pose_trajectory as pose_tools


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_SPEC_DIR = PROJECT_ROOT / "backend" / "action_engine" / "action_specs"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "datasets" / "pose" / "recorded_actions"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recognize Task10 actions from a recorded video.")
    parser.add_argument("input", type=Path, help="Raw video file or existing pose output directory.")
    parser.add_argument("--view", choices=["front", "side", "diagonal"], default="front")
    parser.add_argument("--spec-dir", type=Path, default=DEFAULT_SPEC_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--backend", choices=["auto", "solutions", "tasks"], default="auto")
    parser.add_argument(
        "--model-asset-path",
        type=Path,
        default=SCRIPT_DIR / "models" / "pose_landmarker_lite.task",
    )
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--visibility-threshold", type=float, default=0.5)
    parser.add_argument("--resize-width", type=int, default=960)
    parser.add_argument("--buffer-sec", type=float, default=3.0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--render-overlay", action="store_true", help="Write mixed_action_overlay.mp4.")
    parser.add_argument("--draw-extraction-overlay", action="store_true", help="Also write extraction overlay.mp4.")
    parser.add_argument("--no-snapshots", action="store_true", help="Disable extraction snapshots.")
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_float(value: Any) -> float | None:
    return action_sm.safe_float(value)


def numeric_row(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = dict(row)
    for key, value in list(out.items()):
        if key in {"frame_index", "processed_index"}:
            try:
                out[key] = int(value)
            except (TypeError, ValueError):
                pass
        elif key == "pose_detected":
            out[key] = str(value).lower() == "true"
        else:
            num = safe_float(value)
            if num is not None:
                out[key] = num
    return out


def row_timestamp(row: dict[str, Any]) -> float:
    return float(safe_float(row.get("timestamp_sec")) or 0.0)


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
        self.count = 0
        self.seen_high = False
        self.high_run = 0
        self.low_run = 0
        self.count_events: list[dict[str, Any]] = []
        self.quality_counts: defaultdict[str, int] = defaultdict(int)

    def update(
        self,
        stage: str,
        can_count: bool,
        timestamp_sec: float,
        frame_index: int,
        event_data: dict[str, Any] | None = None,
    ) -> bool:
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
                    counted = True
                    event = {
                        "count": self.count,
                        "timestamp_sec": timestamp_sec,
                        "frame_index": frame_index,
                    }
                    if event_data:
                        event.update(event_data)
                    quality = str(event.get("quality") or "counted")
                    self.quality_counts[quality] += 1
                    self.count_events.append(event)
                self.seen_high = False
        else:
            self.high_run = 0
            self.low_run = 0
        return counted


def feature_valid_ratio(rows: list[dict[str, Any]], feature: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if safe_float(row.get(feature)) is not None) / len(rows)


def ratio(count: int, total: int) -> float:
    return count / total if total else 0.0


def mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def frame_window(rows: list[dict[str, Any]], center_sec: float, radius_sec: float) -> list[dict[str, Any]]:
    return [row for row in rows if abs(row_timestamp(row) - center_sec) <= radius_sec]


def smoothed_feature(rows: list[dict[str, Any]], feature: str, window_frames: int = 7) -> list[float | None]:
    half = max(0, window_frames // 2)
    values = [safe_float(row.get(feature)) for row in rows]
    smoothed: list[float | None] = []
    for index in range(len(values)):
        nums = [
            value
            for value in values[max(0, index - half) : min(len(values), index + half + 1)]
            if value is not None
        ]
        smoothed.append(mean_or_none(nums))
    return smoothed


def numeric_values(rows: list[dict[str, Any]], feature: str) -> list[float]:
    return [value for value in (safe_float(row.get(feature)) for row in rows) if value is not None]


def row_range(rows: list[dict[str, Any]], feature: str) -> float | None:
    values = numeric_values(rows, feature)
    if not values:
        return None
    return max(values) - min(values)


def event_summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = rolling_summary_row(rows)
    return summary


def find_recorded_squat_attempts(
    rows: list[dict[str, Any]],
    view: str,
    spec: dict[str, Any],
    start_sec: float = 0.0,
) -> list[dict[str, Any]]:
    """Post-process recorded clips into squat attempts using local knee-depth troughs."""
    if view not in {"front", "side"} or not rows:
        return []

    rows = [row for row in rows if row_timestamp(row) >= start_sec]
    if not rows:
        return []

    smoothed_knee = smoothed_feature(rows, "mean_knee_angle", window_frames=7)
    candidate_max_knee = 160.0 if view == "front" else 85.0
    min_knee_range = 18.0 if view == "front" else 40.0
    min_hip_range = 0.22 if view == "front" else 0.0
    min_gap_sec = 1.20
    local_radius_sec = 0.35
    event_radius_sec = 0.75

    attempts: list[dict[str, Any]] = []
    last_event_sec = -999.0
    for index, knee_value in enumerate(smoothed_knee):
        if knee_value is None or knee_value > candidate_max_knee:
            continue
        timestamp_sec = row_timestamp(rows[index])
        if timestamp_sec - last_event_sec < min_gap_sec:
            continue

        local_values = [
            smoothed
            for row, smoothed in zip(rows, smoothed_knee)
            if smoothed is not None and abs(row_timestamp(row) - timestamp_sec) <= local_radius_sec
        ]
        if local_values and knee_value > min(local_values) + 1e-9:
            continue

        window_rows = frame_window(rows, timestamp_sec, event_radius_sec)
        if not window_rows:
            continue
        knee_values = numeric_values(window_rows, "mean_knee_angle")
        if not knee_values or min(knee_values) > candidate_max_knee:
            continue
        knee_range = row_range(window_rows, "mean_knee_angle")
        hip_range = row_range(window_rows, "hip_height_above_ankles_torso")
        foot_range = row_range(window_rows, "foot_spread_shoulder_ratio") or 0.0
        hands_values = numeric_values(window_rows, "hands_above_shoulders_ratio")
        hands_mean = mean_or_none(hands_values) or 0.0

        if knee_range is None or knee_range < min_knee_range:
            continue
        if view == "front" and (hip_range is None or hip_range < min_hip_range):
            continue
        if hip_range is not None and hip_range > 2.0:
            continue
        if hands_mean > 0.20 and foot_range > 0.70:
            continue

        summary_row = event_summary_row(window_rows)
        supported, standard_passed, standard_reason = action_sm.summary_rules_pass(summary_row, spec, view)
        total_reason = (
            f"local_knee_min={min(knee_values):.3f} <= {candidate_max_knee:.3f}; "
            f"local_knee_range={knee_range:.3f} >= {min_knee_range:.3f}; "
            f"local_hip_range={hip_range if hip_range is not None else 0.0:.3f} >= {min_hip_range:.3f}"
        )
        quality = action_sm.squat_quality_from_summary(summary_row, view, bool(supported and standard_passed), standard_reason)
        attempts.append(
            {
                "count": len(attempts) + 1,
                "timestamp_sec": timestamp_sec,
                "frame_index": int(rows[index].get("frame_index") or 0),
                "quality": quality.get("quality"),
                "quality_reason": quality.get("quality_reason"),
                "advice": quality.get("advice", ""),
                "total_gate_reason": total_reason,
                "standard_gate_reason": standard_reason,
                "evidence": {
                    "mean_knee_angle_min": min(knee_values),
                    "mean_knee_angle_range": knee_range,
                    "hip_height_range": hip_range,
                    "hands_above_shoulders_mean": hands_mean,
                    "foot_spread_range": foot_range,
                },
            }
        )
        last_event_sec = timestamp_sec
    return attempts


def apply_recorded_squat_refinement(
    rows: list[dict[str, Any]],
    sequence: list[dict[str, Any]],
    summary: dict[str, Any],
    specs: list[dict[str, Any]],
    view: str,
) -> None:
    squat_spec = next((spec for spec in specs if spec.get("action_name") == "squat"), None)
    if not squat_spec:
        return
    squat_summary = summary.get("actions", {}).get("squat")
    jumping_events = summary.get("actions", {}).get("jumping_jack", {}).get("count_events", [])
    start_sec = 0.0
    if jumping_events:
        start_sec = max(float(event.get("timestamp_sec") or 0.0) for event in jumping_events) + 1.0
        raw_squat_events = squat_summary.get("count_events", []) if squat_summary else []
        if raw_squat_events:
            first_squat_sec = min(float(event.get("timestamp_sec") or 0.0) for event in raw_squat_events)
            start_sec = min(start_sec, max(0.0, first_squat_sec - 4.0))
    attempts = find_recorded_squat_attempts(rows, view, squat_spec, start_sec=start_sec)
    if not squat_summary or len(attempts) <= int(squat_summary.get("count") or 0):
        return

    original_count = int(squat_summary.get("count") or 0)
    original_events = list(squat_summary.get("count_events") or [])
    raw_standard_events = [
        event
        for event in original_events
        if str(event.get("quality") or "standard") == "standard"
    ]
    raw_standard_index = 0
    for attempt in attempts:
        attempt_sec = float(attempt.get("timestamp_sec") or 0.0)
        while raw_standard_index < len(raw_standard_events):
            raw_sec = float(raw_standard_events[raw_standard_index].get("timestamp_sec") or 0.0)
            if raw_sec < attempt_sec - 0.10:
                raw_standard_index += 1
                continue
            if attempt_sec - 0.10 <= raw_sec <= attempt_sec + 1.70:
                attempt["quality"] = "standard"
                attempt["quality_reason"] = "matched_strict_state_machine_event"
                attempt["advice"] = ""
                raw_standard_index += 1
            break

    quality_counts: defaultdict[str, int] = defaultdict(int)
    for event in attempts:
        quality_counts[str(event.get("quality") or "counted")] += 1

    squat_summary["raw_state_machine_count"] = original_count
    squat_summary["raw_state_machine_events"] = original_events
    squat_summary["count"] = len(attempts)
    squat_summary["count_events"] = attempts
    squat_summary["quality_counts"] = dict(quality_counts)
    squat_summary["standard_count"] = int(quality_counts.get("standard", 0))
    squat_summary["shallow_count"] = int(quality_counts.get("shallow", 0))
    squat_summary["recorded_refinement"] = {
        "enabled": True,
        "method": "local_knee_trough_with_motion_gates",
        "reason": "Recorded clips can be segmented after seeing the full sequence; realtime still uses online state machine.",
    }

    attempt_index = 0
    for seq_row in sequence:
        timestamp_sec = safe_float(seq_row.get("timestamp_sec")) or 0.0
        while attempt_index < len(attempts) and timestamp_sec >= float(attempts[attempt_index]["timestamp_sec"]):
            attempt_index += 1
        for result in seq_row.get("results", []):
            if result.get("action") == "squat":
                result["raw_state_machine_count"] = result.get("count")
                result["count"] = attempt_index
                result["recorded_refined_count"] = True


def ensure_pose_output(args: argparse.Namespace) -> Path:
    input_path = args.input
    if input_path.is_dir():
        if not (input_path / "features.csv").exists():
            raise SystemExit(f"Pose output directory does not contain features.csv: {input_path}")
        return input_path

    if not input_path.exists() or input_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise SystemExit(f"Input must be a video file or pose output directory: {input_path}")

    extract_args = argparse.Namespace(
        input=input_path,
        backend=args.backend,
        model_asset_path=args.model_asset_path,
        output_root=args.output_root,
        recursive=False,
        frame_stride=args.frame_stride,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        max_frames=args.max_frames,
        label="mixed",
        view=args.view,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        visibility_threshold=args.visibility_threshold,
        resize_width=args.resize_width,
        draw_overlay=args.draw_extraction_overlay,
        no_snapshots=args.no_snapshots,
        snapshot_interval_sec=1.0,
        snapshot_frames="",
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f"Extracting pose: {input_path}")
    out_dir = pose_tools.process_video(extract_args, input_path)
    print(f"Pose output: {out_dir}")
    return out_dir


def evaluate_rows(
    rows: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    view: str,
    buffer_sec: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    counters = {str(spec.get("action_name")): OnlineCounter(str(spec.get("action_name")), spec) for spec in specs}
    action_gate_seen: dict[str, int] = defaultdict(int)
    action_gate_ok: dict[str, int] = defaultdict(int)
    action_stage_counts: dict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    action_reasons: dict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    sequence = []
    buffer: deque[dict[str, Any]] = deque()

    for row in rows:
        timestamp_sec = row_timestamp(row)
        buffer.append(row)
        while buffer and timestamp_sec - row_timestamp(buffer[0]) > buffer_sec:
            buffer.popleft()

        buffer_rows = list(buffer)
        summary_row = rolling_summary_row(buffer_rows)
        frame_results = []
        for spec in specs:
            action = str(spec.get("action_name"))
            supported, summary_passed, reason = action_sm.summary_rules_pass(summary_row, spec, view)
            count_supported = supported
            count_passed = summary_passed
            count_reason = reason
            event_data = None
            if action == "squat":
                count_supported, count_passed, count_reason = action_sm.squat_total_rules_pass(summary_row, spec, view)
                event_data = action_sm.squat_quality_from_summary(summary_row, view, summary_passed, reason)
            scores, stages, rolling_cycles = action_sm.run_total_count_state_machine(buffer_rows, spec)
            score = next((x for x in reversed(scores) if x is not None), None)
            stage = next((x for x in reversed(stages) if x != "unknown"), stages[-1] if stages else "unknown")
            frame_index = int(row.get("frame_index") or 0)
            counted = counters[action].update(
                stage,
                bool(count_supported and count_passed),
                timestamp_sec,
                frame_index,
                event_data,
            )

            action_gate_seen[action] += 1
            action_stage_counts[action][stage] += 1
            if count_passed:
                action_gate_ok[action] += 1
            elif count_reason:
                action_reasons[action][count_reason] += 1

            frame_results.append(
                {
                    "action": action,
                    "supported": supported,
                    "summary_rules_passed": summary_passed,
                    "reason": reason,
                    "count_rules_passed": count_passed,
                    "count_reason": count_reason,
                    "quality": event_data.get("quality") if event_data else None,
                    "quality_reason": event_data.get("quality_reason") if event_data else None,
                    "stage": stage,
                    "score": score,
                    "rolling_cycles": rolling_cycles,
                    "count": counters[action].count,
                    "counted": counted,
                }
            )

        sequence.append(
            {
                "frame_index": int(row.get("frame_index") or 0),
                "timestamp_sec": timestamp_sec,
                "pose_detected": bool(row.get("pose_detected")),
                "results": frame_results,
            }
        )

    summary = {
        "frames": len(rows),
        "duration_sec": row_timestamp(rows[-1]) - row_timestamp(rows[0]) if len(rows) >= 2 else 0.0,
        "pose_detected_ratio": ratio(sum(1 for row in rows if row.get("pose_detected")), len(rows)),
        "feature_valid_ratio": {
            "mean_knee_angle": feature_valid_ratio(rows, "mean_knee_angle"),
            "foot_spread_shoulder_ratio": feature_valid_ratio(rows, "foot_spread_shoulder_ratio"),
            "hip_height_above_ankles_torso": feature_valid_ratio(rows, "hip_height_above_ankles_torso"),
            "hands_above_shoulders_ratio": feature_valid_ratio(rows, "hands_above_shoulders_ratio"),
        },
        "actions": {},
    }
    for action, counter in counters.items():
        reasons = sorted(action_reasons[action].items(), key=lambda item: item[1], reverse=True)
        summary["actions"][action] = {
            "count": counter.count,
            "count_events": counter.count_events,
            "quality_counts": dict(counter.quality_counts),
            "gate_ok_ratio": ratio(action_gate_ok[action], action_gate_seen[action]),
            "stage_counts": dict(action_stage_counts[action]),
            "top_gate_reasons": [{"reason": reason, "count": count} for reason, count in reasons[:8]],
        }
        if action == "squat":
            quality_counts = counter.quality_counts
            summary["actions"][action]["standard_count"] = int(quality_counts.get("standard", 0))
            summary["actions"][action]["shallow_count"] = int(quality_counts.get("shallow", 0))
    return sequence, summary


def write_sequence(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_overlay(output_dir: Path, sequence: list[dict[str, Any]], args: argparse.Namespace) -> Path:
    cv2, _mp, _np = pose_tools.import_runtime_deps()
    pose_rows = []
    with (output_dir / "pose_sequence.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pose_rows.append(json.loads(line))

    summary = read_json(output_dir / "summary.json")
    source_video = Path(str(summary.get("source_video") or ""))
    if not source_video.is_absolute():
        source_video = PROJECT_ROOT / source_video

    sequence_by_frame = {int(row["frame_index"]): row for row in sequence}
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source video for overlay: {source_video}")

    first_row = pose_rows[0] if pose_rows else {}
    width = int(first_row.get("image_width") or summary.get("input_video", {}).get("width") or 640)
    height = int(first_row.get("image_height") or summary.get("input_video", {}).get("height") or 480)
    fps = float(summary.get("input_video", {}).get("fps") or 30.0) / max(1, int(summary.get("processing", {}).get("frame_stride") or 1))

    out_path = output_dir / "mixed_action_overlay.mp4"
    if out_path.exists():
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = output_dir / f"mixed_action_overlay_{stamp}.mp4"

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create overlay video: {out_path}")

    for pose_row in pose_rows:
        frame_index = int(pose_row["frame_index"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        pose_tools.draw_landmarks(cv2, frame, pose_row.get("landmarks") or {}, args.visibility_threshold)
        seq = sequence_by_frame.get(frame_index, {})
        draw_mixed_panel(cv2, frame, seq)
        writer.write(frame)

    writer.release()
    cap.release()
    return out_path


def draw_mixed_panel(cv2: Any, frame: Any, sequence_row: dict[str, Any]) -> None:
    h, w = frame.shape[:2]
    panel_h = 112
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    timestamp = safe_float(sequence_row.get("timestamp_sec"))
    timestamp_text = "" if timestamp is None else f" t={timestamp:.2f}s"
    cv2.putText(
        frame,
        f"Task10 recorded action recognition{timestamp_text}",
        (14, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    y = 58
    for result in sequence_row.get("results", [])[:2]:
        action = result.get("action")
        stage = result.get("stage")
        score = safe_float(result.get("score"))
        score_text = "n/a" if score is None else f"{score:.2f}"
        count = result.get("count")
        gate_ok = bool(result.get("count_rules_passed"))
        quality = result.get("quality")
        quality_text = f" {quality}" if action == "squat" and quality else ""
        color = (60, 220, 80) if gate_ok else (70, 130, 255)
        cv2.putText(
            frame,
            f"{action}: stage={stage} score={score_text} count={count}{quality_text}",
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 26


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Task10 Recorded Action Recognition",
        "",
        f"- Created: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Frames: {summary.get('frames')}",
        f"- Duration sec: {summary.get('duration_sec'):.3f}",
        f"- Pose detected ratio: {summary.get('pose_detected_ratio'):.3f}",
        "",
        "## Feature Valid Ratios",
        "",
    ]
    for name, value in summary.get("feature_valid_ratio", {}).items():
        lines.append(f"- {name}: {value:.3f}")
    lines.extend(["", "## Counts", ""])
    for action, item in summary.get("actions", {}).items():
        lines.append(f"- {action}: {item.get('count', 0)}")
        if action == "squat":
            lines.append(
                "  quality: "
                f"standard={item.get('standard_count', 0)}, "
                f"shallow={item.get('shallow_count', 0)}"
            )
        events = item.get("count_events", [])
        if events:
            event_text = ", ".join(
                f"#{event['count']}@{event['timestamp_sec']:.2f}s"
                + (f"[{event.get('quality')}]" if event.get("quality") else "")
                for event in events
            )
            lines.append(f"  events: {event_text}")
    lines.extend(["", "## Gate Reasons", ""])
    for action, item in summary.get("actions", {}).items():
        reasons = item.get("top_gate_reasons", [])
        if not reasons:
            continue
        lines.append(f"### {action}")
        for reason in reasons[:5]:
            lines.append(f"- {reason['count']}: {reason['reason']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = ensure_pose_output(args)
    specs = action_sm.load_specs(args.spec_dir)
    rows = [numeric_row(row) for row in read_csv_rows(output_dir / "features.csv")]
    if not rows:
        raise SystemExit(f"No feature rows found: {output_dir / 'features.csv'}")

    sequence, summary = evaluate_rows(rows, specs, args.view, args.buffer_sec)
    apply_recorded_squat_refinement(rows, sequence, summary, specs, args.view)
    summary["pose_output_dir"] = str(output_dir)
    summary["view"] = args.view
    summary["buffer_sec"] = args.buffer_sec

    sequence_path = output_dir / "mixed_action_sequence.jsonl"
    summary_path = output_dir / "mixed_action_summary.json"
    md_path = output_dir / "mixed_action_summary.md"
    write_sequence(sequence_path, sequence)
    write_json(summary_path, summary)
    write_markdown(md_path, summary)

    overlay_path = None
    if args.render_overlay:
        overlay_path = render_overlay(output_dir, sequence, args)
        summary["mixed_action_overlay"] = str(overlay_path)
        write_json(summary_path, summary)

    print(f"Pose output: {output_dir}")
    print(f"Sequence: {sequence_path}")
    print(f"Summary: {summary_path}")
    print(f"Markdown: {md_path}")
    if overlay_path:
        print(f"Overlay: {overlay_path}")
    for action, item in summary.get("actions", {}).items():
        print(f"{action}: count={item.get('count', 0)}")
        if action == "squat":
            print(
                "squat quality: "
                f"standard={item.get('standard_count', 0)}, "
                f"shallow={item.get('shallow_count', 0)}"
            )
        events = item.get("count_events", [])
        if events:
            event_text = ", ".join(
                f"#{event['count']}@{event['timestamp_sec']:.2f}s"
                + (f"[{event.get('quality')}]" if event.get("quality") else "")
                for event in events
            )
            print(f"{action} events: {event_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
