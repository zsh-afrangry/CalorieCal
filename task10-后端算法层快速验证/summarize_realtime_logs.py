#!/usr/bin/env python3
"""Summarize Task10 realtime validation JSONL logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = SCRIPT_DIR / "datasets" / "pose" / "realtime_logs"
DEFAULT_REPORT_DIR = SCRIPT_DIR / "datasets" / "pose" / "reports"

FEATURES = [
    "mean_visibility",
    "core_visible_ratio",
    "mean_knee_angle",
    "foot_spread_shoulder_ratio",
    "hip_height_above_ankles_torso",
    "hands_above_shoulders_ratio",
]

REPORT_COLUMNS = [
    "log_file",
    "frames",
    "duration_sec",
    "pose_detected_ratio",
    "mean_fps",
    "mean_latency_ms",
    "p95_latency_ms",
    "mean_visibility",
    "mean_core_visible_ratio",
    "knee_angle_valid_ratio",
    "foot_spread_valid_ratio",
    "hip_height_valid_ratio",
    "hands_above_valid_ratio",
    "jumping_jack_count",
    "squat_count",
    "jumping_jack_gate_ok_ratio",
    "squat_gate_ok_ratio",
    "top_quality_hint",
    "top_gate_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize realtime Task10 JSONL logs.")
    parser.add_argument("log_dir", nargs="?", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--name", default="realtime_summary")
    parser.add_argument("--latest", type=int, default=None, help="Only summarize the latest N non-empty logs.")
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    low = int(math.floor(index))
    high = int(math.ceil(index))
    if low == high:
        return ordered[low]
    weight = index - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def read_log(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ratio(count: int, total: int) -> float | None:
    if total <= 0:
        return None
    return count / total


def top_counter(counter: Counter[str]) -> str:
    if not counter:
        return ""
    value, count = counter.most_common(1)[0]
    return f"{value} ({count})"


def summarize_log(path: Path) -> dict[str, Any]:
    rows = read_log(path)
    frames = len(rows)
    if not rows:
        return {
            "log_file": str(path),
            "frames": 0,
            "duration_sec": 0,
        }

    timestamps = [safe_float(row.get("timestamp_sec")) for row in rows]
    timestamps = [x for x in timestamps if x is not None]
    duration = (max(timestamps) - min(timestamps)) if len(timestamps) >= 2 else 0.0
    fps_values = [safe_float(row.get("fps")) for row in rows]
    fps_values = [x for x in fps_values if x is not None]
    latency_values = [safe_float(row.get("latency_ms")) for row in rows]
    latency_values = [x for x in latency_values if x is not None]

    feature_values: dict[str, list[float]] = {}
    feature_valid_counts: dict[str, int] = {}
    for feature in FEATURES:
        values = [safe_float(row.get("features", {}).get(feature)) for row in rows]
        nums = [x for x in values if x is not None]
        feature_values[feature] = nums
        feature_valid_counts[feature] = len(nums)

    action_counts: dict[str, int] = defaultdict(int)
    action_seen: dict[str, int] = defaultdict(int)
    action_gate_ok: dict[str, int] = defaultdict(int)
    gate_reasons: Counter[str] = Counter()
    for row in rows:
        for result in row.get("results", []):
            action = str(result.get("action") or "")
            if not action:
                continue
            action_seen[action] += 1
            action_counts[action] = max(action_counts[action], int(result.get("count") or 0))
            if result.get("summary_rules_passed"):
                action_gate_ok[action] += 1
            else:
                reason = str(result.get("reason") or "")
                if reason:
                    gate_reasons[reason] += 1

    hint_counter: Counter[str] = Counter()
    for row in rows:
        hint = str(row.get("quality_hint") or "")
        if hint:
            hint_counter[hint] += 1

    pose_detected_count = sum(1 for row in rows if row.get("pose_detected"))
    return {
        "log_file": str(path),
        "frames": frames,
        "duration_sec": duration,
        "pose_detected_ratio": ratio(pose_detected_count, frames),
        "mean_fps": mean(fps_values),
        "mean_latency_ms": mean(latency_values),
        "p95_latency_ms": percentile(latency_values, 0.95),
        "mean_visibility": mean(feature_values["mean_visibility"]),
        "mean_core_visible_ratio": mean(feature_values["core_visible_ratio"]),
        "knee_angle_valid_ratio": ratio(feature_valid_counts["mean_knee_angle"], frames),
        "foot_spread_valid_ratio": ratio(feature_valid_counts["foot_spread_shoulder_ratio"], frames),
        "hip_height_valid_ratio": ratio(feature_valid_counts["hip_height_above_ankles_torso"], frames),
        "hands_above_valid_ratio": ratio(feature_valid_counts["hands_above_shoulders_ratio"], frames),
        "jumping_jack_count": action_counts.get("jumping_jack", 0),
        "squat_count": action_counts.get("squat", 0),
        "jumping_jack_gate_ok_ratio": ratio(action_gate_ok.get("jumping_jack", 0), action_seen.get("jumping_jack", 0)),
        "squat_gate_ok_ratio": ratio(action_gate_ok.get("squat", 0), action_seen.get("squat", 0)),
        "top_quality_hint": top_counter(hint_counter),
        "top_gate_reason": top_counter(gate_reasons),
    }


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key)) for key in REPORT_COLUMNS})


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Task10 Realtime Summary",
        "",
        f"- Created: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Logs: {len(rows)}",
        "",
    ]
    for row in rows:
        name = Path(str(row.get("log_file"))).name
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"- Frames: {row.get('frames', 0)}")
        lines.append(f"- Duration sec: {format_value(row.get('duration_sec'))}")
        lines.append(f"- Pose detected ratio: {format_value(row.get('pose_detected_ratio'))}")
        lines.append(f"- Mean FPS: {format_value(row.get('mean_fps'))}")
        lines.append(f"- Mean latency ms: {format_value(row.get('mean_latency_ms'))}")
        lines.append(f"- P95 latency ms: {format_value(row.get('p95_latency_ms'))}")
        lines.append(
            "- Feature valid ratios: "
            f"knee={format_value(row.get('knee_angle_valid_ratio'))}, "
            f"foot={format_value(row.get('foot_spread_valid_ratio'))}, "
            f"hip={format_value(row.get('hip_height_valid_ratio'))}, "
            f"hands={format_value(row.get('hands_above_valid_ratio'))}"
        )
        lines.append(
            "- Counts: "
            f"jumping_jack={row.get('jumping_jack_count', 0)}, "
            f"squat={row.get('squat_count', 0)}"
        )
        lines.append(
            "- Gate ok ratios: "
            f"jumping_jack={format_value(row.get('jumping_jack_gate_ok_ratio'))}, "
            f"squat={format_value(row.get('squat_gate_ok_ratio'))}"
        )
        if row.get("top_quality_hint"):
            lines.append(f"- Top quality hint: {row.get('top_quality_hint')}")
        if row.get("top_gate_reason"):
            lines.append(f"- Top gate reason: {row.get('top_gate_reason')}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.log_dir.exists():
        raise SystemExit(f"Log dir does not exist: {args.log_dir}")

    logs = [p for p in sorted(args.log_dir.glob("realtime_*.jsonl"), key=lambda x: x.stat().st_mtime) if p.stat().st_size > 0]
    if args.latest is not None:
        logs = logs[-args.latest :]
    rows = [summarize_log(path) for path in logs]

    args.report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.report_dir / f"{args.name}.csv"
    md_path = args.report_dir / f"{args.name}.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    print(f"Logs: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
