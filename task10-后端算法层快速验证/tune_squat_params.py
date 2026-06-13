#!/usr/bin/env python3
"""Evaluate candidate front-view squat parameters against current Task10 assets."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path
from typing import Any

import offline_action_state_machine as action_sm
import recognize_recorded_actions as recorded


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_SPEC_DIR = PROJECT_ROOT / "backend" / "action_engine" / "action_specs"
DEFAULT_ASSET_ROOT = SCRIPT_DIR / "datasets" / "pose" / "videos"
DEFAULT_RECORDED_ROOT = SCRIPT_DIR / "datasets" / "pose" / "recorded_actions"
DEFAULT_REPORT_DIR = SCRIPT_DIR / "datasets" / "pose" / "reports"

REPORT_COLUMNS = [
    "name",
    "accept_knee",
    "reject_knee",
    "knee_range",
    "hip_range",
    "down_min",
    "stand_max",
    "mixed_squat_count",
    "mixed_jumping_jack_count",
    "asset_counted",
    "asset_passed",
    "asset_failed",
    "asset_accuracy",
    "failures",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune squat params against recorded and asset samples.")
    parser.add_argument(
        "--recorded-dir",
        type=Path,
        default=DEFAULT_RECORDED_ROOT / "人物1_混合_正面_10深蹲10开合跳",
    )
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--spec-dir", type=Path, default=DEFAULT_SPEC_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--name", default="squat_param_tuning")
    return parser.parse_args()


def read_features(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [recorded.numeric_row(row) for row in csv.DictReader(f)]


def set_squat_params(
    specs: list[dict[str, Any]],
    accept_knee: float,
    reject_knee: float,
    knee_range: float,
    hip_range: float,
    down_min: float,
    stand_max: float,
) -> list[dict[str, Any]]:
    specs2 = copy.deepcopy(specs)
    for spec in specs2:
        if spec.get("action_name") != "squat":
            continue
        front = spec["summary_rules"]["front"]
        for rule in front.get("accept_all", []):
            if rule.get("feature") == "mean_knee_angle_min":
                rule["value"] = accept_knee
            elif rule.get("feature") == "mean_knee_angle_range":
                rule["value"] = knee_range
            elif rule.get("feature") == "hip_height_range":
                rule["value"] = hip_range
        for rule in front.get("reject_any", []):
            if rule.get("feature") == "mean_knee_angle_min":
                rule["value"] = reject_knee
        spec["frame_state_machine"]["stage_thresholds"]["down_min"] = down_min
        spec["frame_state_machine"]["stage_thresholds"]["stand_max"] = stand_max
    return specs2


def candidate_specs(base_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        ("current", 120.0, 125.0, 45.0, 0.30, 0.75, 0.20),
        ("mild_depth", 130.0, 140.0, 45.0, 0.30, 0.75, 0.20),
        ("medium_depth", 135.0, 145.0, 45.0, 0.30, 0.75, 0.20),
        ("medium_depth_lower_down", 140.0, 150.0, 45.0, 0.30, 0.70, 0.20),
        ("medium_depth_loose_down", 140.0, 150.0, 45.0, 0.30, 0.65, 0.20),
        ("loose_depth_loose_down", 145.0, 155.0, 45.0, 0.30, 0.65, 0.20),
        ("loose_depth_more_stand", 145.0, 155.0, 45.0, 0.30, 0.65, 0.25),
        ("very_loose", 150.0, 160.0, 35.0, 0.25, 0.60, 0.25),
    ]
    out = []
    for name, accept, reject, knee_range, hip_range, down_min, stand_max in candidates:
        out.append(
            {
                "name": name,
                "accept_knee": accept,
                "reject_knee": reject,
                "knee_range": knee_range,
                "hip_range": hip_range,
                "down_min": down_min,
                "stand_max": stand_max,
                "specs": set_squat_params(base_specs, accept, reject, knee_range, hip_range, down_min, stand_max),
            }
        )
    return out


def evaluate_recorded(recorded_dir: Path, specs: list[dict[str, Any]]) -> dict[str, Any]:
    rows = read_features(recorded_dir / "features.csv")
    _sequence, summary = recorded.evaluate_rows(rows, specs, "front", 3.0)
    return {
        "mixed_squat_count": summary["actions"]["squat"]["count"],
        "mixed_jumping_jack_count": summary["actions"]["jumping_jack"]["count"],
        "mixed_squat_events": [round(event["timestamp_sec"], 2) for event in summary["actions"]["squat"]["count_events"]],
    }


def evaluate_assets(asset_root: Path, specs: list[dict[str, Any]]) -> dict[str, Any]:
    output_dirs = action_sm.find_output_dirs(asset_root)
    rows = []
    for output_dir in output_dirs:
        rows.extend(action_sm.process_output_dir(output_dir, specs))
    counted = [row for row in rows if row.get("status") != "skipped_unsupported_view"]
    passed = [row for row in counted if row.get("status") == "pass"]
    failed = [row for row in counted if row.get("status") == "fail"]
    failures = [
        f"{row['sample_id']}:{row['spec_action']}:expected={row['expected_positive']}:predicted={row['predicted_positive']}"
        for row in failed
    ]
    return {
        "asset_counted": len(counted),
        "asset_passed": len(passed),
        "asset_failed": len(failed),
        "asset_accuracy": len(passed) / len(counted) if counted else 0.0,
        "failures": "; ".join(failures[:10]),
    }


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, list):
        return ", ".join(str(x) for x in value)
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key, "")) for key in REPORT_COLUMNS})


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Task10 Squat Param Tuning",
        "",
        f"- Created: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| name | accept/reject knee | down/stand | mixed squat | mixed jj | asset pass | failures |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['accept_knee']}/{row['reject_knee']} | "
            f"{row['down_min']}/{row['stand_max']} | {row['mixed_squat_count']} | "
            f"{row['mixed_jumping_jack_count']} | {row['asset_passed']}/{row['asset_counted']} | "
            f"{row.get('failures', '')} |"
        )
    lines.append("")
    lines.append("## Mixed Squat Events")
    lines.append("")
    for row in rows:
        lines.append(f"- {row['name']}: {format_value(row.get('mixed_squat_events', []))}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    base_specs = action_sm.load_specs(args.spec_dir)
    rows = []
    for candidate in candidate_specs(base_specs):
        recorded_result = evaluate_recorded(args.recorded_dir, candidate["specs"])
        asset_result = evaluate_assets(args.asset_root, candidate["specs"])
        rows.append({**candidate, **recorded_result, **asset_result})

    csv_path = args.report_dir / f"{args.name}.csv"
    md_path = args.report_dir / f"{args.name}.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
