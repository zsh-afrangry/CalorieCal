#!/usr/bin/env python3
"""Batch-evaluate recorded Task10 action outputs.

This script reuses recognize_recorded_actions.py on existing pose output
directories and writes one compact report for positives and confusions.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import offline_action_state_machine as action_sm
import recognize_recorded_actions as recorded


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOTS = [
    SCRIPT_DIR / "datasets" / "pose" / "recorded_actions",
    SCRIPT_DIR / "datasets" / "pose" / "videos",
]
DEFAULT_SPEC_DIR = SCRIPT_DIR.parent / "backend" / "action_engine" / "action_specs"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "datasets" / "pose" / "reports"

ACTION_PATTERNS = [
    ("jumping_jack", ["开合跳", "jumping_jack", "jumpingjack"]),
    ("squat", ["深蹲", "squat"]),
    ("rest", ["静息", "rest", "idle", "任意"]),
]
SAMPLE_TYPE_PATTERNS = [
    ("positive", ["正样本", "positive", "pos"]),
    ("confusion", ["混淆", "confusion"]),
    ("negative", ["负样本", "negative", "neg"]),
]
VIEW_PATTERNS = [
    ("front", ["正面", "front"]),
    ("side", ["侧面", "side"]),
    ("diagonal", ["斜侧", "斜面", "diagonal", "oblique"]),
]

CSV_COLUMNS = [
    "sample_id",
    "output_dir",
    "label_action",
    "sample_type",
    "view",
    "expected_squat",
    "actual_squat",
    "squat_standard",
    "squat_shallow",
    "expected_jumping_jack",
    "actual_jumping_jack",
    "status",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-evaluate recorded Task10 action recognizer outputs.")
    parser.add_argument(
        "--input-root",
        action="append",
        type=Path,
        default=None,
        help="Pose output root to scan. Can be repeated.",
    )
    parser.add_argument("--spec-dir", type=Path, default=DEFAULT_SPEC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default="recorded_action_eval")
    parser.add_argument("--buffer-sec", type=float, default=3.0)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_pattern(text: str, patterns: list[tuple[str, list[str]]]) -> str:
    text_lower = text.lower()
    for value, keys in patterns:
        if any(key.lower() in text_lower for key in keys):
            return value
    return "unknown"


def find_pose_dirs(roots: list[Path]) -> list[Path]:
    dirs: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.glob("*/features.csv"):
            dirs.add(path.parent)
    return sorted(dirs)


def infer_meta(output_dir: Path) -> dict[str, str]:
    summary = read_json(output_dir / "summary.json") if (output_dir / "summary.json").exists() else {}
    source_video = str(summary.get("source_video") or "")
    text = f"{output_dir.name} {source_video}"
    return {
        "sample_id": output_dir.name,
        "output_dir": str(output_dir),
        "label_action": str(summary.get("label") or "") or find_pattern(text, ACTION_PATTERNS),
        "sample_type": find_pattern(text, SAMPLE_TYPE_PATTERNS),
        "view": str(summary.get("view") or "") or find_pattern(text, VIEW_PATTERNS),
    }


def expected_count(meta: dict[str, str], action: str) -> int | None:
    text = meta["sample_id"]
    sample_type = meta["sample_type"]
    label = meta["label_action"]
    if sample_type in {"negative", "confusion"}:
        return 0 if label in {action, "rest", "unknown"} else None
    if sample_type != "positive":
        return None
    if action == "squat":
        match = re.search(r"(\d+)\s*个\s*深蹲|(\d+)\s*深蹲|深蹲.*?(\d+)\s*个", text)
        if not match and label == action:
            match = re.search(r"(\d+)\s*个", text)
    elif action == "jumping_jack":
        match = re.search(r"(\d+)\s*个\s*开合跳|(\d+)\s*开合跳|开合跳.*?(\d+)\s*个", text)
        if not match and label == action:
            match = re.search(r"(\d+)\s*个", text)
    else:
        match = None
    if match:
        values = [value for value in match.groups() if value]
        if values:
            return int(values[0])
    return 1 if label == action else 0


def evaluate_dir(output_dir: Path, specs: list[dict[str, Any]], buffer_sec: float) -> dict[str, Any]:
    meta = infer_meta(output_dir)
    rows = [recorded.numeric_row(row) for row in recorded.read_csv_rows(output_dir / "features.csv")]
    sequence, summary = recorded.evaluate_rows(rows, specs, meta["view"], buffer_sec)
    recorded.apply_recorded_squat_refinement(rows, sequence, summary, specs, meta["view"])

    squat = summary.get("actions", {}).get("squat", {})
    jumping_jack = summary.get("actions", {}).get("jumping_jack", {})
    expected_squat = expected_count(meta, "squat")
    expected_jj = expected_count(meta, "jumping_jack")
    actual_squat = int(squat.get("count") or 0)
    actual_jj = int(jumping_jack.get("count") or 0)

    notes = []
    status = "pass"
    if expected_squat is not None and actual_squat != expected_squat:
        status = "review"
        notes.append(f"squat expected {expected_squat}, got {actual_squat}")
    if expected_jj is not None and actual_jj != expected_jj:
        status = "review"
        notes.append(f"jumping_jack expected {expected_jj}, got {actual_jj}")
    if meta["sample_type"] in {"negative", "confusion"} and (actual_squat > 0 or actual_jj > 0):
        status = "fail"
        notes.append("confusion/negative sample produced action count")

    return {
        **meta,
        "expected_squat": "" if expected_squat is None else expected_squat,
        "actual_squat": actual_squat,
        "squat_standard": int(squat.get("standard_count") or 0),
        "squat_shallow": int(squat.get("shallow_count") or 0),
        "expected_jumping_jack": "" if expected_jj is None else expected_jj,
        "actual_jumping_jack": actual_jj,
        "status": status,
        "notes": "; ".join(notes),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Task10 Recorded Action Evaluation",
        "",
        f"- Created: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Samples: {len(rows)}",
        f"- Status: {dict(Counter(str(row.get('status')) for row in rows))}",
        "",
        "## Review / Fail",
        "",
    ]
    flagged = [row for row in rows if row.get("status") != "pass"]
    if not flagged:
        lines.append("- none")
    for row in flagged:
        lines.append(
            f"- `{row['sample_id']}` status={row['status']} "
            f"squat={row['actual_squat']} standard={row['squat_standard']} shallow={row['squat_shallow']} "
            f"jumping_jack={row['actual_jumping_jack']} | {row['notes']}"
        )
    lines.extend(["", "## All Samples", ""])
    for row in rows:
        lines.append(
            f"- `{row['sample_id']}` type={row['sample_type']} view={row['view']} "
            f"squat={row['actual_squat']}({row['squat_standard']}/{row['squat_shallow']}) "
            f"jumping_jack={row['actual_jumping_jack']} status={row['status']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    roots = args.input_root if args.input_root else DEFAULT_INPUT_ROOTS
    specs = action_sm.load_specs(args.spec_dir)
    output_dirs = find_pose_dirs(roots)
    rows = [evaluate_dir(output_dir, specs, args.buffer_sec) for output_dir in output_dirs]
    csv_path = args.output_dir / f"{args.name}.csv"
    md_path = args.output_dir / f"{args.name}.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    print(f"Evaluated samples: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    print(f"Status: {dict(Counter(str(row.get('status')) for row in rows))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
