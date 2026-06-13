"""Realtime action recognition engine.

Accepts per-frame landmarks from the frontend (MediaPipe Pose, normalised 0-1
coordinates), computes the same features used by the offline state machine, and
runs the frame-level state machine from each action spec.

One RealtimeActionEngine instance per WebSocket connection so every user session
has its own rolling buffer and state.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

# Calorie estimator (optional — graceful degradation if not found)
try:
    _BACKEND_ROOT = Path(__file__).resolve().parent.parent
    if str(_BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(_BACKEND_ROOT.parent))
    from backend.calorie_engine.calorie_estimator import estimate_rep_kcal, estimate_lunge_kcal, load_met_params
    _MET_PARAMS = load_met_params()
    _CALORIE_ENGINE_AVAILABLE = True
except Exception:
    _CALORIE_ENGINE_AVAILABLE = False
    _MET_PARAMS = {}

# Default directory for count-event snapshots (can be overridden in __init__)
DEFAULT_EVENT_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "datasets" / "pose" / "count_event_logs"

# ---------------------------------------------------------------------------
# MediaPipe index → landmark name mapping (matches LANDMARK_INDEX in PoseView.vue)
# ---------------------------------------------------------------------------

MEDIAPIPE_INDEX_TO_NAME: dict[int, str] = {
    0: "NOSE",
    7: "LEFT_EAR",
    8: "RIGHT_EAR",
    11: "LEFT_SHOULDER",
    12: "RIGHT_SHOULDER",
    13: "LEFT_ELBOW",
    14: "RIGHT_ELBOW",
    15: "LEFT_WRIST",
    16: "RIGHT_WRIST",
    23: "LEFT_HIP",
    24: "RIGHT_HIP",
    25: "LEFT_KNEE",
    26: "RIGHT_KNEE",
    27: "LEFT_ANKLE",
    28: "RIGHT_ANKLE",
    29: "LEFT_HEEL",
    30: "RIGHT_HEEL",
}

CORE_LANDMARKS = [
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW",    "RIGHT_ELBOW",
    "LEFT_WRIST",    "RIGHT_WRIST",
    "LEFT_HIP",      "RIGHT_HIP",
    "LEFT_KNEE",     "RIGHT_KNEE",
    "LEFT_ANKLE",    "RIGHT_ANKLE",
]

MIN_VISIBILITY = 0.5

# ---------------------------------------------------------------------------
# Geometry helpers (mirror of video_pose_trajectory.py)
# ---------------------------------------------------------------------------

Point = tuple[float, float, float]


def _pt(lm: dict[str, float] | None, min_vis: float = MIN_VISIBILITY) -> Point | None:
    if not lm:
        return None
    if lm.get("visibility", 0.0) < min_vis:
        return None
    return (float(lm["x"]), float(lm["y"]), float(lm.get("z", 0.0)))


def _dist2d(a: Point | None, b: Point | None) -> float | None:
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _mid(a: Point | None, b: Point | None) -> Point | None:
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0)


def _angle(a: Point | None, b: Point | None, c: Point | None) -> float | None:
    """Angle at joint b formed by a-b-c, in degrees."""
    if a is None or b is None or c is None:
        return None
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    cos_val = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cos_val = max(-1.0, min(1.0, cos_val))
    return math.degrees(math.acos(cos_val))


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or abs(den) < 1e-9:
        return None
    return num / den


def _mean(vals: list[float | None]) -> float | None:
    nums = [v for v in vals if v is not None]
    return sum(nums) / len(nums) if nums else None


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def normalise_landmarks(raw: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Accept either:
      - {name: {x, y, z?, visibility?}}   (already named, from our frontend)
      - {index: {x, y, z?, visibility?}}   (numeric indices)

    Returns a dict keyed by uppercase landmark name.
    """
    result: dict[str, dict[str, float]] = {}
    for key, val in raw.items():
        if isinstance(val, dict):
            # Try converting numeric index
            try:
                idx = int(key)
                name = MEDIAPIPE_INDEX_TO_NAME.get(idx)
                if name:
                    result[name] = val
            except (ValueError, TypeError):
                # Already a string key — normalise to upper case
                result[str(key).upper()] = val
    return result


def compute_frame_features(
    landmarks: dict[str, dict[str, float]],
    prev_landmarks: dict[str, dict[str, float]] | None = None,
) -> dict[str, float | None]:
    """Compute the same numeric features as video_pose_trajectory.compute_features."""
    pts = {name: _pt(landmarks.get(name)) for name in MEDIAPIPE_INDEX_TO_NAME.values()}

    ls = pts["LEFT_SHOULDER"]
    rs = pts["RIGHT_SHOULDER"]
    lh = pts["LEFT_HIP"]
    rh = pts["RIGHT_HIP"]
    lk = pts["LEFT_KNEE"]
    rk = pts["RIGHT_KNEE"]
    la = pts["LEFT_ANKLE"]
    ra = pts["RIGHT_ANKLE"]
    lw = pts["LEFT_WRIST"]
    rw = pts["RIGHT_WRIST"]

    shoulder_center = _mid(ls, rs)
    hip_center = _mid(lh, rh)
    ankle_center = _mid(la, ra)

    shoulder_width = _dist2d(ls, rs)
    torso_length = _dist2d(shoulder_center, hip_center)
    scale = torso_length or shoulder_width

    foot_spread = (abs(la[0] - ra[0]) if la and ra else None)
    wrist_spread = (abs(lw[0] - rw[0]) if lw and rw else None)

    lw_ht = _ratio(ls[1] - lw[1], scale) if ls and lw else None
    rw_ht = _ratio(rs[1] - rw[1], scale) if rs and rw else None
    hands_above = _mean([
        (1.0 if lw_ht is not None and lw_ht > 0.0 else 0.0) if lw_ht is not None else None,
        (1.0 if rw_ht is not None and rw_ht > 0.0 else 0.0) if rw_ht is not None else None,
    ])

    lk_angle = _angle(lh, lk, la)
    rk_angle = _angle(rh, rk, ra)
    mean_knee_angle = _mean([lk_angle, rk_angle])

    # Hip angle: shoulder-hip-knee (for high_knee detection)
    lh_angle = _angle(ls, lh, lk)
    rh_angle = _angle(rs, rh, rk)

    hip_h = (_ratio(ankle_center[1] - hip_center[1], scale)
             if ankle_center and hip_center else None)

    # Knee height relative to hip (positive = knee above hip = leg raised)
    lk_height_vs_hip = _ratio(lh[1] - lk[1], scale) if lh and lk else None
    rk_height_vs_hip = _ratio(rh[1] - rk[1], scale) if rh and rk else None

    motion_energy: float | None = None
    if prev_landmarks:
        prev_pts = {name: _pt(prev_landmarks.get(name)) for name in CORE_LANDMARKS}
        motions = []
        for name in CORE_LANDMARKS:
            cur = pts.get(name)
            prv = prev_pts.get(name)
            d = _dist2d(cur, prv)
            if d is not None and scale:
                motions.append(d / scale)
        if motions:
            motion_energy = sum(motions) / len(motions)

    all_vis = [v.get("visibility", 0.0) for v in landmarks.values() if isinstance(v, dict)]
    mean_vis = _mean(all_vis) if all_vis else None
    core_vis_count = sum(
        1 for name in CORE_LANDMARKS
        if landmarks.get(name, {}).get("visibility", 0.0) >= MIN_VISIBILITY
    )

    return {
        "mean_visibility": mean_vis,
        "core_visible_ratio": core_vis_count / len(CORE_LANDMARKS),
        "motion_energy": motion_energy,
        "shoulder_width": shoulder_width,
        "torso_length": torso_length,
        "hip_height_above_ankles_torso": hip_h,
        "foot_spread_shoulder_ratio": _ratio(foot_spread, shoulder_width),
        "wrist_spread_shoulder_ratio": _ratio(wrist_spread, shoulder_width),
        "hands_above_shoulders_ratio": hands_above,
        "mean_knee_angle": mean_knee_angle,
        "left_knee_angle": lk_angle,
        "right_knee_angle": rk_angle,
        "left_knee_height_vs_hip": lk_height_vs_hip,
        "right_knee_height_vs_hip": rk_height_vs_hip,
        "left_hip_angle": lh_angle,
        "right_hip_angle": rh_angle,
    }


# ---------------------------------------------------------------------------
# Per-buffer state machine (mirrors offline_action_state_machine.py logic)
# ---------------------------------------------------------------------------

def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _buf_values(buffer: list[dict], key: str) -> list[float]:
    return [v for row in buffer if (v := _safe_float(row.get(key))) is not None]


def _normalize(val: float | None, mn: float | None, mx: float | None, inverse: bool = False) -> float | None:
    if val is None or mn is None or mx is None:
        return None
    span = mx - mn
    if abs(span) < 1e-9:
        return None
    raw = (val - mn) / span
    if inverse:
        raw = 1.0 - raw
    return max(0.0, min(1.0, raw))


def _smooth_scores(scores: list[float | None], window: int) -> list[float | None]:
    if window <= 1:
        return scores
    half = window // 2
    out: list[float | None] = []
    for i in range(len(scores)):
        nums = [v for v in scores[max(0, i - half): i + half + 1] if v is not None]
        out.append(sum(nums) / len(nums) if nums else None)
    return out


def _stage_sequence(
    scores: list[float | None],
    low_thresh: float, high_thresh: float,
    low_stage: str, rising_stage: str, high_stage: str, falling_stage: str,
) -> list[str]:
    stages: list[str] = []
    prev: float | None = None
    for s in scores:
        if s is None:
            stages.append("unknown")
            continue
        if s <= low_thresh:
            stage = low_stage
        elif s >= high_thresh:
            stage = high_stage
        else:
            stage = rising_stage if (prev is None or s >= prev) else falling_stage
        stages.append(stage)
        prev = s
    return stages


def _count_cycles(
    stages: list[str],
    low_stage: str, high_stage: str,
    min_high_hold: int, min_low_hold: int,
) -> int:
    seen_high = False
    high_run = low_run = count = 0
    for stage in stages:
        if stage == high_stage:
            high_run += 1
            low_run = 0
            if high_run >= min_high_hold:
                seen_high = True
        elif stage == low_stage:
            low_run += 1
            high_run = 0
            if seen_high and low_run >= min_low_hold:
                count += 1
                seen_high = False
        else:
            high_run = low_run = 0
    return count


def _score_squat_buffer(buffer: list[dict], spec: dict) -> tuple[float | None, str, int]:
    """Returns (current_score, current_stage, cumulative_count_in_buffer)."""
    knee_vals = _buf_values(buffer, "mean_knee_angle")
    hip_vals = _buf_values(buffer, "hip_height_above_ankles_torso")

    knee_min = min(knee_vals) if knee_vals else None
    knee_max = max(knee_vals) if knee_vals else None
    hip_min = min(hip_vals) if hip_vals else None
    hip_max = max(hip_vals) if hip_vals else None

    scores: list[float | None] = []
    for row in buffer:
        knee_d = _normalize(_safe_float(row.get("mean_knee_angle")), knee_min, knee_max, inverse=True)
        hip_d = _normalize(_safe_float(row.get("hip_height_above_ankles_torso")), hip_min, hip_max, inverse=True)
        parts = [(0.65, knee_d), (0.35, hip_d)]
        valid = [(w, v) for w, v in parts if v is not None]
        if not valid:
            scores.append(None)
        else:
            ws = sum(w for w, _ in valid)
            scores.append(sum(w * v for w, v in valid) / ws)

    params = spec.get("frame_state_machine", {})
    window = int(params.get("smoothing_window_frames", 5))
    scores = _smooth_scores(scores, window)
    thresholds = params.get("stage_thresholds", {})
    stages = _stage_sequence(
        scores,
        float(thresholds.get("stand_max", 0.20)),
        float(thresholds.get("down_min", 0.70)),
        "stand", "downing", "down", "rising",
    )
    count = _count_cycles(
        stages, "stand", "down",
        int(params.get("min_down_hold_frames", 2)),
        int(params.get("min_return_to_stand_frames", 2)),
    )
    current_score = scores[-1] if scores else None
    current_stage = stages[-1] if stages else "unknown"
    return current_score, current_stage, count


def _score_jumping_jack_buffer(buffer: list[dict], spec: dict) -> tuple[float | None, str, int]:
    foot_vals = _buf_values(buffer, "foot_spread_shoulder_ratio")
    wrist_vals = _buf_values(buffer, "wrist_spread_shoulder_ratio")

    foot_min = min(foot_vals) if foot_vals else None
    foot_max = max(foot_vals) if foot_vals else None
    wrist_min = min(wrist_vals) if wrist_vals else None
    wrist_max = max(wrist_vals) if wrist_vals else None

    scores: list[float | None] = []
    for row in buffer:
        foot = _normalize(_safe_float(row.get("foot_spread_shoulder_ratio")), foot_min, foot_max)
        wrist = _normalize(_safe_float(row.get("wrist_spread_shoulder_ratio")), wrist_min, wrist_max)
        hands = _safe_float(row.get("hands_above_shoulders_ratio"))
        parts = [(0.45, foot), (0.35, wrist), (0.20, max(0.0, min(1.0, hands)) if hands is not None else None)]
        valid = [(w, v) for w, v in parts if v is not None]
        if not valid:
            scores.append(None)
        else:
            ws = sum(w for w, _ in valid)
            scores.append(sum(w * v for w, v in valid) / ws)

    params = spec.get("frame_state_machine", {})
    window = int(params.get("smoothing_window_frames", 5))
    scores = _smooth_scores(scores, window)
    thresholds = params.get("stage_thresholds", {})
    stages = _stage_sequence(
        scores,
        float(thresholds.get("closed_max", 0.25)),
        float(thresholds.get("open_min", 0.65)),
        "closed", "opening", "open", "closing",
    )
    count = _count_cycles(
        stages, "closed", "open",
        int(params.get("min_open_hold_frames", 1)),
        int(params.get("min_return_to_closed_frames", 1)),
    )
    current_score = scores[-1] if scores else None
    current_stage = stages[-1] if stages else "unknown"
    return current_score, current_stage, count


def _score_clap_under_knee_buffer(
    buffer: list[dict], spec: dict,
) -> tuple[float | None, str, int, int]:
    """Per-side cyclic state machine for clap_under_knee.

    Returns (current_score, current_stage, left_count_in_buffer, right_count_in_buffer).
    Caller accumulates deltas against previous buffer counts.
    """
    cfg      = spec.get("frame_state_machine", {})
    side_cfg = cfg.get("left_leg", {})  # left/right share same thresholds

    lk_ht_min    = float(side_cfg.get("conditions", {}).get("knee_height_min", 0.05))
    lk_angle_max = float(side_cfg.get("conditions", {}).get("knee_angle_max", 70.0))
    ws_max       = float(side_cfg.get("conditions", {}).get("wrist_spread_max", 1.0))
    min_peak     = int(side_cfg.get("min_peak_hold_frames", 2))
    min_return   = int(side_cfg.get("min_return_frames", 3))
    window       = int(cfg.get("smoothing_window_frames", 3))

    lk_ht_vals  = _smooth_scores([_safe_float(r.get("left_knee_height_vs_hip"))  for r in buffer], window)
    rk_ht_vals  = _smooth_scores([_safe_float(r.get("right_knee_height_vs_hip")) for r in buffer], window)
    lk_ang_vals = _smooth_scores([_safe_float(r.get("left_knee_angle"))  for r in buffer], window)
    rk_ang_vals = _smooth_scores([_safe_float(r.get("right_knee_angle")) for r in buffer], window)
    ws_vals     = _smooth_scores([_safe_float(r.get("wrist_spread_shoulder_ratio")) for r in buffer], window)

    def count_side(ht_vals: list[float | None], ang_vals: list[float | None]) -> int:
        seen_peak = False
        in_peak = return_run = cnt = 0
        for ht, ang, ws in zip(ht_vals, ang_vals, ws_vals):
            is_peak = (
                ht  is not None and ht  >= lk_ht_min and
                ang is not None and ang <= lk_angle_max and
                ws  is not None and ws  <= ws_max
            )
            is_low = ht is not None and ht < lk_ht_min * 0.3
            if is_peak:
                in_peak += 1
                return_run = 0
                if in_peak >= min_peak:
                    seen_peak = True
            elif is_low:
                return_run += 1
                in_peak = 0
                if seen_peak and return_run >= min_return:
                    cnt += 1
                    seen_peak = False
            else:
                in_peak = return_run = 0
        return cnt

    left_cnt  = count_side(lk_ht_vals, lk_ang_vals)
    right_cnt = count_side(rk_ht_vals, rk_ang_vals)

    # Stage and score from last frame
    lk_ht  = lk_ht_vals[-1]  if lk_ht_vals  else None
    rk_ht  = rk_ht_vals[-1]  if rk_ht_vals  else None
    lk_ang = lk_ang_vals[-1] if lk_ang_vals else None
    rk_ang = rk_ang_vals[-1] if rk_ang_vals else None
    ws     = ws_vals[-1]     if ws_vals     else None

    left_active  = lk_ht is not None and lk_ht  >= lk_ht_min and lk_ang is not None and lk_ang <= lk_angle_max and ws is not None and ws <= ws_max
    right_active = rk_ht is not None and rk_ht  >= lk_ht_min and rk_ang is not None and rk_ang <= lk_angle_max and ws is not None and ws <= ws_max

    if left_active:
        stage = "left_clap"
        score = max(0.0, 1.0 - (lk_ang / lk_angle_max)) if lk_ang is not None else None
    elif right_active:
        stage = "right_clap"
        score = max(0.0, 1.0 - (rk_ang / lk_angle_max)) if rk_ang is not None else None
    elif lk_ht is not None and lk_ht >= lk_ht_min * 0.5:
        stage = "left_raising"
        score = None
    elif rk_ht is not None and rk_ht >= lk_ht_min * 0.5:
        stage = "right_raising"
        score = None
    else:
        stage = "stand"
        score = None

    return score, stage, left_cnt, right_cnt


def _score_high_knee_buffer(
    buffer: list[dict], spec: dict,
) -> tuple[float | None, str, int, int]:
    """Per-side cyclic state machine for high_knee.

    Uses hip_angle as primary signal. Returns
    (current_score, current_stage, left_count_in_buffer, right_count_in_buffer).
    """
    cfg      = spec.get("frame_state_machine", {})
    left_cfg  = cfg.get("left_leg",  {})

    raise_thresh  = float(left_cfg.get("raise_threshold",  150.0))
    stand_thresh  = float(left_cfg.get("stand_threshold",  155.0))
    min_raise     = int(left_cfg.get("min_raise_frames",     2))
    min_return    = int(left_cfg.get("min_return_frames",    3))
    window        = int(cfg.get("smoothing_window_frames",   3))

    lh_vals = _smooth_scores([_safe_float(r.get("left_hip_angle"))  for r in buffer], window)
    rh_vals = _smooth_scores([_safe_float(r.get("right_hip_angle")) for r in buffer], window)
    lk_vals = _smooth_scores([_safe_float(r.get("left_knee_angle"))  for r in buffer], window)
    rk_vals = _smooth_scores([_safe_float(r.get("right_knee_angle")) for r in buffer], window)

    def count_side(ha_vals: list[float | None], opp_ka_vals: list[float | None]) -> tuple[int, float | None]:
        """Returns (count, peak_quality_score_of_last_event)."""
        seen_raise = False
        in_raise = return_run = cnt = 0
        last_peak_score: float | None = None
        peak_ha_in_event: float | None = None

        for ha, opp_ka in zip(ha_vals, opp_ka_vals):
            is_raised = ha is not None and ha < raise_thresh
            is_stand  = ha is not None and ha > stand_thresh

            if is_raised:
                in_raise += 1
                return_run = 0
                if peak_ha_in_event is None or ha < peak_ha_in_event:
                    peak_ha_in_event = ha
                if in_raise >= min_raise:
                    seen_raise = True
            elif is_stand:
                return_run += 1
                in_raise = 0
                if seen_raise and return_run >= min_return:
                    cnt += 1
                    seen_raise = False
                    # Quality from hip angle at peak
                    if peak_ha_in_event is not None:
                        lvls = spec.get("quality_scoring", {}).get("levels", {})
                        exc = float(lvls.get("excellent", {}).get("hip_angle_max", 80.0))
                        gd  = float(lvls.get("good",      {}).get("hip_angle_max", 120.0))
                        fa  = float(lvls.get("fair",      {}).get("hip_angle_max", 150.0))
                        if peak_ha_in_event <= exc:
                            last_peak_score = 1.0
                        elif peak_ha_in_event <= gd:
                            last_peak_score = 0.70
                        elif peak_ha_in_event <= fa:
                            last_peak_score = 0.40
                        else:
                            last_peak_score = 0.0
                    peak_ha_in_event = None
            else:
                in_raise = return_run = 0

        return cnt, last_peak_score

    left_cnt,  left_score  = count_side(lh_vals, rk_vals)
    right_cnt, right_score = count_side(rh_vals, lk_vals)

    # Stage and score from last frame
    lh_last = lh_vals[-1] if lh_vals else None
    rh_last = rh_vals[-1] if rh_vals else None

    if lh_last is not None and lh_last < raise_thresh:
        stage = "left_raise"
        # Realtime quality score (how good is current raise)
        lvls = spec.get("quality_scoring", {}).get("levels", {})
        exc = float(lvls.get("excellent", {}).get("hip_angle_max", 80.0))
        gd  = float(lvls.get("good",      {}).get("hip_angle_max", 120.0))
        fa  = float(lvls.get("fair",      {}).get("hip_angle_max", 150.0))
        if lh_last <= exc:   score: float | None = 1.0
        elif lh_last <= gd:  score = 0.70
        elif lh_last <= fa:  score = 0.40
        else:                score = 0.0
    elif rh_last is not None and rh_last < raise_thresh:
        stage = "right_raise"
        lvls = spec.get("quality_scoring", {}).get("levels", {})
        exc = float(lvls.get("excellent", {}).get("hip_angle_max", 80.0))
        gd  = float(lvls.get("good",      {}).get("hip_angle_max", 120.0))
        fa  = float(lvls.get("fair",      {}).get("hip_angle_max", 150.0))
        if rh_last <= exc:   score = 1.0
        elif rh_last <= gd:  score = 0.70
        elif rh_last <= fa:  score = 0.40
        else:                score = 0.0
    else:
        stage = "stand"
        score = left_score or right_score  # last counted event's quality

    return score, stage, left_cnt, right_cnt




class RealtimeActionEngine:
    """Stateful per-connection action recognition engine.

    Usage:
        engine = RealtimeActionEngine(spec_dir=Path("backend/action_engine/action_specs"))
        result = engine.push_frame(landmarks_dict, timestamp_ms=1234.5)
        # result["actions"] is a list of action results
    """

    def __init__(
        self,
        spec_dir: Path,
        view: str = "front",
        buffer_sec: float = 3.0,
        fps_estimate: float = 30.0,
        event_log_dir: Path | None = None,
    ) -> None:
        self.view = view
        self.buffer_max = max(10, int(buffer_sec * fps_estimate))
        self._buffer: deque[dict] = deque(maxlen=self.buffer_max)
        self._prev_landmarks: dict[str, dict[str, float]] | None = None
        self._cumulative_counts: dict[str, int] = {}
        self._prev_buffer_counts: dict[str, int] = {}
        self._lunge_hold_sec: dict[str, float] = {"left": 0.0, "right": 0.0}
        self._lunge_state: str = "idle"
        self._lunge_state_frames: int = 0
        self._lunge_last_ts: float | None = None
        self._lunge_candidate: str = "idle"
        self._lunge_candidate_frames: int = 0
        self._clap_left_count: int = 0
        self._clap_right_count: int = 0
        self._clap_prev_buf_left: int = 0
        self._clap_prev_buf_right: int = 0
        self._hk_left_count: int = 0
        self._hk_right_count: int = 0
        self._hk_prev_buf_left: int = 0
        self._hk_prev_buf_right: int = 0
        self._event_log_dir: Path | None = event_log_dir if event_log_dir is not None else DEFAULT_EVENT_LOG_DIR
        # Calorie tracking
        self._weight_kg: float = 70.0
        self._session_kcal: dict[str, float] = {}
        self._motion_kcal: float = 0.0
        # Per-rep timing (for rep_duration_sec)
        self._last_rep_ts: dict[str, float] = {}
        # Pending count events (flushed each frame)
        self._pending_events: list[dict] = []
        # FPS estimation (sliding window over last 30 frame intervals)
        self._fps_ts_deque: deque[float] = deque(maxlen=30)
        self._prev_frame_ts: float | None = None
        # Previous hip height for Δhip computation
        self._prev_hip_height: float | None = None
        self.specs = self._load_specs(spec_dir)

    def _log_count_event(self, action: str, side: str | None, count: int) -> None:
        """Write a snapshot of the current buffer to a JSONL file when a count fires."""
        if self._event_log_dir is None:
            return
        try:
            self._event_log_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{action}_{ts}.jsonl"
            path = self._event_log_dir / filename
            buf = list(self._buffer)
            event = {
                "event": "count",
                "action": action,
                "side": side,
                "cumulative_count": count,
                "timestamp": time.time(),
                "buffer_len": len(buf),
                "buffer": [
                    {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in row.items()}
                    for row in buf
                ],
            }
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _make_count_event(
        self,
        action: str,
        side: str | None,
        timestamp_ms: float,
    ) -> dict:
        """Build a count event dict with rep metadata and calorie estimate."""
        buf = list(self._buffer)

        # rep_duration_sec
        last_ts = self._last_rep_ts.get(action)
        if last_ts is not None:
            rep_duration_sec = round(max(0.3, min((timestamp_ms - last_ts) / 1000.0, 10.0)), 2)
        else:
            rep_duration_sec = None
        self._last_rep_ts[action] = timestamp_ms

        # amplitude
        amplitude: float | None = None
        if action == "squat":
            vals = _buf_values(buf, "hip_height_above_ankles_torso")
            amplitude = round(max(vals) - min(vals), 3) if len(vals) >= 2 else None
        elif action == "jumping_jack":
            vals = _buf_values(buf, "foot_spread_shoulder_ratio")
            amplitude = round(max(vals) - min(vals), 3) if len(vals) >= 2 else None
        elif action in ("high_knee", "clap_under_knee"):
            key = "left_knee_height_vs_hip" if side == "left" else "right_knee_height_vs_hip"
            vals = _buf_values(buf, key)
            amplitude = round(max(vals), 3) if vals else None

        # quality (squat uses amplitude gate; others default to standard)
        quality = "standard"
        if action == "squat" and amplitude is not None:
            quality = "standard" if amplitude >= 0.30 else "shallow"

        # calorie
        kcal = 0.0
        if rep_duration_sec is not None:
            kcal = estimate_rep_kcal(
                action=action,
                rep_duration_sec=rep_duration_sec,
                amplitude=amplitude,
                quality=quality,
                weight_kg=self._weight_kg,
            )
            self._session_kcal[action] = self._session_kcal.get(action, 0.0) + kcal

        return {
            "action":           action,
            "side":             side,
            "rep_duration_sec": rep_duration_sec,
            "amplitude":        amplitude,
            "quality":          quality,
            "kcal":             round(kcal, 4),
            "timestamp_ms":     round(timestamp_ms, 1),
        }

    @staticmethod
    def _load_specs(spec_dir: Path) -> list[dict]:
        specs = []
        for p in sorted(spec_dir.glob("*.json")):
            try:
                spec = json.loads(p.read_text(encoding="utf-8"))
                if spec.get("action_name") in {"squat", "jumping_jack", "lunge", "clap_under_knee", "high_knee"}:
                    specs.append(spec)
            except Exception:
                pass
        return specs

    def set_user_config(self, weight_kg: float | None = None, height_cm: float | None = None) -> None:
        if weight_kg is not None:
            self._weight_kg = float(weight_kg)

    def reset(self) -> None:
        self._buffer.clear()
        self._prev_landmarks = None
        self._cumulative_counts = {}
        self._prev_buffer_counts = {}
        self._lunge_hold_sec: dict[str, float] = {"left": 0.0, "right": 0.0}
        self._lunge_state: str = "idle"
        self._lunge_state_frames: int = 0
        self._lunge_last_ts: float | None = None
        self._clap_left_count = 0
        self._clap_right_count = 0
        self._clap_prev_buf_left = 0
        self._clap_prev_buf_right = 0
        self._hk_left_count = 0
        self._hk_right_count = 0
        self._hk_prev_buf_left = 0
        self._hk_prev_buf_right = 0
        self._session_kcal = {}
        self._motion_kcal = 0.0
        self._last_rep_ts = {}
        self._pending_events = []
        self._fps_ts_deque: deque[float] = deque(maxlen=30)
        self._prev_frame_ts = None
        self._prev_hip_height = None

    def push_frame(
        self,
        raw_landmarks: dict[str, Any],
        timestamp_ms: float | None = None,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()

        landmarks = normalise_landmarks(raw_landmarks)
        features = compute_frame_features(landmarks, self._prev_landmarks)
        self._prev_landmarks = landmarks

        features["timestamp_ms"] = timestamp_ms or (time.time() * 1000)
        self._buffer.append(features)

        actions = []
        buf = list(self._buffer)

        for spec in self.specs:
            action = spec["action_name"]

            # --- lunge: hold-duration mode ---
            if action == "lunge":
                lunge_result = self._run_lunge(
                    landmarks, features, spec,
                    features.get("timestamp_ms") or time.time() * 1000,
                )
                actions.append(lunge_result)
                continue

            # --- clap_under_knee: per-side cyclic count ---
            if action == "clap_under_knee":
                score, stage, buf_left, buf_right = _score_clap_under_knee_buffer(buf, spec)
                if buf_left > self._clap_prev_buf_left:
                    self._clap_left_count += buf_left - self._clap_prev_buf_left
                    ev = self._make_count_event("clap_under_knee", "left", score)
                    self._pending_events.append(ev)
                    self._session_kcal["clap_under_knee"] = self._session_kcal.get("clap_under_knee", 0.0) + ev["kcal"]
                    self._log_count_event("clap_under_knee", "left", self._clap_left_count + self._clap_right_count)
                elif self._clap_prev_buf_left - buf_left > 2:
                    self._clap_prev_buf_left = buf_left
                self._clap_prev_buf_left = buf_left
                if buf_right > self._clap_prev_buf_right:
                    self._clap_right_count += buf_right - self._clap_prev_buf_right
                    ev = self._make_count_event("clap_under_knee", "right", score)
                    self._pending_events.append(ev)
                    self._session_kcal["clap_under_knee"] = self._session_kcal.get("clap_under_knee", 0.0) + ev["kcal"]
                    self._log_count_event("clap_under_knee", "right", self._clap_left_count + self._clap_right_count)
                elif self._clap_prev_buf_right - buf_right > 2:
                    self._clap_prev_buf_right = buf_right
                self._clap_prev_buf_right = buf_right
                actions.append({
                    "name":        "clap_under_knee",
                    "count":       self._clap_left_count + self._clap_right_count,
                    "count_left":  self._clap_left_count,
                    "count_right": self._clap_right_count,
                    "stage":       stage,
                    "score":       round(score, 3) if score is not None else None,
                })
                continue

            # --- high_knee: per-side cyclic count ---
            if action == "high_knee":
                score, stage, buf_left, buf_right = _score_high_knee_buffer(buf, spec)
                if buf_left > self._hk_prev_buf_left:
                    self._hk_left_count += buf_left - self._hk_prev_buf_left
                    ev = self._make_count_event("high_knee", "left", score)
                    self._pending_events.append(ev)
                    self._session_kcal["high_knee"] = self._session_kcal.get("high_knee", 0.0) + ev["kcal"]
                    self._log_count_event("high_knee", "left", self._hk_left_count + self._hk_right_count)
                elif self._hk_prev_buf_left - buf_left > 2:
                    self._hk_prev_buf_left = buf_left
                self._hk_prev_buf_left = buf_left
                if buf_right > self._hk_prev_buf_right:
                    self._hk_right_count += buf_right - self._hk_prev_buf_right
                    ev = self._make_count_event("high_knee", "right", score)
                    self._pending_events.append(ev)
                    self._session_kcal["high_knee"] = self._session_kcal.get("high_knee", 0.0) + ev["kcal"]
                    self._log_count_event("high_knee", "right", self._hk_left_count + self._hk_right_count)
                elif self._hk_prev_buf_right - buf_right > 2:
                    self._hk_prev_buf_right = buf_right
                self._hk_prev_buf_right = buf_right
                actions.append({
                    "name":        "high_knee",
                    "count":       self._hk_left_count + self._hk_right_count,
                    "count_left":  self._hk_left_count,
                    "count_right": self._hk_right_count,
                    "stage":       stage,
                    "score":       round(score, 3) if score is not None else None,
                })
                continue

            # --- cyclic actions (squat, jumping_jack): count mode ---
            if action not in self._cumulative_counts:
                self._cumulative_counts[action] = 0
                self._prev_buffer_counts[action] = 0

            score, stage, buf_count = self._run_on_buffer(buf, spec)

            prev = self._prev_buffer_counts[action]
            if buf_count > prev:
                delta = buf_count - prev
                self._cumulative_counts[action] += delta
                ts = features.get("timestamp_ms") or time.time() * 1000
                for _ in range(delta):
                    ev = self._make_count_event(action, None, ts)
                    self._pending_events.append(ev)
                self._log_count_event(action, None, self._cumulative_counts[action])
            elif prev - buf_count > 2:
                self._prev_buffer_counts[action] = buf_count
                prev = buf_count
            self._prev_buffer_counts[action] = buf_count

            actions.append({
                "name": action,
                "count": self._cumulative_counts[action],
                "stage": stage,
                "score": round(score, 3) if score is not None else None,
            })

        # --- motion_energy continuous calorie integration ---
        ts_now = features.get("timestamp_ms") or (time.time() * 1000)
        core_vis = features.get("core_visible_ratio") or 0.0

        if core_vis >= 0.7:
            # FPS estimation from frame timestamps
            if self._prev_frame_ts is not None:
                dt_ms = ts_now - self._prev_frame_ts
                if 10.0 < dt_ms < 500.0:  # only sane intervals (2–100 fps)
                    self._fps_ts_deque.append(dt_ms)
            self._prev_frame_ts = ts_now
            fps = (1000.0 / (sum(self._fps_ts_deque) / len(self._fps_ts_deque))
                   if self._fps_ts_deque else 30.0)

            motion_e = features.get("motion_energy") or 0.0
            hip_h = features.get("hip_height_above_ankles_torso")
            delta_hip = (abs(hip_h - self._prev_hip_height)
                         if hip_h is not None and self._prev_hip_height is not None else 0.0)
            self._prev_hip_height = hip_h

            # Combined intensity: hip displacement (w=3) + motion_energy (w=1)
            intensity = 3.0 * delta_hip + 1.0 * motion_e

            if intensity < 0.005:
                met = 1.5
            elif intensity < 0.02:
                met = 2.5
            elif intensity < 0.05:
                met = 5.0
            elif intensity < 0.10:
                met = 7.0
            else:
                met = 9.0

            self._motion_kcal += met * 3.5 * self._weight_kg / 200.0 / fps / 60.0
        else:
            self._prev_frame_ts = ts_now
            self._prev_hip_height = None  # reset on pose loss

        # Flush pending events
        new_events = list(self._pending_events)
        self._pending_events.clear()

        event_kcal = sum(self._session_kcal.values())
        calorie_summary = {
            "motion_kcal": round(self._motion_kcal, 3),
            "event_kcal":  round(event_kcal, 3),
            "total_kcal":  round(self._motion_kcal, 3),
            "by_action":   {k: round(v, 3) for k, v in self._session_kcal.items()},
        }

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "actions": actions,
            "count_events": new_events,
            "calorie_summary": calorie_summary,
            "features": {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in features.items()
                if k != "timestamp_ms"
            },
            "latency_ms": latency_ms,
            "buffer_size": len(self._buffer),
        }

    def _run_lunge(
        self,
        landmarks: dict[str, dict[str, float]],
        features: dict[str, Any],
        spec: dict,
        timestamp_ms: float,
    ) -> dict[str, Any]:
        """Hold-duration state machine for lunge.

        Returns a result dict with:
            name, stage, score (front knee quality 0-1),
            hold_sec_left, hold_sec_right (cumulative valid hold seconds),
            count (always 0 — lunges don't count reps)
        """
        cfg  = spec.get("hold_state_machine", {})
        sdcfg = spec.get("side_detection", {})
        qcfg  = spec.get("quality_scoring", {})

        z_left_max  = float(sdcfg.get("left_leg_forward_threshold",  -0.30))
        z_right_min = float(sdcfg.get("right_leg_forward_threshold",  0.30))
        knee_valid  = float(qcfg.get("levels", {}).get("invalid", {}).get("min_angle", 150.0))
        enter_n = int(cfg.get("enter_hold_frames", 3))

        # --- z diff from raw landmarks (not from features.csv which lacks z) ---
        la = landmarks.get("LEFT_ANKLE",  {})
        ra = landmarks.get("RIGHT_ANKLE", {})
        la_z = _safe_float(la.get("z"))
        ra_z = _safe_float(ra.get("z"))
        z_diff = (la_z - ra_z) if la_z is not None and ra_z is not None else None

        lk = _safe_float(features.get("left_knee_angle"))
        rk = _safe_float(features.get("right_knee_angle"))

        # Determine raw candidate state
        if z_diff is not None and z_diff < z_left_max and rk is not None and rk < knee_valid:
            candidate = "left_hold"
            front_knee = rk
        elif z_diff is not None and z_diff > z_right_min and lk is not None and lk < knee_valid:
            candidate = "right_hold"
            front_knee = lk
        else:
            candidate = "idle"
            front_knee = None

        # Hysteresis: require enter_n consecutive matching frames
        if candidate == self._lunge_state:
            self._lunge_state_frames += 1
            self._lunge_candidate_frames = 0
        else:
            if self._lunge_candidate == candidate:
                self._lunge_candidate_frames += 1
                if self._lunge_candidate_frames >= enter_n:
                    self._lunge_state = candidate
                    self._lunge_state_frames = enter_n
                    self._lunge_candidate_frames = 0
            else:
                self._lunge_candidate = candidate
                self._lunge_candidate_frames = 1

        # Accumulate hold time and lunge calories
        if self._lunge_last_ts is not None:
            dt_sec = max(0.0, (timestamp_ms - self._lunge_last_ts) / 1000.0)
            dt_sec = min(dt_sec, 0.2)  # cap at 200ms to ignore large gaps
            if self._lunge_state in ("left_hold", "right_hold"):
                self._lunge_hold_sec[self._lunge_state.split("_")[0]] += dt_sec
                inc = estimate_lunge_kcal(dt_sec, quality_score or 0.0, self._weight_kg)
                self._session_kcal["lunge"] = self._session_kcal.get("lunge", 0.0) + inc
        self._lunge_last_ts = timestamp_ms

        # Quality score for current frame (0-1, based on front knee angle)
        quality_score: float | None = None
        if front_knee is not None:
            levels = qcfg.get("levels", {})
            if front_knee <= float(levels.get("excellent", {}).get("max_angle", 120)):
                quality_score = 1.0
            elif front_knee <= float(levels.get("good", {}).get("max_angle", 135)):
                quality_score = 0.75
            elif front_knee <= float(levels.get("fair", {}).get("max_angle", 150)):
                quality_score = 0.50
            else:
                quality_score = 0.0

        return {
            "name":            "lunge",
            "count":           0,
            "stage":           self._lunge_state,
            "score":           round(quality_score, 3) if quality_score is not None else None,
            "hold_sec_left":   round(self._lunge_hold_sec["left"],  1),
            "hold_sec_right":  round(self._lunge_hold_sec["right"], 1),
        }

    def _run_on_buffer(
        self, buf: list[dict], spec: dict
    ) -> tuple[float | None, str, int]:
        action = spec["action_name"]
        if action == "squat":
            return _score_squat_buffer(buf, spec)
        if action == "jumping_jack":
            return _score_jumping_jack_buffer(buf, spec)
        return None, "unsupported", 0
