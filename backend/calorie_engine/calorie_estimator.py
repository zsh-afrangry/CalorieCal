"""Calorie estimator for realtime action recognition.

Formula (per rep):
    kcal = MET_eff × 3.5 × weight_kg × (rep_duration_sec / 60) / 200
    MET_eff = met_base × speed_factor × depth_factor

For hold-duration actions (lunge):
    kcal = MET_eff × 3.5 × weight_kg × (hold_sec / 60) / 200
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_PARAMS_PATH = Path(__file__).resolve().parent / "met_params.json"


def _load_params() -> dict[str, Any]:
    return json.loads(_PARAMS_PATH.read_text(encoding="utf-8"))


_PARAMS: dict[str, Any] | None = None


def get_params() -> dict[str, Any]:
    global _PARAMS
    if _PARAMS is None:
        _PARAMS = _load_params()
    return _PARAMS


def estimate_rep_kcal(
    action: str,
    rep_duration_sec: float,
    amplitude: float | None,
    quality: str,
    weight_kg: float,
) -> float:
    """Estimate kcal for one rep of a cyclic action.

    Returns 0.0 on any invalid input rather than raising.
    """
    if rep_duration_sec <= 0 or weight_kg <= 0:
        return 0.0

    # Cap duration to avoid polluting from missed frames or pauses
    rep_duration_sec = min(rep_duration_sec, 10.0)
    rep_duration_sec = max(rep_duration_sec, 0.3)

    params = get_params()
    action_params = params.get(action)
    if action_params is None:
        # Fallback: moderate MET
        met_base = 5.0
        speed_factor = 1.0
        depth_factor = 1.0
    else:
        met_base = float(action_params.get("met_base", 5.0))

        # Speed factor
        sf_cfg = action_params.get("speed_factor")
        if sf_cfg:
            fast_t = float(sf_cfg.get("fast_threshold_sec", 2.0))
            slow_t = float(sf_cfg.get("slow_threshold_sec", 4.0))
            fast_b = float(sf_cfg.get("fast_bonus", 1.2))
            slow_p = float(sf_cfg.get("slow_penalty", 0.9))
            if rep_duration_sec < fast_t:
                speed_factor = fast_b
            elif rep_duration_sec > slow_t:
                speed_factor = slow_p
            else:
                speed_factor = 1.0
        else:
            speed_factor = 1.0

        # Depth / quality factor
        df_cfg = action_params.get("depth_factor", {})
        if quality in df_cfg:
            depth_factor = float(df_cfg[quality])
        elif quality == "unknown":
            depth_factor = float(df_cfg.get("standard", 1.0))
        else:
            depth_factor = 1.0

    met_eff = met_base * speed_factor * depth_factor
    kcal = met_eff * 3.5 * weight_kg * (rep_duration_sec / 60.0) / 200.0
    return round(kcal, 5)


def estimate_hold_kcal(
    action: str,
    hold_sec: float,
    quality: str,
    weight_kg: float,
) -> float:
    """Estimate kcal for a hold-duration action (e.g. lunge).

    Called incrementally (e.g. every second delta).
    """
    if hold_sec <= 0 or weight_kg <= 0:
        return 0.0

    params = get_params()
    action_params = params.get(action, {})
    met_base = float(action_params.get("met_base", 4.0))
    df_cfg = action_params.get("depth_factor", {})
    depth_factor = float(df_cfg.get(quality, df_cfg.get("standard", 1.0)))

    met_eff = met_base * depth_factor
    kcal = met_eff * 3.5 * weight_kg * (hold_sec / 60.0) / 200.0
    return round(kcal, 5)
