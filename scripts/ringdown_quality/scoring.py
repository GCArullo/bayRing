from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import waveform_io
from .alignment import (
    apply_z_rotation,
    covers_window,
    global_resolution_phase_offset,
    interpolate_complex,
    shifted_time,
    window_grid,
)
from .config import Config
from .metrics import (
    complex_l2_norm,
    extrapolation_error,
    floor_penalty,
    metrics_are_finite,
    mode_quality_score,
    resolution_error,
    symmetry_error,
)


PER_MODE_COLUMNS = [
    "sxs_id",
    "ell",
    "m",
    "lev_high",
    "lev_low",
    "reference_extrapolation_order",
    "comparison_extrapolation_orders",
    "window_start_M",
    "window_end_M",
    "n_samples",
    "mode_norm",
    "delta_res",
    "delta_ext",
    "delta_floor",
    "delta_sym",
    "E_mode",
    "Q_mode",
    "P_mode",
    "valid_mode",
    "flags",
    "rejection_reason",
]

SIMULATION_SCORE_COLUMNS = [
    "sxs_id",
    "S_common",
    "aggregate_method",
    "P_min",
    "P_median",
    "P_mean",
    "E_median",
    "Q_median",
    "lev_high",
    "lev_low",
    "reference_mass_ratio",
    "reference_chi_eff",
    "reference_chi1_perp",
    "reference_chi2_perp",
    "reference_eccentricity",
    "number_of_orbits",
    "valid_all_modes",
    "flags",
]


def _orders_string(orders: list[str]) -> str:
    return " ".join(str(order) for order in orders)


def _mode_base_row(config: Config, sxs_id: str, mode: tuple[int, int], lev_high: int | None, lev_low: int | None) -> dict[str, Any]:
    ell, m = mode
    window = config.waveform.ringdown_window_M
    return {
        "sxs_id": sxs_id,
        "ell": ell,
        "m": m,
        "lev_high": lev_high,
        "lev_low": lev_low,
        "reference_extrapolation_order": config.waveform.reference_extrapolation_order,
        "comparison_extrapolation_orders": _orders_string(config.waveform.comparison_extrapolation_orders),
        "window_start_M": float(window[0]),
        "window_end_M": float(window[1]),
        "n_samples": 0,
        "mode_norm": np.nan,
        "delta_res": np.nan,
        "delta_ext": np.nan,
        "delta_floor": np.nan,
        "delta_sym": np.nan,
        "E_mode": np.nan,
        "Q_mode": np.nan,
        "P_mode": np.nan,
        "valid_mode": False,
        "flags": "",
        "rejection_reason": "",
    }


def _invalid_mode_row(
    config: Config,
    sxs_id: str,
    mode: tuple[int, int],
    lev_high: int | None,
    lev_low: int | None,
    reason: str,
    flags: list[str] | None = None,
) -> dict[str, Any]:
    row = _mode_base_row(config, sxs_id, mode, lev_high, lev_low)
    row["flags"] = ";".join(flags or [reason])
    row["rejection_reason"] = reason
    return row


def _candidate_sxs_id(candidate: pd.Series) -> str:
    if "sxs_id" in candidate.index:
        return str(candidate["sxs_id"])
    return str(candidate.name)


def _finite_int(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def _lev_pair(candidate: pd.Series, config: Config) -> tuple[int | None, int | None]:
    lev_low = _finite_int(candidate.get("lev_low"))
    lev_high = _finite_int(candidate.get("lev_high"))
    if lev_low is not None and lev_high is not None:
        return lev_low, lev_high
    levs = waveform_io.discover_available_levs(_candidate_sxs_id(candidate), config=config)
    if len(levs) < 2:
        return None, None
    return int(levs[-2]), int(levs[-1])


def _load_required_waveforms(
    sxs_id: str,
    lev_low: int,
    lev_high: int,
    config: Config,
) -> tuple[waveform_io.WaveformData, waveform_io.WaveformData, dict[str, waveform_io.WaveformData]]:
    ref_order = config.waveform.reference_extrapolation_order
    if waveform_io.extrapolation_order_available(sxs_id, lev_high, ref_order, config) is False:
        raise FileNotFoundError(f"{sxs_id} Lev{lev_high} has no reference strain file for {ref_order}.")
    if waveform_io.extrapolation_order_available(sxs_id, lev_low, ref_order, config) is False:
        raise FileNotFoundError(f"{sxs_id} Lev{lev_low} has no reference strain file for {ref_order}.")
    missing_orders = [
        str(order)
        for order in config.waveform.comparison_extrapolation_orders
        if waveform_io.extrapolation_order_available(sxs_id, lev_high, str(order), config) is False
    ]
    if missing_orders and config.filters.require_extrapolation_comparison:
        raise FileNotFoundError(f"missing high-resolution strain files for extrapolation orders {missing_orders}")

    high_ref = waveform_io.load_waveform(sxs_id, lev_high, ref_order, config)
    low_ref = waveform_io.load_waveform(sxs_id, lev_low, ref_order, config)
    comparison_waveforms = {}
    for order in config.waveform.comparison_extrapolation_orders:
        try:
            comparison_waveforms[str(order)] = waveform_io.load_waveform(sxs_id, lev_high, str(order), config)
        except Exception:
            if config.filters.require_extrapolation_comparison:
                raise
    return high_ref, low_ref, comparison_waveforms


def _large_metric_flags(delta_res: float, delta_ext: float, delta_floor: float, delta_sym: float) -> list[str]:
    flags = []
    if delta_res > 1.0:
        flags.append("large_resolution_error")
    if delta_ext > 1.0:
        flags.append("large_extrapolation_error")
    if delta_floor > 1.0:
        flags.append("large_floor_penalty")
    if delta_sym > 1.0:
        flags.append("large_symmetry_error")
    return flags


def _mode_rows_for_global_failure(
    config: Config,
    sxs_id: str,
    lev_high: int | None,
    lev_low: int | None,
    reason: str,
    flags: list[str],
) -> list[dict[str, Any]]:
    return [
        _invalid_mode_row(config, sxs_id, mode, lev_high, lev_low, reason=reason, flags=flags)
        for mode in config.modes.target_modes
    ]


def score_candidate(candidate: pd.Series, config: Config) -> list[dict[str, Any]]:
    sxs_id = _candidate_sxs_id(candidate)
    lev_low, lev_high = _lev_pair(candidate, config)
    if lev_low is None or lev_high is None:
        return _mode_rows_for_global_failure(
            config, sxs_id, lev_high, lev_low, "fewer than two Levs are available", ["insufficient_levs"]
        )

    try:
        high_ref, low_ref, comparison_waveforms = _load_required_waveforms(sxs_id, lev_low, lev_high, config)
    except Exception as exc:
        return _mode_rows_for_global_failure(
            config,
            sxs_id,
            lev_high,
            lev_low,
            f"failed to load required waveform: {exc}",
            ["waveform_load_failed"],
        )

    try:
        t_high = shifted_time(high_ref)
        t_low = shifted_time(low_ref)
        t_comparisons = {order: shifted_time(waveform) for order, waveform in comparison_waveforms.items()}
    except Exception as exc:
        return _mode_rows_for_global_failure(
            config,
            sxs_id,
            lev_high,
            lev_low,
            f"failed to define merger time: {exc}",
            ["short_window"],
        )

    window = config.waveform.ringdown_window_M
    if not covers_window(t_high, window) or not covers_window(t_low, window):
        return _mode_rows_for_global_failure(
            config,
            sxs_id,
            lev_high,
            lev_low,
            "reference or resolution waveform does not cover [0,30]M",
            ["short_window"],
        )
    for order, time in t_comparisons.items():
        if not covers_window(time, window):
            return _mode_rows_for_global_failure(
                config,
                sxs_id,
                lev_high,
                lev_low,
                f"comparison extrapolation order {order} does not cover [0,30]M",
                ["short_window"],
            )

    t_grid = window_grid(t_high, window)
    if len(t_grid) < config.waveform.min_samples_in_window:
        return _mode_rows_for_global_failure(
            config,
            sxs_id,
            lev_high,
            lev_low,
            "too few samples in reference [0,30]M window",
            ["too_few_samples"],
        )

    try:
        delta_phi = global_resolution_phase_offset(
            high_ref,
            low_ref,
            t_high,
            t_low,
            t_grid,
            interpolation=config.waveform.interpolation,
            fallback=config.waveform.interpolation_fallback,
        )
    except Exception as exc:
        return _mode_rows_for_global_failure(
            config,
            sxs_id,
            lev_high,
            lev_low,
            f"failed global resolution alignment: {exc}",
            ["missing_mode"],
        )

    high_window_mask = (t_high >= float(window[0])) & (t_high <= float(window[1]))
    rows = []
    for mode in config.modes.target_modes:
        rows.append(
            _score_mode(
                config=config,
                sxs_id=sxs_id,
                mode=mode,
                lev_high=lev_high,
                lev_low=lev_low,
                high_ref=high_ref,
                low_ref=low_ref,
                comparison_waveforms=comparison_waveforms,
                t_high=t_high,
                t_low=t_low,
                t_comparisons=t_comparisons,
                t_grid=t_grid,
                high_window_mask=high_window_mask,
                delta_phi=delta_phi,
            )
        )
    return rows


def _score_mode(
    *,
    config: Config,
    sxs_id: str,
    mode: tuple[int, int],
    lev_high: int,
    lev_low: int,
    high_ref: waveform_io.WaveformData,
    low_ref: waveform_io.WaveformData,
    comparison_waveforms: dict[str, waveform_io.WaveformData],
    t_high: np.ndarray,
    t_low: np.ndarray,
    t_comparisons: dict[str, np.ndarray],
    t_grid: np.ndarray,
    high_window_mask: np.ndarray,
    delta_phi: float,
) -> dict[str, Any]:
    ell, m = mode
    row = _mode_base_row(config, sxs_id, mode, lev_high, lev_low)
    row["n_samples"] = int(len(t_grid))

    if mode not in high_ref.modes or mode not in low_ref.modes:
        row["flags"] = "missing_mode"
        row["rejection_reason"] = "target mode is missing"
        return row
    partner_mode = (ell, -m)
    if m != 0 and partner_mode not in high_ref.modes:
        row["flags"] = "missing_partner_mode"
        row["rejection_reason"] = "negative-m symmetry partner is missing"
        return row

    try:
        h_ref = np.asarray(high_ref.modes[mode][high_window_mask], dtype=complex)
        h_partner = (
            np.asarray(high_ref.modes[partner_mode][high_window_mask], dtype=complex) if m != 0 else h_ref
        )
        h_low = interpolate_complex(
            t_low,
            low_ref.modes[mode],
            t_grid,
            method=config.waveform.interpolation,
            fallback=config.waveform.interpolation_fallback,
        )
        h_low_aligned = apply_z_rotation(h_low, m, delta_phi)
        comparison_modes = []
        missing_comparisons = []
        for order, waveform in comparison_waveforms.items():
            if mode not in waveform.modes:
                missing_comparisons.append(order)
                continue
            comparison_modes.append(
                interpolate_complex(
                    t_comparisons[order],
                    waveform.modes[mode],
                    t_grid,
                    method=config.waveform.interpolation,
                    fallback=config.waveform.interpolation_fallback,
                )
            )
    except Exception as exc:
        row["flags"] = "short_window"
        row["rejection_reason"] = f"failed interpolation or window extraction: {exc}"
        return row

    if missing_comparisons or (
        config.filters.require_extrapolation_comparison
        and len(comparison_modes) != len(config.waveform.comparison_extrapolation_orders)
    ):
        row["flags"] = "missing_extrapolation_order"
        row["rejection_reason"] = "required extrapolation comparison is missing"
        return row
    if not np.all(np.isfinite(h_ref)) or not np.all(np.isfinite(h_low_aligned)):
        row["flags"] = "nan_metric"
        row["rejection_reason"] = "interpolation returned NaN or infinity"
        return row

    mode_norm = complex_l2_norm(h_ref, t_grid)
    row["mode_norm"] = mode_norm
    if mode_norm <= config.metrics.norm_floor:
        row["flags"] = "low_mode_norm"
        row["rejection_reason"] = "mode norm is at or below norm floor"
        return row

    delta_res = resolution_error(h_ref, h_low_aligned, t_grid, config.metrics.norm_floor)
    if comparison_modes:
        delta_ext = extrapolation_error(h_ref, comparison_modes, t_grid, config.metrics.norm_floor)
    elif config.filters.require_extrapolation_comparison or config.metrics.weights.get("extrapolation", 0.0) != 0.0:
        row["flags"] = "missing_extrapolation_order"
        row["rejection_reason"] = "required extrapolation comparison is missing"
        return row
    else:
        delta_ext = 0.0
    delta_floor = floor_penalty(
        h_ref,
        h_low_aligned,
        comparison_modes,
        t_grid,
        config.metrics.pointwise_error_floor,
        q=config.metrics.signal_to_error_quantile,
    )
    delta_sym = symmetry_error(h_ref, h_partner, ell, m, t_grid, config.metrics.norm_floor)

    if not metrics_are_finite(delta_res, delta_ext, delta_floor, delta_sym):
        row["delta_res"] = delta_res
        row["delta_ext"] = delta_ext
        row["delta_floor"] = delta_floor
        row["delta_sym"] = delta_sym
        row["flags"] = "nan_metric"
        row["rejection_reason"] = "one or more metrics are NaN or infinity"
        return row

    e_mode, q_mode = mode_quality_score(
        delta_res=delta_res,
        delta_ext=delta_ext,
        delta_floor=delta_floor,
        delta_sym=delta_sym,
        weights=config.metrics.weights,
    )
    flags = _large_metric_flags(delta_res, delta_ext, delta_floor, delta_sym)
    row.update(
        {
            "delta_res": delta_res,
            "delta_ext": delta_ext,
            "delta_floor": delta_floor,
            "delta_sym": delta_sym,
            "E_mode": e_mode,
            "Q_mode": q_mode,
            "valid_mode": True,
            "flags": ";".join(flags),
            "rejection_reason": "",
        }
    )
    return row


def score_candidates(candidates: pd.DataFrame, config: Config) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    iterable = candidates
    if config.runtime.max_simulations is not None:
        iterable = candidates.head(int(config.runtime.max_simulations))
    for count, (_, candidate) in enumerate(iterable.iterrows(), start=1):
        rows.extend(score_candidate(candidate, config))
        if config.runtime.checkpoint_every and count % int(config.runtime.checkpoint_every) == 0:
            checkpoint = Path(config.output.directory) / "per_mode_metrics_checkpoint.csv"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows, columns=PER_MODE_COLUMNS).to_csv(checkpoint, index=False)
    metrics = pd.DataFrame(rows, columns=PER_MODE_COLUMNS)
    return add_mode_percentiles(metrics)


def add_mode_percentiles(per_mode_metrics: pd.DataFrame) -> pd.DataFrame:
    df = per_mode_metrics.copy()
    if "P_mode" not in df.columns:
        df["P_mode"] = np.nan
    valid = df["valid_mode"].astype(bool)
    df.loc[:, "P_mode"] = np.nan
    if valid.any():
        df.loc[valid, "P_mode"] = df.loc[valid].groupby(["ell", "m"])["Q_mode"].rank(method="average", pct=True)
    return df


def aggregate_min_percentile(df_modes_for_one_sim: pd.DataFrame) -> float:
    if df_modes_for_one_sim["P_mode"].isna().any():
        return float("nan")
    return float(df_modes_for_one_sim["P_mode"].min())


def aggregate_geometric_mean_percentile(df_modes_for_one_sim: pd.DataFrame) -> float:
    if df_modes_for_one_sim["P_mode"].isna().any():
        return float("nan")
    values = df_modes_for_one_sim["P_mode"].astype(float).to_numpy()
    return float(np.exp(np.mean(np.log(values + 1.0e-12))))


def _metadata_value(row: pd.Series | None, aliases: list[str]) -> Any:
    if row is None:
        return np.nan
    for alias in aliases:
        if alias in row.index:
            return row[alias]
    return np.nan


def _metadata_float(row: pd.Series | None, aliases: list[str]) -> float:
    value = _metadata_value(row, aliases)
    try:
        return float(value)
    except Exception:
        return float("nan")


def _candidate_lookup(candidates: pd.DataFrame) -> dict[str, pd.Series]:
    if candidates.empty:
        return {}
    if "sxs_id" in candidates.columns:
        return {str(row["sxs_id"]): row for _, row in candidates.iterrows()}
    return {str(index): row for index, row in candidates.iterrows()}


def simulation_scores(per_mode_metrics: pd.DataFrame, candidates: pd.DataFrame, config: Config) -> pd.DataFrame:
    df = add_mode_percentiles(per_mode_metrics)
    lookup = _candidate_lookup(candidates)
    rows = []
    aggregator = (
        aggregate_geometric_mean_percentile
        if config.selection.aggregate_method == "geometric_mean_percentile"
        else aggregate_min_percentile
    )
    expected_modes = set(config.modes.target_modes)
    for sxs_id, group in df.groupby("sxs_id", sort=True):
        present_modes = set(zip(group["ell"].astype(int), group["m"].astype(int)))
        valid_all = expected_modes.issubset(present_modes) and group["valid_mode"].astype(bool).all()
        if config.selection.require_all_modes_valid and not valid_all:
            s_common = float("nan")
        else:
            s_common = aggregator(group)
        candidate = lookup.get(str(sxs_id))
        p_values = group["P_mode"].astype(float)
        e_values = group.loc[group["valid_mode"].astype(bool), "E_mode"].astype(float)
        q_values = group.loc[group["valid_mode"].astype(bool), "Q_mode"].astype(float)
        rows.append(
            {
                "sxs_id": sxs_id,
                "S_common": s_common,
                "aggregate_method": config.selection.aggregate_method,
                "P_min": float(p_values.min(skipna=True)) if p_values.notna().any() else np.nan,
                "P_median": float(p_values.median(skipna=True)) if p_values.notna().any() else np.nan,
                "P_mean": float(p_values.mean(skipna=True)) if p_values.notna().any() else np.nan,
                "E_median": float(e_values.median(skipna=True)) if not e_values.empty else np.nan,
                "Q_median": float(q_values.median(skipna=True)) if not q_values.empty else np.nan,
                "lev_high": _metadata_value(candidate, ["lev_high"]),
                "lev_low": _metadata_value(candidate, ["lev_low"]),
                "reference_mass_ratio": _metadata_value(
                    candidate,
                    ["reference_mass_ratio", "reference-mass-ratio", "mass_ratio", "q"],
                ),
                "reference_chi_eff": _metadata_value(candidate, ["reference_chi_eff", "chi_eff"]),
                "reference_chi1_perp": _metadata_float(candidate, ["reference_chi1_perp"]),
                "reference_chi2_perp": _metadata_float(candidate, ["reference_chi2_perp"]),
                "reference_eccentricity": _metadata_float(candidate, ["reference_eccentricity"]),
                "number_of_orbits": _metadata_value(candidate, ["number_of_orbits", "number-of-orbits"]),
                "valid_all_modes": bool(valid_all),
                "flags": ";".join(sorted(set(flag for flags in group["flags"].dropna() for flag in str(flags).split(";") if flag))),
            }
        )
    scores = pd.DataFrame(rows, columns=SIMULATION_SCORE_COLUMNS)
    return sort_simulation_scores(scores)


def sort_simulation_scores(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return scores
    sorted_scores = scores.copy()
    sorted_scores["_S_sort"] = sorted_scores["S_common"].fillna(-np.inf)
    sorted_scores["_P_median_sort"] = sorted_scores["P_median"].fillna(-np.inf)
    sorted_scores["_E_median_sort"] = sorted_scores["E_median"].fillna(np.inf)
    sorted_scores = sorted_scores.sort_values(
        by=["_S_sort", "_P_median_sort", "_E_median_sort", "sxs_id"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    return sorted_scores.drop(columns=["_S_sort", "_P_median_sort", "_E_median_sort"]).reset_index(drop=True)


def select_simulations(scores: pd.DataFrame, config: Config) -> pd.DataFrame:
    eligible = scores[np.isfinite(scores["S_common"].astype(float))].copy()
    if eligible.empty:
        return eligible
    if config.selection.top_k is not None:
        return eligible.head(int(config.selection.top_k)).reset_index(drop=True)
    if config.selection.top_fraction is not None:
        n_select = max(1, int(math.ceil(len(eligible) * float(config.selection.top_fraction))))
        return eligible.head(n_select).reset_index(drop=True)
    if config.selection.min_common_score is not None:
        return eligible[eligible["S_common"] >= float(config.selection.min_common_score)].reset_index(drop=True)
    return eligible.head(50).reset_index(drop=True)


def selected_per_mode_metrics(per_mode_metrics: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    selected_ids = set(selected["sxs_id"].astype(str))
    return per_mode_metrics[per_mode_metrics["sxs_id"].astype(str).isin(selected_ids)].copy()
