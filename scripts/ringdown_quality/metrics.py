from __future__ import annotations

import math

import numpy as np


def trapezoid(values: np.ndarray, t: np.ndarray) -> float | complex:
    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, t)
    return np.trapz(values, t)


def complex_l2_norm(y: np.ndarray, t: np.ndarray) -> float:
    return float(np.sqrt(trapezoid(np.abs(y) ** 2, t)))


def resolution_error(h_high: np.ndarray, h_low_aligned: np.ndarray, t: np.ndarray, norm_floor: float) -> float:
    den = max(complex_l2_norm(h_high, t), norm_floor)
    return float(complex_l2_norm(h_high - h_low_aligned, t) / den)


def extrapolation_error(
    h_ref: np.ndarray,
    h_comparison_list: list[np.ndarray],
    t: np.ndarray,
    norm_floor: float,
) -> float:
    den = max(complex_l2_norm(h_ref, t), norm_floor)
    vals = [complex_l2_norm(h_ref - h_cmp, t) / den for h_cmp in h_comparison_list]
    if not vals:
        return float("nan")
    return float(np.median(vals))


def floor_penalty(
    h_ref: np.ndarray,
    h_low_aligned: np.ndarray,
    h_comparison_list: list[np.ndarray],
    t: np.ndarray,
    pointwise_floor: float,
    q: float = 0.10,
) -> float:
    del t
    res_err2 = np.abs(h_ref - h_low_aligned) ** 2
    if h_comparison_list:
        ext_stack = np.stack([np.abs(h_ref - h_cmp) ** 2 for h_cmp in h_comparison_list], axis=0)
        ext_err2 = np.median(ext_stack, axis=0)
    else:
        ext_err2 = 0.0
    eps = np.sqrt(res_err2 + ext_err2)
    ser = np.abs(h_ref) / (eps + pointwise_floor)
    finite = ser[np.isfinite(ser)]
    if finite.size == 0:
        return float("inf")
    ser_q = float(np.quantile(finite, q))
    if ser_q <= 0 or not np.isfinite(ser_q):
        return float("inf")
    return float(1.0 / ser_q)


def symmetry_error(
    h_target: np.ndarray,
    h_partner: np.ndarray,
    ell: int,
    m: int,
    t: np.ndarray,
    norm_floor: float,
) -> float:
    if m == 0:
        return symmetry_error_m0(h_target, t, norm_floor)
    residual = h_partner - ((-1) ** ell) * np.conjugate(h_target)
    num = complex_l2_norm(residual, t)
    den = math.sqrt(
        complex_l2_norm(h_partner, t) ** 2 + complex_l2_norm(h_target, t) ** 2 + norm_floor**2
    )
    return float(num / den)


def symmetry_error_m0(h_mode: np.ndarray, t: np.ndarray, norm_floor: float) -> float:
    den = max(complex_l2_norm(h_mode, t), norm_floor)
    return float(complex_l2_norm(np.imag(h_mode), t) / den)


def mode_quality_score(
    *,
    delta_res: float,
    delta_ext: float,
    delta_floor: float,
    delta_sym: float,
    weights: dict[str, float],
) -> tuple[float, float]:
    e_mode = math.sqrt(
        weights["resolution"] * delta_res**2
        + weights["extrapolation"] * delta_ext**2
        + weights["floor"] * delta_floor**2
        + weights["symmetry"] * delta_sym**2
    )
    q_mode = 1.0 / (e_mode + 1.0e-12)
    return float(e_mode), float(q_mode)


def metrics_are_finite(*values: float) -> bool:
    return all(np.isfinite(value) for value in values)
