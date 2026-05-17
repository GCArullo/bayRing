from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline, interp1d

from .metrics import trapezoid
from .waveform_io import WaveformData


def find_merger_time_ell2_norm(waveform: WaveformData) -> float:
    ell2_modes = [waveform.modes[(2, m)] for m in range(-2, 3) if (2, m) in waveform.modes]
    if not ell2_modes:
        raise ValueError(f"{waveform.sxs_id} Lev{waveform.lev} has no ell=2 modes for merger-time finding.")
    stack = np.stack([np.abs(mode) ** 2 for mode in ell2_modes], axis=0)
    norm = np.sqrt(np.sum(stack, axis=0))
    return float(waveform.time[int(np.argmax(norm))])


def shifted_time(waveform: WaveformData) -> np.ndarray:
    return waveform.time - find_merger_time_ell2_norm(waveform)


def _can_cubic(t_src: np.ndarray, method: str) -> bool:
    return method == "cubic" and len(t_src) >= 4 and np.all(np.diff(t_src) > 0)


def interpolate_complex(
    t_src: np.ndarray,
    y_src: np.ndarray,
    t_dst: np.ndarray,
    method: str = "cubic",
    fallback: str = "linear",
) -> np.ndarray:
    t_src = np.asarray(t_src, dtype=float)
    y_src = np.asarray(y_src, dtype=complex)
    t_dst = np.asarray(t_dst, dtype=float)
    finite = np.isfinite(t_src) & np.isfinite(np.real(y_src)) & np.isfinite(np.imag(y_src))
    t_src = t_src[finite]
    y_src = y_src[finite]
    order = np.argsort(t_src)
    t_src = t_src[order]
    y_src = y_src[order]
    unique = np.concatenate(([True], np.diff(t_src) > 0.0)) if len(t_src) else np.array([], dtype=bool)
    t_src = t_src[unique]
    y_src = y_src[unique]
    if len(t_src) < 2:
        raise ValueError("Need at least two source samples for interpolation.")
    if t_dst[0] < t_src[0] or t_dst[-1] > t_src[-1]:
        raise ValueError("Source waveform does not cover the destination grid.")

    use_method = method if _can_cubic(t_src, method) else fallback
    if use_method == "cubic" and _can_cubic(t_src, use_method):
        real = CubicSpline(t_src, np.real(y_src))(t_dst)
        imag = CubicSpline(t_src, np.imag(y_src))(t_dst)
    elif use_method == "linear":
        real = interp1d(t_src, np.real(y_src), kind="linear", bounds_error=True)(t_dst)
        imag = interp1d(t_src, np.imag(y_src), kind="linear", bounds_error=True)(t_dst)
    else:
        raise ValueError(f"Unsupported interpolation method `{method}` with fallback `{fallback}`.")

    result = np.asarray(real) + 1j * np.asarray(imag)
    if not np.all(np.isfinite(result)):
        raise ValueError("Complex interpolation produced NaNs or infinities.")
    return result


def covers_window(t: np.ndarray, window: tuple[float, float] | list[float]) -> bool:
    t = np.asarray(t, dtype=float)
    return bool(len(t) and np.nanmin(t) <= float(window[0]) and np.nanmax(t) >= float(window[1]))


def window_grid(t_ref_shifted: np.ndarray, window: tuple[float, float] | list[float]) -> np.ndarray:
    mask = (t_ref_shifted >= float(window[0])) & (t_ref_shifted <= float(window[1]))
    return np.asarray(t_ref_shifted[mask], dtype=float)


def global_resolution_phase_offset(
    high_ref: WaveformData,
    low_ref: WaveformData,
    t_high_shifted: np.ndarray,
    t_low_shifted: np.ndarray,
    t_grid: np.ndarray,
    *,
    interpolation: str,
    fallback: str,
) -> float:
    if (2, 2) not in high_ref.modes or (2, 2) not in low_ref.modes:
        raise ValueError("The (2,2) mode is required for global resolution alignment.")
    high_22 = interpolate_complex(
        t_high_shifted,
        high_ref.modes[(2, 2)],
        t_grid,
        method=interpolation,
        fallback=fallback,
    )
    low_22 = interpolate_complex(
        t_low_shifted,
        low_ref.modes[(2, 2)],
        t_grid,
        method=interpolation,
        fallback=fallback,
    )
    cross = trapezoid(np.conjugate(high_22) * low_22, t_grid)
    if not np.isfinite(cross):
        raise ValueError("Global phase-alignment integral is not finite.")
    return float(-0.5 * np.angle(cross))


def apply_z_rotation(h_lm: np.ndarray, m: int, delta_phi: float) -> np.ndarray:
    return np.exp(-1j * m * delta_phi) * h_lm
