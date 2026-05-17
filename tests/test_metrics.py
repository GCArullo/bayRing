from __future__ import annotations

import numpy as np

from scripts.ringdown_quality.metrics import (
    extrapolation_error,
    floor_penalty,
    resolution_error,
)


def _waveform():
    t = np.linspace(0.0, 30.0, 1000)
    h = np.exp(-t / 10.0) * np.exp(-1j * 0.5 * t)
    return t, h


def test_identical_resolution_waveforms_have_zero_error():
    t, h = _waveform()
    assert resolution_error(h, h, t, 1.0e-14) == 0.0


def test_resolution_noise_increases_error():
    t, h = _waveform()
    clean = resolution_error(h, h, t, 1.0e-14)
    noisy = resolution_error(h, h * (1.0 + 1.0e-2), t, 1.0e-14)
    assert noisy > clean


def test_identical_extrapolation_comparisons_have_zero_error():
    t, h = _waveform()
    assert extrapolation_error(h, [h.copy(), h.copy()], t, 1.0e-14) == 0.0


def test_pointwise_noise_increases_floor_penalty():
    t, h = _waveform()
    clean = floor_penalty(h, h, [h.copy()], t, 1.0e-15, q=0.10)
    noisy = floor_penalty(h, h * 1.05, [h * 0.95], t, 1.0e-15, q=0.10)
    assert noisy > clean
