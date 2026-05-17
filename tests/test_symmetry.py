from __future__ import annotations

import numpy as np

from scripts.ringdown_quality.metrics import symmetry_error


def _waveform():
    t = np.linspace(0.0, 30.0, 1000)
    h = np.exp(-t / 10.0) * np.exp(-1j * 0.5 * t)
    return t, h


def test_nonprecessing_partner_symmetry_is_zero():
    t, h_lm = _waveform()
    ell = 3
    h_l_minus_m = (-1) ** ell * np.conjugate(h_lm)
    assert symmetry_error(h_lm, h_l_minus_m, ell, 2, t, 1.0e-14) < 1.0e-12


def test_partner_perturbation_increases_symmetry_error():
    t, h_lm = _waveform()
    ell = 2
    h_l_minus_m = (-1) ** ell * np.conjugate(h_lm)
    clean = symmetry_error(h_lm, h_l_minus_m, ell, 2, t, 1.0e-14)
    perturbed = symmetry_error(h_lm, h_l_minus_m + 0.01 * h_lm, ell, 2, t, 1.0e-14)
    assert perturbed > clean


def test_m_zero_imaginary_part_consistency():
    t = np.linspace(0.0, 30.0, 1000)
    real_mode = np.exp(-t / 10.0)
    complex_mode = real_mode + 1j * 0.01 * real_mode

    assert symmetry_error(real_mode, real_mode, 2, 0, t, 1.0e-14) == 0.0
    assert symmetry_error(complex_mode, complex_mode, 2, 0, t, 1.0e-14) > 0.0
