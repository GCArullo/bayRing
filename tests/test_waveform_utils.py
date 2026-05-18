import cmath
import math

import numpy as np

from bayRing import waveform_utils


def _array(values):
    return np.array(values)


def test_amp_phase_from_re_im_returns_expected_amplitude_and_phase():
    h_re = _array([3.0, 0.0])
    h_im = _array([4.0, 5.0])

    amp, phase = waveform_utils.amp_phase_from_re_im(h_re, h_im)

    assert list(amp) == [5.0, 5.0]
    expected_phase = [cmath.phase(3.0 - 4.0j), cmath.phase(-5.0j)]
    for actual, expected in zip(phase, expected_phase):
        assert abs(actual - expected) < 1e-6


def test_mismatch_waveforms_zero_for_identical_inputs():
    time = _array([0.0, 0.5, 1.0, 1.5])

    def amp_func(sample_times):
        return _array([1.0 for _ in sample_times])

    def phase_func(sample_times):
        return _array([0.1 * value for value in sample_times])

    delta = np.array([0.0, 0.0])
    mismatch = waveform_utils.mismatch_waveforms(
        delta,
        time,
        amp_func,
        amp_func,
        phase_func,
        phase_func,
        0.0,
        1.5,
    )

    assert abs(mismatch) < 1e-12


def test_resolve_mismatch_window_preserves_legacy_fractional_inputs():
    t_min, t_max = waveform_utils._resolve_mismatch_window(100.0, 0.3, 0.004)
    assert math.isclose(t_min, 70.0)
    assert math.isclose(t_max, 99.6)


def test_resolve_mismatch_window_uses_absolute_offsets_for_current_default():
    assert waveform_utils._resolve_mismatch_window(100.0, 0.0, 30.0) == (100.0, 130.0)


def test_find_peak_time_uses_last_peak_for_eccentric_waveforms(monkeypatch):
    time = _array([0.0, 1.0, 2.0, 3.0])
    amplitude = _array([0.1, 0.5, 0.4, 0.9])

    monkeypatch.setattr(
        waveform_utils,
        "find_peaks",
        lambda data, height=None: ([1, 3], {"peak_heights": [0.5, 0.9]}),
    )

    peak_time = waveform_utils.find_peak_time(time, amplitude, ecc=0.2)

    assert peak_time == 3.0


def test_find_peak_time_defaults_to_maximum_for_low_eccentricity():
    time = _array([0.0, 1.0, 2.0])
    amplitude = _array([0.1, 0.7, 0.5])

    peak_time = waveform_utils.find_peak_time(time, amplitude, ecc=1e-4)

    assert peak_time == 1.0
