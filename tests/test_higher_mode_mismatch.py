import numpy as np
import lal

from bayRing import postprocess


def test_project_modes_to_polarizations_uses_lal_spin_weighted_harmonics():
    mode_series = {(2, 2): np.array([1.0 + 2.0j, 3.0 + 4.0j])}
    inclination = 0.4
    azimuth = 0.2

    h_plus, h_cross = postprocess._project_modes_to_polarizations(
        mode_series,
        inclination,
        azimuth,
        include_negative_m=False,
    )

    y_lm = lal.SpinWeightedSphericalHarmonic(inclination, azimuth, -2, 2, 2)
    expected = mode_series[(2, 2)] * y_lm

    np.testing.assert_allclose(h_plus, np.real(expected))
    np.testing.assert_allclose(h_cross, -np.imag(expected))


def test_negative_m_symmetry_fills_missing_counterpart():
    mode_series = {(3, 3): np.array([1.0 + 2.0j])}

    expanded = postprocess._mode_series_with_negative_m_symmetry(mode_series, include_negative_m=True)

    assert (3, -3) in expanded
    np.testing.assert_allclose(expanded[(3, -3)], -np.conjugate(mode_series[(3, 3)]))


def test_mode_percentile_waveform_uses_real_and_imaginary_percentiles():
    model_samples = np.array([
        [1.0 + 4.0j, 2.0 + 5.0j],
        [3.0 + 6.0j, 4.0 + 7.0j],
    ])

    percentile = postprocess._mode_percentile_waveform(model_samples, 50)

    np.testing.assert_allclose(percentile, np.array([2.0 + 5.0j, 3.0 + 6.0j]))
