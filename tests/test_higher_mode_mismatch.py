import csv

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


def test_hm_sum_mismatch_selects_best_polarisation(tmp_path, monkeypatch):
    mode_products = [{
        "mode": (2, 2),
        "time": np.array([0.0, 1.0]),
        "nr": np.array([1.0, 0.0]),
        "model_samples": np.array([
            [2.0, 0.0],
            [2.0, 0.0],
        ]),
    }]
    base_parameters = {
        "I/O": {"outdir": str(tmp_path)},
        "Mismatch-GW-parameters": {
            "M": 1.0,
            "dL": 1.0,
            "ra": 0.0,
            "dec": 0.0,
            "psi": 0.0,
            "azimuth": 0.0,
            "inclination-list": [0.0],
            "polarisation-list": [0.0, 1.0],
            "hm-include-negative-m": 0,
        },
    }

    class FakeAntennaResponse:
        def __init__(self, detector, ra, dec, psi, tensor, times):
            self.plus = psi
            self.cross = 0.0

    def fake_interp1d(times, series, **kwargs):
        return lambda common_time: np.asarray(series)

    def fake_project_modes_to_detector(mode_series, inclination, azimuth, F_plus, F_cross, include_negative_m=True):
        series = next(iter(mode_series.values()))
        is_nr = abs(series[0] - 1.0) < 1e-12
        if(F_plus == 0.0):
            return np.array([1.0, 0.0]) if is_nr else np.array([0.0, 1.0])
        return np.array([1.0, 0.0])

    def fake_toeplitz_whitened_norm(acf, waveform):
        norm = sum(value*value for value in waveform)**0.5
        return waveform, norm

    monkeypatch.setattr(postprocess, "AntennaResponse", FakeAntennaResponse)
    monkeypatch.setattr(postprocess, "interp1d", fake_interp1d)
    monkeypatch.setattr(postprocess, "_project_modes_to_detector", fake_project_modes_to_detector)
    monkeypatch.setattr(postprocess, "_toeplitz_whitened_norm", fake_toeplitz_whitened_norm)
    monkeypatch.setattr(postprocess, "_physical_strain_scale", lambda M, dL: 1.0)
    monkeypatch.setattr(postprocess.np, "dot", lambda left, right: sum(l_value*r_value for l_value, r_value in zip(left, right)), raising=False)
    monkeypatch.setattr(postprocess.np, "minimum", min, raising=False)

    postprocess.compute_higher_mode_sum_mismatch(
        mode_products,
        base_parameters,
        t_start=20.0,
        n_start_times=1,
        acf=np.array([1.0, 0.0]),
        N_FFT=2,
        window_size_DX=0.8,
        window_size_SX=0.8,
        k=7.0,
        saturation_DX=1.0,
        saturation_SX=1.0,
    )

    result_path = tmp_path / "HM_sum" / "Algorithm" / "Mismatch" / "mismatch_and_snr_diagnostics.tsv"
    rows = list(csv.DictReader(result_path.read_text().splitlines(), delimiter="\t"))

    assert len(rows) == 3
    for row in rows:
        assert row["diagnostic_type"] == "higher_mode_sum"
        assert float(row["psi"]) == 1.0
        assert float(row["mismatch"]) == 0.0
