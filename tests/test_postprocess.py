from types import SimpleNamespace

import numpy as np
import pytest

from bayRing import postprocess


MODEL_FAMILIES = [
    "Kerr",
    "Damped-sinusoids",
    "Kerr-Damped-sinusoids",
    "KerrBinary",
    "TEOBPM",
]


class FakeInferenceModel:
    def __init__(self, wf_model):
        self.wf_model = wf_model
        self.calls = []

    def model(self, sample):
        t_eval = np.array(self.wf_model.t_NR, dtype=float)
        self.calls.append(t_eval.copy())
        scale = sample.get("scale", 1.0)
        phase = sample.get("phase", 0.0)
        return scale * np.exp(-0.03 * t_eval) * np.exp(1j * (0.2 * t_eval + phase))


def _nr_simulation():
    t_NR = np.array([-4.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0])
    NR_r = np.cos(0.2 * t_NR)
    NR_i = np.sin(0.2 * t_NR)
    NR_freq = np.array([0.2 / postprocess.twopi for _ in t_NR])
    NR_err_r = np.array([0.01 for _ in t_NR])
    NR_err_i = np.array([0.02 for _ in t_NR])

    return SimpleNamespace(
        t_NR=t_NR,
        t_peak=0.0,
        NR_r=NR_r,
        NR_i=NR_i,
        NR_amp=np.sqrt(NR_r**2 + NR_i**2),
        NR_freq=NR_freq,
        NR_err_cmplx=NR_err_r + 1j * NR_err_i,
    )


@pytest.mark.parametrize("wf_model_name", MODEL_FAMILIES)
def test_pre_fit_extrapolation_uses_peak_to_fit_start_for_each_model_family(monkeypatch, wf_model_name):
    original_grid = np.array([10.0, 12.0, 14.0])
    wf_model = SimpleNamespace(wf_model=wf_model_name, t_NR=original_grid.copy())
    inference_model = FakeInferenceModel(wf_model)
    samples = [{"scale": 0.95, "phase": -0.03}, {"scale": 1.05, "phase": 0.03}]
    NR_sim = _nr_simulation()

    def fake_quantiles(models_re_list, models_im_list, t_eval):
        t_eval = postprocess._as_1d_float_array(t_eval)
        return {
            perc: {
                "real": t_eval + perc,
                "imag": t_eval + 2.0 * perc,
                "amp": t_eval + 3.0 * perc,
                "freq": t_eval + 4.0 * perc,
            }
            for perc in [5, 50, 95]
        }

    monkeypatch.setattr(postprocess, "_model_waveform_quantiles", fake_quantiles)

    extrapolation = postprocess._pre_fit_extrapolation(
        NR_sim, samples, inference_model, None, original_grid
    )

    np.testing.assert_allclose(extrapolation["t"], [0.0, 2.0, 4.0, 6.0, 8.0])
    np.testing.assert_allclose(extrapolation["x"], extrapolation["t"] - NR_sim.t_peak)
    np.testing.assert_allclose(wf_model.t_NR, original_grid)
    assert extrapolation["has_band"] is True
    assert len(inference_model.calls) == len(samples)

    expected_eval_grid = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0])
    for call_grid in inference_model.calls:
        np.testing.assert_allclose(call_grid, expected_eval_grid)

    for quantity in ["real", "imag", "amp", "freq"]:
        assert extrapolation["quantiles"][50][quantity].shape == extrapolation["t"].shape

    np.testing.assert_allclose(extrapolation["NR"]["real"], NR_sim.NR_r[1:6])
    np.testing.assert_allclose(extrapolation["NR"]["imag_err"], [0.02 for _ in range(5)])


def test_pre_fit_extrapolation_returns_none_when_fit_starts_at_peak():
    original_grid = np.array([0.0, 2.0, 4.0])
    wf_model = SimpleNamespace(wf_model="Kerr", t_NR=original_grid.copy())
    inference_model = FakeInferenceModel(wf_model)

    extrapolation = postprocess._pre_fit_extrapolation(
        _nr_simulation(), [{"scale": 1.0}], inference_model, None, original_grid
    )

    assert extrapolation is None
    np.testing.assert_allclose(wf_model.t_NR, original_grid)
    assert inference_model.calls == []
