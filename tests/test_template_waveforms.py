import cmath
import math

import numpy as np

from bayRing import template_waveforms


def _array(values):
    return np.array(values)


def _basic_metadata(**overrides):
    metadata = {
        "Mf": 1.0,
        "af": 0.5,
        "m1": 30.0,
        "m2": 20.0,
        "chi1": 0.1,
        "chi2": -0.2,
    }
    metadata.update(overrides)
    return metadata


def _build_model(**kwargs):
    params = {
        "t_NR": _array([0.0, 1.0]),
        "tM_start": 0.0,
        "tM_peak": 1.0,
        "wf_model": "Kerr",
        "N_ds_modes": 0,
        "Kerr_modes": [(2, 2, 0)],
        "metadata": _basic_metadata(**kwargs.pop("metadata_overrides", {})),
        "fit_metadata": None,
        "qnm_cached": "cached",
        "l_NR": 2,
        "m_NR": 2,
        "tail": 0,
        "tail_modes": None,
        "quadratic_modes": None,
        "const_params": None,
    }
    params.update(kwargs)
    return template_waveforms.WaveformModel(**params)


def test_kerr_waveform_collects_mode_parameters(monkeypatch):
    model = _build_model()

    params = {"ln_A_220": math.log(2.5), "phi_220": 0.3}
    fixed = {}

    monkeypatch.setattr(
        template_waveforms.utils,
        "get_param_override",
        lambda fixed_params, sample, key: sample[key],
    )

    captured = {}

    def fake_kerrbh(*args, **kwargs):
        captured.update(kwargs)
        captured["amps"] = args[3]
        return "ringdown"

    monkeypatch.setattr(template_waveforms.wf, "KerrBH", fake_kerrbh)

    result = model.Kerr_waveform(params, fixed)

    expected_amp = cmath.exp(params["ln_A_220"] + 1j * params["phi_220"])
    assert captured["amps"][(2, 2, 2, 0)] == expected_amp
    assert captured["tail_parameters"] == {}
    assert captured["quadratic_modes"] == {}
    assert captured["TGR_params"] is None
    assert model.charge == 0
    assert result == "ringdown"


def test_kerr_waveform_activates_charge_when_q_present(monkeypatch):
    model = _build_model(metadata_overrides={"qf": 0.8})

    params = {"ln_A_220": 0.0, "phi_220": 0.0}

    monkeypatch.setattr(
        template_waveforms.utils,
        "get_param_override",
        lambda fixed_params, sample, key: sample[key],
    )

    captured = {}

    def fake_kerrbh(*args, **kwargs):
        captured.update(kwargs)
        return "charged"

    monkeypatch.setattr(template_waveforms.wf, "KerrBH", fake_kerrbh)

    result = model.Kerr_waveform(params, {})

    assert captured["TGR_params"] == {"Q": 0.8}
    assert captured["charge"] == 1
    assert model.charge == 1
    assert result == "charged"


def test_kerrbinary_waveform_passes_filtered_modes(monkeypatch):
    model = _build_model()

    params = {"phi": 0.25}

    monkeypatch.setattr(
        template_waveforms.utils,
        "get_param_override",
        lambda fixed_params, sample, key: sample[key],
    )

    def fake_filter_dict(source, target_key):
        return {"linear": {target_key: ["filtered"]}, "quadratic": {}}

    monkeypatch.setattr(template_waveforms.utils, "filter_dict_by_key", fake_filter_dict)

    captured = {}

    def fake_kerrbinary(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        return "binary"

    monkeypatch.setattr(template_waveforms.wf, "KerrBinary", fake_kerrbinary)

    result = model.KerrBinary_waveform(params, {})

    assert captured["modes"] == {"linear": {(2, 2): ["filtered"]}, "quadratic": {}}
    assert captured["args"][10] == params["phi"]
    assert result == "binary"


def test_teobpm_waveform_passes_string_template(monkeypatch):
    model = _build_model(
        wf_model="TEOBPM",
        TEOB_template="RatExp",
        TEOB_merger_data=1,
        TEOB_global_fit=0,
        metadata_overrides={
            "omg_peak_22": 0.3,
            "A_peak_22": 0.4,
            "A_peak22dotdot": -0.01,
        },
    )

    params = {
        "phi_mrg_22": 0.0,
        "c2A_22": 1.0,
        "c2p_22": 1.0,
        "c3A_22": 1.0,
        "c3p_22": 1.0,
        "c4p_22": 1.0,
    }
    captured = {}

    def fake_teobpm(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        return "teob"

    monkeypatch.setattr(template_waveforms.wf, "TEOBPM", fake_teobpm, raising=False)

    result = model.TEOBPM_waveform(params, {})

    assert captured["template"] == "RatExp"
    assert not isinstance(captured["template"], int)
    assert captured["merger_data"] == 1
    assert captured["global_fit"] == 0
    assert result == "teob"


def test_teobpm_development_path_keeps_nr_fit_sigmoid_and_quadratic(monkeypatch):
    model = _build_model(
        wf_model="TEOBPM",
        TEOB_template="HypTan",
        TEOB_NR_fit=1,
        TEOB_global_fit=1,
        TEOB_merger_data=0,
        TEOB_qc_fit_type="equal-mass",
        sigmoid_flag=1,
        quadratic_fits=1,
        quadratic_modes={"sum": [((4, 4, 0), (2, 2, 0), (2, 2, 0))]},
    )

    params = {
        "phi_mrg_22": 0.0,
        "c3A_22": 1.0,
        "c3p_22": 2.0,
        "c4p_22": 3.0,
        "ln_A_220": math.log(2.0),
        "phi_220": 0.1,
        "dphi_220": 0.2,
        "t_sigmoid_220": 4.0,
        "width_sigmoid_220": 5.0,
        "ln_A_sum_440_220_220": math.log(3.0),
        "phi_sum_440_220_220": 0.3,
        "dphi_sum_440_220_220": 0.4,
        "t_sigmoid_sum_440_220_220": 6.0,
        "width_sigmoid_sum_440_220_220": 7.0,
    }
    captured = {}

    def fake_teobpm(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        return "teob-dev"

    monkeypatch.setattr(template_waveforms.wf, "TEOBPM", fake_teobpm, raising=False)

    result = model.TEOBPM_waveform(params, {})

    assert captured["template"] == 0
    assert captured["ecc_par"] == 0
    assert captured["sigmoid_flag"] == 1
    assert captured["quadratic_fits"] == 1
    assert captured["order_S_hat"] == 342
    assert captured["NR_fit_coeffs"][(2, 2)]["c3A"] == 1.0
    assert abs(captured["args"][6][(2, 2, 2, 0)] - 2.0 * cmath.exp(1j * 0.1)) < 1e-12
    assert abs(captured["args"][10]["sum"][((2, 4, 4, 0), (2, 2, 2, 0), (2, 2, 2, 0))] - 3.0 * cmath.exp(1j * 0.3)) < 1e-12
    assert result == "teob-dev"
