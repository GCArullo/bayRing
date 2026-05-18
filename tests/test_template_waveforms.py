import cmath
import math

import numpy as np
import pytest

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
        captured["t_ref"] = args[1]
        captured["amps"] = args[4]
        return "ringdown"

    monkeypatch.setattr(template_waveforms.wf, "KerrBH", fake_kerrbh)

    result = model.Kerr_waveform(params, fixed)

    expected_amp = cmath.exp(params["ln_A_220"] + 1j * params["phi_220"])
    assert captured["amps"][(2, 2, 2, 0)] == expected_amp
    assert captured["tail_parameters"] == {}
    assert captured["quadratic_modes"] == {}
    assert captured["t_ref"] == model.t_peak
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
            "DeltaT_22": 0.0,
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
    assert captured["calibration"] == "qc"
    assert not isinstance(captured["template"], int)
    assert captured["merger_data"] == 1
    assert captured["global_fit"] == 0
    assert captured["NR_fit_coeffs"][(2, 2)]["omg_peak"] == 0.3
    assert captured["NR_fit_coeffs"][(2, 2)]["A_peak_over_nu"] == 0.4 / (30.0 * 20.0 / 50.0**2)
    assert captured["NR_fit_coeffs"][(2, 2)]["A_peakdotdot_over_nu"] == -0.01 / (30.0 * 20.0 / 50.0**2)
    assert captured["NR_fit_coeffs"][(2, 2)]["DeltaT"] == 0.0


def test_teobpm_waveform_passes_seobnrv5_template_without_c4p(monkeypatch):
    model = _build_model(
        wf_model="TEOBPM",
        TEOB_template="SEOBNRv5",
        TEOB_merger_data=0,
        TEOB_global_fit=0,
    )

    params = {
        "phi_mrg_22": 0.0,
        "c2A_22": 0.11,
        "c3A_22": -0.5,
        "c2p_22": 0.21,
        "c3p_22": 3.0,
    }
    captured = {}

    def fake_teobpm(*args, **kwargs):
        captured.update(kwargs)
        return "teob"

    monkeypatch.setattr(template_waveforms.wf, "TEOBPM", fake_teobpm, raising=False)

    result = model.TEOBPM_waveform(params, {})

    assert result == "teob"
    assert captured["template"] == "SEOBNRv5"
    assert captured["NR_fit_coeffs"][(2, 2)]["c2A"] == 0.11
    assert captured["NR_fit_coeffs"][(2, 2)]["c3A"] == -0.5
    assert captured["NR_fit_coeffs"][(2, 2)]["c2p"] == 0.21
    assert captured["NR_fit_coeffs"][(2, 2)]["c3p"] == 3.0
    assert captured["NR_fit_coeffs"][(2, 2)]["c4p"] == 0.0


def test_teobpm_waveform_passes_mode_mixing_parent_inputs(monkeypatch):
    model = _build_model(
        wf_model="TEOBPM",
        l_NR=3,
        m_NR=2,
        TEOB_template="HypTan",
        TEOB_merger_data=0,
        TEOB_global_fit=0,
        TEOB_mode_mixing=1,
    )

    params = {
        "phi_mrg_32": 0.4,
        "DeltaT_32": 7.5,
        "A_ref_over_nu_32": 0.08,
        "A_refdot_over_nu_32": -0.002,
        "omg_ref_32": 0.24,
        "c2A_32": 0.7,
        "c3A_32": -0.3,
        "c2p_32": 0.8,
        "c3p_32": 4.0,
        "c4p_32": 3.0,
    }
    fixed = {
        "phi_mrg_22": 0.2,
        "DeltaT_22": 7.5,
        "A_ref_over_nu_22": 0.35,
        "omg_ref_22": 0.32,
        "c2A_22": 0.9,
        "c3A_22": -0.5,
        "c2p_22": 1.1,
        "c3p_22": 5.0,
        "c4p_22": 2.0,
    }
    captured = {}

    def fake_teobpm(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        return "teob-mixed"

    monkeypatch.setattr(template_waveforms.wf, "TEOBPM", fake_teobpm, raising=False)

    result = model.TEOBPM_waveform(params, fixed)

    assert result == "teob-mixed"
    assert captured["mode_mixing"] == {(3, 2): (2, 2)}
    assert captured["args"][9] == [(3, 2)]
    assert captured["args"][5][(3, 2)] == 0.4
    assert captured["args"][5][(2, 2)] == 0.2
    assert captured["NR_fit_coeffs"][(3, 2)]["DeltaT"] == 7.5
    assert captured["NR_fit_coeffs"][(2, 2)]["DeltaT"] == 7.5
    assert captured["NR_fit_coeffs"][(3, 2)]["A_ref_over_nu"] == 0.08
    assert captured["NR_fit_coeffs"][(3, 2)]["A_refdot_over_nu"] == -0.002
    assert captured["NR_fit_coeffs"][(3, 2)]["omg_ref"] == 0.24
    assert captured["NR_fit_coeffs"][(2, 2)]["A_ref_over_nu"] == 0.35
    assert captured["NR_fit_coeffs"][(2, 2)]["omg_ref"] == 0.32
    assert captured["NR_fit_coeffs"][(3, 2)]["c2A"] == 0.7
    assert captured["NR_fit_coeffs"][(3, 2)]["c2p"] == 0.8
    assert captured["NR_fit_coeffs"][(2, 2)]["c2A"] == 0.9
    assert captured["NR_fit_coeffs"][(2, 2)]["c2p"] == 1.1
    assert captured["NR_fit_coeffs"][(3, 2)]["c3A"] == -0.3
    assert captured["NR_fit_coeffs"][(2, 2)]["c3A"] == -0.5


def test_teobpm_quadratic_44_uses_parent_22_template_and_qqnm_ratio(monkeypatch):
    model = _build_model(
        wf_model="TEOBPM",
        l_NR=4,
        m_NR=4,
        t_NR=_array([0.0, 1.0, 2.0]),
        tM_peak=0.0,
        metadata_overrides={"af": 0.0},
        qnm_cached={},
        TEOB_template="HypTan",
        TEOB_global_fit=0,
        TEOB_merger_data=0,
        TEOB_quadratic_44=1,
        TEOB_quadratic_44_window_start=0.0,
        TEOB_quadratic_44_window_width=0.0,
        TEOB_quadratic_44_ratio_fit="khera-total",
    )

    class FakeTEOBPM:
        def waveform(self, times):
            return None, None, None, -np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0])

    monkeypatch.setattr(template_waveforms.wf, "TEOBPM", lambda *args, **kwargs: FakeTEOBPM(), raising=False)

    params = {}
    fixed = {
        "phi_mrg_22": 0.1,
        "c3A_22": -0.2,
        "c3p_22": 3.0,
        "c4p_22": 2.0,
    }

    waveform = model.TEOBPM_quadratic_44_waveform(params, fixed)
    ratio = (
        0.137 * np.exp(-1j * 0.08375)
        + 0.01651 * np.exp(1j * 0.05858)
    )
    expected = ratio * np.array([1.0, 2.0, 3.0])**2

    np.testing.assert_allclose(waveform, expected)


def test_teobpm_quadratic_44_rejects_other_modes():
    model = _build_model(
        wf_model="TEOBPM",
        l_NR=3,
        m_NR=3,
        qnm_cached={(2, 2, 2, 0): {"f": 0.1, "tau": 10.0}},
        TEOB_quadratic_44=1,
    )

    with pytest.raises(ValueError, match="selected NR mode"):
        model.TEOBPM_quadratic_44_waveform({}, {})


def test_teobpm_quadratic_44_samples_target_peak_window_parameters():
    model = _build_model(
        wf_model="TEOBPM",
        l_NR=4,
        m_NR=4,
        t_NR=_array([10.0, 13.0, 15.0, 17.0]),
        tM_peak=10.0,
        metadata_overrides={"DeltaT_44": 2.0},
        qnm_cached={},
        TEOB_quadratic_44=1,
        TEOB_quadratic_44_window_start=0.0,
        TEOB_quadratic_44_window_width=10.0,
        TEOB_quadratic_44_window_steepness=1.0,
    )

    params = {
        "quad44_window_delay": 1.0,
        "quad44_window_width": 4.0,
        "quad44_window_steepness": 2.0,
    }

    start, width, steepness = model._TEOBPM_quadratic_44_window_parameters(params, {})
    window = model._TEOBPM_quadratic_44_window(params, {})

    assert start == 3.0
    assert width == 4.0
    assert steepness == 2.0
    np.testing.assert_allclose(window, [0.0, 0.0, 0.5, 1.0])


def test_teobpm_quadratic_44_window_uses_global_fit_metadata():
    model = _build_model(
        wf_model="TEOBPM",
        l_NR=4,
        m_NR=4,
        t_NR=_array([10.0, 12.0, 14.0, 16.0]),
        tM_peak=10.0,
        metadata_overrides={
            "DeltaT_44": 2.0,
            "chi1z": 0.2,
            "chi2z": -0.1,
        },
        fit_metadata={
            "fit_type": "nu_chi_eff",
            "fit_order": "1",
            "quad44_window_delay_p0": "1.0",
            "quad44_window_delay_p1": "2.0",
            "quad44_window_delay_p2": "3.0",
            "quad44_window_width": "4.0",
            "quad44_window_steepness": "2.0",
        },
        qnm_cached={},
        TEOB_global_fit=1,
        TEOB_quadratic_44=1,
        TEOB_quadratic_44_window_start=0.0,
        TEOB_quadratic_44_window_width=10.0,
        TEOB_quadratic_44_window_steepness=1.0,
    )

    start, width, steepness = model._TEOBPM_quadratic_44_window_parameters({}, {})
    nu = (30.0*20.0)/(30.0 + 20.0)**2
    chi_eff = (30.0*0.2 + 20.0*(-0.1))/(30.0 + 20.0)

    assert start == pytest.approx(2.0 + 1.0 + 2.0*nu + 3.0*chi_eff)
    assert width == 4.0
    assert steepness == 2.0


def test_teobpm_quadratic_44_fixed_window_end_derives_width_from_delay():
    model = _build_model(
        wf_model="TEOBPM",
        l_NR=4,
        m_NR=4,
        t_NR=_array([10.0, 15.0, 20.0, 25.0, 30.0]),
        tM_peak=10.0,
        metadata_overrides={"DeltaT_44": 2.0},
        qnm_cached={},
        TEOB_quadratic_44=1,
        TEOB_quadratic_44_window_start=0.0,
        TEOB_quadratic_44_window_width=10.0,
        TEOB_quadratic_44_window_end=20.0,
        TEOB_quadratic_44_window_steepness=1.0,
    )

    start, width, steepness = model._TEOBPM_quadratic_44_window_parameters(
        {"quad44_window_delay": 5.0, "quad44_window_steepness": 2.0},
        {},
    )

    assert start == 7.0
    assert width == 15.0
    assert steepness == 2.0


def test_teobpm_quadratic_44_fixed_window_end_rejects_width_parameter():
    model = _build_model(
        wf_model="TEOBPM",
        l_NR=4,
        m_NR=4,
        t_NR=_array([10.0, 15.0, 20.0]),
        tM_peak=10.0,
        metadata_overrides={"DeltaT_44": 2.0},
        qnm_cached={},
        TEOB_quadratic_44=1,
        TEOB_quadratic_44_window_end=20.0,
    )

    with pytest.raises(ValueError, match="quad44_window_width"):
        model._TEOBPM_quadratic_44_window_parameters(
            {"quad44_window_delay": 5.0, "quad44_window_width": 10.0},
            {},
        )


def test_teobpm_quadratic_waveform_uses_selected_qqnm_term():
    model = _build_model(
        wf_model="TEOBPM",
        l_NR=5,
        m_NR=5,
        t_NR=_array([0.0, 1.0]),
        tM_peak=0.0,
        quadratic_modes={"sum": [((5, 5, 0), (2, 2, 0), (3, 3, 0))], "diff": []},
        qnm_cached={
            (2, 2, 2, 0): {"f": 0.10, "tau": 10.0},
            (2, 3, 3, 0): {"f": 0.15, "tau": 15.0},
        },
    )

    waveform = model.TEOBPM_quadratic_waveform(
        {"ln_A_sum_550_220_330": math.log(2.0), "phi_sum_550_220_330": 0.3},
        {},
    )

    tau = (10.0 * 15.0)/(10.0 + 15.0)
    expected = 2.0*np.exp(1j*0.3)*np.exp(-model.t_NR/tau)*np.exp(-1j*2.0*np.pi*0.25*model.t_NR)
    np.testing.assert_allclose(waveform, expected)


def test_teobpm_tapered_overtone_44_uses_441_basis():
    model = _build_model(
        wf_model="TEOBPM",
        l_NR=4,
        m_NR=4,
        t_NR=_array([0.0, 1.0, 2.0]),
        tM_peak=0.0,
        qnm_cached={(2, 4, 4, 1): {"f": 0.3, "tau": 4.0}},
        TEOB_tapered_overtone_44=1,
        TEOB_tapered_overtone_44_window_start=0.0,
        TEOB_tapered_overtone_44_window_width=0.0,
    )

    params = {
        "ln_A_tapered_441": math.log(3.0),
        "phi_tapered_441": 0.4,
    }

    waveform = model.TEOBPM_tapered_overtone_44_waveform(params, {})
    t_rel = np.array([0.0, 1.0, 2.0])
    expected = 3.0*np.exp(1j*0.4)*np.exp(-t_rel/4.0)*np.exp(-1j*2.0*np.pi*0.3*t_rel)

    np.testing.assert_allclose(waveform, expected)


def test_teobpm_counter_rotating_waveform_uses_time_dependent_teob_branch(monkeypatch):
    model = _build_model(
        wf_model="TEOBPM",
        l_NR=2,
        m_NR=1,
        TEOB_counter_rotating=1,
        TEOB_template="HypTan",
        TEOB_merger_data=0,
    )

    captured = {}

    class FakeTEOBPM:
        def waveform(self, times):
            captured["times"] = np.array(times)
            return None, None, None, np.array([1.0, 2.0]), np.array([3.0, 4.0])

    def fake_teobpm(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeTEOBPM()

    monkeypatch.setattr(template_waveforms.wf, "TEOBPM", fake_teobpm, raising=False)

    waveform = model.TEOBPM_counter_rotating_waveform(
        {
            "ln_A_counter_scale_2-1": math.log(0.5),
            "phi_mrg_counter_2-1": 0.7,
            "c2A_counter_2-1": 0.9,
            "c3A_counter_2-1": -0.2,
            "c2p_counter_2-1": 1.1,
            "c3p_counter_2-1": 1.5,
            "c4p_counter_2-1": 0.4,
        },
        {},
    )

    expected = 0.5 * np.conjugate(np.array([-1.0, -2.0]) + 1j*np.array([3.0, 4.0]))

    assert captured["args"][5][(2, 1)] == 0.7
    assert captured["args"][9] == [(2, 1)]
    assert captured["kwargs"]["global_fit"] == 0
    assert "mode_mixing" not in captured["kwargs"]
    assert captured["kwargs"]["NR_fit_coeffs"][(2, 1)]["c3A"] == -0.2
    assert captured["kwargs"]["NR_fit_coeffs"][(2, 1)]["c2A"] == 0.9
    assert captured["kwargs"]["NR_fit_coeffs"][(2, 1)]["c2p"] == 1.1
    assert captured["kwargs"]["NR_fit_coeffs"][(2, 1)]["c3p"] == 1.5
    assert captured["kwargs"]["NR_fit_coeffs"][(2, 1)]["c4p"] == 0.4
    np.testing.assert_allclose(waveform, expected)


def test_teobpm_qc_global_fit_rejects_noncircular_fit_metadata():
    model = _build_model(
        wf_model="TEOBPM",
        TEOB_template="RatExp",
        TEOB_merger_data=0,
        TEOB_global_fit=1,
        fit_metadata={"fit_type": "nu_ecc", "fit_order": "1"},
    )

    with pytest.raises(ValueError, match="TEOB-calibration = noncirc"):
        model.TEOBPM_waveform({"phi_mrg_22": 0.0}, {})


def test_teobpm_noncirc_global_fit_accepts_hyptan_metadata(monkeypatch):
    model = _build_model(
        wf_model="TEOBPM",
        TEOB_template="HypTan",
        TEOB_calibration="noncirc",
        TEOB_merger_data=0,
        TEOB_global_fit=1,
        fit_metadata={"fit_type": "nu_bmrg", "fit_order": "1", "c_3_A_p0": "0.1"},
        metadata_overrides={"bmrg": 2.5},
    )
    captured = {}

    def fake_teobpm(*args, **kwargs):
        captured.update(kwargs)
        return "teob"

    monkeypatch.setattr(template_waveforms.wf, "TEOBPM", fake_teobpm, raising=False)

    assert model.TEOBPM_waveform({"phi_mrg_22": 0.0}, {}) == "teob"
    assert captured["template"] == "HypTan"
    assert captured["calibration"] == "noncirc"
    assert captured["NR_fit_coeffs"]["bmrg"] == 2.5


def test_teobpm_qc_global_fit_accepts_ratexp_nu_metadata(monkeypatch):
    model = _build_model(
        wf_model="TEOBPM",
        TEOB_template="RatExp",
        TEOB_calibration="qc",
        TEOB_merger_data=1,
        TEOB_global_fit=1,
        fit_metadata={"fit_type": "nu", "fit_order": "1", "c_2_A_p0": "0.1"},
        metadata_overrides={
            "omg_peak_22": 0.3,
            "A_peak_22": 0.4,
            "A_peak22dotdot": -0.01,
        },
    )
    captured = {}

    def fake_teobpm(*args, **kwargs):
        captured.update(kwargs)
        return "teob"

    monkeypatch.setattr(template_waveforms.wf, "TEOBPM", fake_teobpm, raising=False)

    assert model.TEOBPM_waveform({"phi_mrg_22": 0.0}, {}) == "teob"
    assert captured["template"] == "RatExp"
    assert captured["calibration"] == "qc"
    assert captured["NR_fit_coeffs"][(2, 2)]["fit_type"] == "nu"
    assert "bmrg" not in captured["NR_fit_coeffs"]
