import os
import pickle

import numpy as np
import pytest

from bayRing import inference


class FakeConfig:
    def __init__(self, values):
        self.values = values

    def getfloat(self, section, option):
        try:
            return self.values[(section, option)]
        except KeyError as exc:
            raise inference.configparser.NoOptionError(option, section) from exc


class PickleBase:
    pass


class FakeWaveform:
    def __init__(self, wf_model, **overrides):
        self.wf_model = wf_model
        self.Kerr_modes = []
        self.N_ds_modes = 0
        self.TEOB_NR_fit = 0
        self.TEOB_template = "qc"
        self.tail = 0
        self.tail_modes = []
        self.quadratic_modes = None
        self.l_NR = 2
        self.m_NR = 2
        self.TEOB_qc_fit_type = "equal-mass"
        for key, value in overrides.items():
            setattr(self, key, value)

    def waveform(self, params, fixed_params):
        return np.array([0j, 0j, 0j])


def test_read_parameter_bounds_uses_config_value(capsys):
    config = FakeConfig({("Priors", "ln_A_220-min"): -1.5, ("Priors", "ln_A_220-max"): 2.5})
    defaults = {"ln_A": (-20.0, 5.0)}

    result = inference.read_parameter_bounds(config, inference.configparser, "ln_A", "ln_A_220", defaults)

    assert result == [-1.5, 2.5]
    captured = capsys.readouterr()
    assert "ln_A_220" in captured.out


def test_read_parameter_bounds_falls_back_to_defaults():
    config = FakeConfig({})
    defaults = {"phi": (0.0, inference.twopi)}

    result = inference.read_parameter_bounds(config, inference.configparser, "phi", "phi_220", defaults)

    assert result == [0.0, inference.twopi]


def test_read_parameter_start_minimization_returns_config_value(capsys):
    config = FakeConfig({("Priors", "frequency-start"): 42.0})

    start = inference.read_parameter_start_minimization(config, inference.configparser, "frequency", [0.0, 1.0], 3)

    assert start == 42.0
    captured = capsys.readouterr()
    assert "frequency" in captured.out


def test_store_evidence_to_file_writes_expected_content(tmp_path):
    output_dir = tmp_path / "Algorithm"
    output_dir.mkdir(parents=True)

    parameters = {"I/O": {"outdir": str(tmp_path)}}
    inference.store_evidence_to_file(parameters, 12.34)

    evidence_path = output_dir / "Evidence.txt"
    content = evidence_path.read_text(encoding="utf-8")
    assert content.splitlines() == ["logZ", "12.34"]


def test_dynamic_inference_model_instances_are_pickleable_after_global_lookup_reset():
    model_class = inference.Dynamic_InferenceModel(PickleBase)
    instance = model_class.__new__(model_class)
    instance.payload = "value"

    payload = pickle.dumps(instance)
    model_class_name = model_class.__name__

    assert "<locals>" not in model_class.__qualname__
    delattr(inference, model_class_name)
    inference._DYNAMIC_INFERENCE_MODEL_CLASSES.pop(model_class_name, None)

    restored = pickle.loads(payload)

    assert type(restored).__name__ == model_class_name
    assert isinstance(restored, PickleBase)
    assert restored.payload == "value"


@pytest.mark.parametrize(
    "wf_model, template, expected",
    [
        (
            "Damped-sinusoids",
            "",
            {
                "ln_A": [-20.0, 5.0],
                "phi": [0.0, inference.twopi],
                "f": [-2.0 / inference.twopi, 2.0 / inference.twopi],
                "tau": [1, 50],
            },
        ),
        ("Kerr", "", {"ln_A": [-20.0, 5.0], "phi": [0.0, inference.twopi]}),
        (
            "Kerr-tail",
            "",
            {"ln_A_tail": [-20.0, 5.0], "phi_tail": [0.0, inference.twopi], "p_tail": [-20.0, 20.0]},
        ),
        ("KerrBinary", "", {"phi": [0.0, inference.twopi]}),
        (
            "TEOBPM",
            "qc",
            {
                "phi_mrg": [0.0, inference.twopi],
                "c3A": [-10.0, 10.0],
                "c3p": [-10.0, 10.0],
                "c4p": [-10.0, 10.0],
            },
        ),
    ],
)
def test_read_default_bounds_selects_expected_ranges(wf_model, template, expected):
    result = inference.read_default_bounds(wf_model, template)
    assert result == expected


def test_railing_check_invokes_pyRing_utils_and_saves_results(tmp_path, monkeypatch):
    output_dir = tmp_path / "Algorithm"
    output_dir.mkdir(parents=True)

    calls = []

    def fake_railing_check(samples, prior_bins, tolerance):
        calls.append((samples, list(prior_bins), tolerance))
        return False, False

    monkeypatch.setattr(inference.pyRing_utils, "railing_check", fake_railing_check)

    saved = {}

    def fake_savetxt(path, data, fmt="%s", header=""):
        saved[os.fspath(path)] = {"data": list(data), "header": header, "fmt": fmt}

    monkeypatch.setattr(inference.np, "savetxt", fake_savetxt)

    class FakeModel:
        names = ["param"]
        bounds = [[0.0, 1.0]]

    results = {"param": [0.1, 0.2, 0.3]}

    inference.railing_check(results, FakeModel(), str(tmp_path), nlive=4, seed=0)

    assert calls, "pyRing.utils.railing_check should be called"
    saved_path = os.path.join(str(tmp_path), "Algorithm", "Parameters_prior_railing.txt")
    assert saved_path in saved
    assert saved[saved_path]["header"].strip() == "param_low\tparam_up"


def test_minimization_constraint_residuals_penalize_damped_sinusoid_frequency_ordering(monkeypatch):
    monkeypatch.setattr(inference.pyRing_utils, "print_subsection", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(inference.pyRing_utils, "print_fixed_parameters", lambda *args, **kwargs: None, raising=False)

    model_class = inference.Dynamic_InferenceModel(PickleBase)
    waveform = FakeWaveform("Damped-sinusoids", N_ds_modes=2)
    model = model_class(
        np.array([0j, 0j, 0j]),
        np.array([1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j]),
        waveform,
        FakeConfig({}),
        "Minimization",
        "trf",
    )

    valid = {"f_0": 0.1, "f_1": 0.2}
    invalid = {"f_0": 0.2, "f_1": 0.1}

    assert len(model.minimization_constraint_residuals(valid)) == 0
    residuals = model.minimization_constraint_residuals(invalid)
    assert len(residuals) == 1
    assert residuals[0] > 0.0


def test_minimization_constraint_residuals_penalize_kerr_tail_exponent_ordering(monkeypatch):
    monkeypatch.setattr(inference.pyRing_utils, "print_subsection", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(inference.pyRing_utils, "print_fixed_parameters", lambda *args, **kwargs: None, raising=False)

    model_class = inference.Dynamic_InferenceModel(PickleBase)
    waveform = FakeWaveform(
        "Kerr",
        Kerr_modes=[(2, 2, 0)],
        tail=1,
        tail_modes=[(2, 2), (3, 2)],
    )
    model = model_class(
        np.array([0j, 0j, 0j]),
        np.array([1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j]),
        waveform,
        FakeConfig({}),
        "Minimization",
        "trf",
    )

    valid = {"p_tail_22": 1.0, "p_tail_32": 2.0}
    invalid = {"p_tail_22": 2.0, "p_tail_32": 1.0}

    assert len(model.minimization_constraint_residuals(valid)) == 0
    residuals = model.minimization_constraint_residuals(invalid)
    assert len(residuals) == 1
    assert residuals[0] > 0.0


def test_dynamic_inference_model_rejects_unknown_template():
    model_class = inference.Dynamic_InferenceModel(PickleBase)
    waveform = FakeWaveform("Unknown-template")

    with pytest.raises(ValueError, match="Unknown template selected"):
        model_class(
            np.array([0j, 0j, 0j]),
            np.array([1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j]),
            waveform,
            FakeConfig({}),
            "Minimization",
            "trf",
        )


def test_minimization_method_rejects_lm():
    minimization = inference.Minimization_Algorithm.__new__(inference.Minimization_Algorithm)

    with pytest.raises(ValueError, match="Available options"):
        minimization._least_squares_method("lm")
