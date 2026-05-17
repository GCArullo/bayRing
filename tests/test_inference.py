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
        self.TEOB_template = "HypTan"
        self.TEOB_global_fit = 1
        self.TEOB_merger_data = 0
        self.tail = 0
        self.tail_modes = []
        self.quadratic_modes = None
        self.l_NR = 2
        self.m_NR = 2
        for key, value in overrides.items():
            setattr(self, key, value)

    def waveform(self, params, fixed_params):
        return np.array([0j, 0j, 0j])


class FakeLinearKerrWaveform:
    def __init__(self, basis, **overrides):
        self.wf_model = "Kerr"
        self.Kerr_modes = [key for key in basis.keys() if isinstance(key, tuple) and len(key) == 3 and isinstance(key[0], int)]
        self.tail = 0
        self.tail_modes = []
        self.quadratic_modes = None
        self.const_params = None
        self.basis = basis
        for key, value in overrides.items():
            setattr(self, key, value)

    def _basis(self, *keys):
        for key in keys:
            if key in self.basis:
                return self.basis[key]
        raise KeyError(keys)

    def kerr_waveform_from_components(self, amplitudes=None, tail_amplitudes=None, tail_exponents=None, quadratic_amplitudes=None, include_const=True):
        if amplitudes is None:
            amplitudes = {}
        if tail_amplitudes is None:
            tail_amplitudes = {}
        if tail_exponents is None:
            tail_exponents = {}
        if quadratic_amplitudes is None:
            quadratic_amplitudes = {}

        waveform = np.zeros(len(next(iter(self.basis.values()))), dtype=np.complex128)
        for mode, amplitude in amplitudes.items():
            waveform += amplitude * self._basis(mode, ("linear", mode))
        for key, amplitude in quadratic_amplitudes.items():
            waveform += amplitude * self._basis(key, ("quadratic", key))
        for mode, amplitude in tail_amplitudes.items():
            waveform += amplitude * self._basis(("tail", mode, tail_exponents[mode]), ("tail", mode))

        if include_const and self.const_params is not None:
            const_value = self.const_params[0] * np.cos(self.const_params[1])
            const_value = -const_value + 1j * self.const_params[0] * np.sin(self.const_params[1])
            waveform += const_value

        return waveform

    def kerr_waveform_from_complex_amplitudes(self, amplitudes, include_const=True):
        return self.kerr_waveform_from_components(amplitudes=amplitudes, include_const=include_const)


class FakeLinearInferenceModel:
    def __init__(self, waveform, data, error, names, fixed_params=None, kind="gaussian"):
        self.wf_model = waveform
        self.data = data
        self.error = error
        self.names = names
        self.fixed_params = fixed_params or {}
        self.kind = kind

    def access_names(self):
        return self.names

    def access_bounds(self):
        return [[-20.0, 5.0] if name.startswith("ln_A") else [0.0, inference.twopi] for name in self.names]


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


def test_linear_inversion_is_the_only_linear_inversion_method():
    assert inference.linear_inversion_methods == ["Linear-inversion"]
    assert inference.point_estimate_methods == ["Minimization", "Linear-inversion"]


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


def test_save_point_estimates_writes_auxiliary_summary(tmp_path):
    output_dir = tmp_path / "Algorithm"
    output_dir.mkdir(parents=True)

    results = {"ln_A_220": 1.2, "phi_220": 0.3}
    errors = {"ln_A_220": 0.04, "phi_220": 0.05}

    summary_path = inference.postprocess.save_point_estimates(results, str(tmp_path), errors=errors)

    summary_file = output_dir / "point_estimates.dat"
    assert summary_path == os.fspath(summary_file)
    assert summary_file.read_text(encoding="utf-8").splitlines() == [
        "# parameter\tvalue\tsigma",
        "ln_A_220\t1.2\t0.04",
        "phi_220\t0.3\t0.05",
    ]


def test_save_point_estimate_posterior_writes_gaussian_samples(tmp_path):
    if not hasattr(inference.np.random, "default_rng"):
        pytest.skip("requires real numpy random generator")

    output_dir = tmp_path / "Algorithm"
    output_dir.mkdir(parents=True)

    results = {"ln_A_220": 1.2, "phi_220": 0.3}
    errors = {"ln_A_220": 0.04, "phi_220": 0.05}
    covariance = np.array([[0.04**2, 0.0], [0.0, 0.05**2]])

    posterior_path = inference.postprocess.save_point_estimate_posterior(
        results,
        str(tmp_path),
        covariance=covariance,
        errors=errors,
        seed=1234,
        n_samples=64,
    )

    posterior_file = output_dir / "posterior.dat"
    assert posterior_path == os.fspath(posterior_file)
    assert posterior_file.exists()
    assert not (output_dir / "Minimization_Results.txt").exists()

    posterior = np.genfromtxt(posterior_file, names=True, deletechars="")
    assert posterior.dtype.names == ("ln_A_220", "phi_220")
    assert posterior.shape == (64,)
    assert np.std(posterior["ln_A_220"]) > 0.0
    assert np.std(posterior["phi_220"]) > 0.0
    assert np.mean(posterior["ln_A_220"]) == pytest.approx(1.2, abs=0.03)
    assert np.mean(posterior["phi_220"]) == pytest.approx(0.3, abs=0.03)


def test_save_point_estimate_posterior_zero_samples_skips_file(tmp_path):
    output_dir = tmp_path / "Algorithm"
    output_dir.mkdir(parents=True)

    posterior_path = inference.postprocess.save_point_estimate_posterior(
        {"ln_A_220": 1.2},
        str(tmp_path),
        errors={"ln_A_220": 0.04},
        n_samples=0,
    )

    assert posterior_path is None
    assert not (output_dir / "posterior.dat").exists()


def test_point_estimate_parameter_samples_use_one_sigma_errors():
    results = inference.postprocess.PointEstimateResults(
        {"ln_A_220": 1.2, "phi_220": 0.3},
        errors={"ln_A_220": 0.04, "phi_220": float("nan")},
    )

    samples = inference.postprocess.waveform_parameter_samples(results, "Minimization")

    assert samples[0] == {"ln_A_220": 1.2, "phi_220": 0.3}
    assert abs(samples[1]["ln_A_220"] - 1.16) < 1e-12
    assert samples[1]["phi_220"] == 0.3
    assert abs(samples[2]["ln_A_220"] - 1.24) < 1e-12
    assert samples[2]["phi_220"] == 0.3


def test_read_point_estimate_postprocessing_uses_posterior_file(tmp_path):
    if not hasattr(inference.np, "genfromtxt"):
        pytest.skip("requires real numpy genfromtxt")

    output_dir = tmp_path / "Algorithm"
    output_dir.mkdir(parents=True)
    (output_dir / "posterior.dat").write_text(
        "# ln_A_220\tphi_220\n1.2\t0.3\n1.3\t0.4\n",
        encoding="utf-8",
    )
    parameters = {"I/O": {"outdir": str(tmp_path)}, "Inference": {"method": "Minimization"}}

    results = inference.postprocess.read_results_object_from_previous_inference(parameters)

    assert results.dtype.names == ("ln_A_220", "phi_220")
    assert results["ln_A_220"][0] == pytest.approx(1.2)
    assert results["phi_220"][1] == pytest.approx(0.4)


def test_read_point_estimate_postprocessing_prefers_point_file_when_zero_samples(tmp_path):
    output_dir = tmp_path / "Algorithm"
    output_dir.mkdir(parents=True)
    (output_dir / "point_estimates.dat").write_text(
        "# parameter\tvalue\tsigma\nln_A_220\t1.2\t0.04\n",
        encoding="utf-8",
    )
    (output_dir / "posterior.dat").write_text(
        "# ln_A_220\n9.0\n",
        encoding="utf-8",
    )
    parameters = {
        "I/O": {"outdir": str(tmp_path)},
        "Inference": {"method": "Minimization", "point-estimate-posterior-samples": 0},
    }

    results = inference.postprocess.read_results_object_from_previous_inference(parameters)

    assert isinstance(results, inference.postprocess.PointEstimateResults)
    assert abs(results["ln_A_220"] - 1.2) < 1e-12
    assert abs(results.errors["ln_A_220"] - 0.04) < 1e-12


def test_read_point_estimate_postprocessing_falls_back_to_point_estimates(tmp_path):
    output_dir = tmp_path / "Algorithm"
    output_dir.mkdir(parents=True)
    (output_dir / "point_estimates.dat").write_text(
        "# parameter\tvalue\tsigma\nln_A_220\t1.2\t0.04\nphi_220\t0.3\t0.05\n",
        encoding="utf-8",
    )
    parameters = {"I/O": {"outdir": str(tmp_path)}, "Inference": {"method": "Minimization"}}

    results = inference.postprocess.read_results_object_from_previous_inference(parameters)

    assert results == {"ln_A_220": 1.2, "phi_220": 0.3}
    assert results.errors == {"ln_A_220": 0.04, "phi_220": 0.05}


def test_point_estimate_inference_defaults_to_point_results(monkeypatch, tmp_path):
    calls = []

    class FakeModel:
        names = ["x"]

    class FakeMinimization:
        errors = {"x": 0.1}
        covariance = "minimization-covariance"

        def __init__(self, inference_model, parameters):
            pass

        def minimize_likelihood(self):
            return [1.0]

    class FakeLinearInversion:
        errors = {"x": 0.2}
        covariance = "linear-covariance"

        def __init__(self, inference_model, parameters):
            pass

        def solve_likelihood(self):
            return [2.0]

    def fake_save_point_estimates(results, outdir, errors=None):
        calls.append(("summary", dict(results), outdir, dict(errors)))

    def fake_save_point_estimate_posterior(*args, **kwargs):
        raise AssertionError("default point-estimate inference should not write posterior samples")

    monkeypatch.setattr(inference, "Minimization_Algorithm", FakeMinimization)
    monkeypatch.setattr(inference, "KerrLinearInversion_Algorithm", FakeLinearInversion)
    monkeypatch.setattr(inference.postprocess, "save_point_estimates", fake_save_point_estimates)
    monkeypatch.setattr(inference.postprocess, "save_point_estimate_posterior", fake_save_point_estimate_posterior)
    monkeypatch.setattr(inference.postprocess, "read_posterior_samples", lambda outdir: (_ for _ in ()).throw(AssertionError("posterior should not be read")))
    monkeypatch.setattr(inference, "minimization_railing_check", lambda results, model, outdir, tolerance=2.0: calls.append(("railing", dict(results), outdir, tolerance)))

    results = []
    for method in ["Minimization", "Linear-inversion"]:
        parameters = {"I/O": {"outdir": str(tmp_path)}, "Inference": {"method": method, "seed": 1234}}
        results.append(inference.run_inference(parameters, FakeModel()))

    assert [dict(result) for result in results] == [{"x": 1.0}, {"x": 2.0}]
    assert results[0].errors == {"x": 0.1}
    assert results[1].errors == {"x": 0.2}

    assert calls == [
        ("summary", {"x": 1.0}, str(tmp_path), {"x": 0.1}),
        ("railing", {"x": 1.0}, str(tmp_path), 2.0),
        ("summary", {"x": 2.0}, str(tmp_path), {"x": 0.2}),
    ]


def test_point_estimate_inference_optionally_writes_posterior(monkeypatch, tmp_path):
    calls = []

    class FakeModel:
        names = ["x"]

    class FakeMinimization:
        errors = {"x": 0.1}
        covariance = "minimization-covariance"

        def __init__(self, inference_model, parameters):
            pass

        def minimize_likelihood(self):
            return [1.0]

    def fake_save_point_estimates(results, outdir, errors=None):
        calls.append(("summary", dict(results), outdir, dict(errors)))

    def fake_save_point_estimate_posterior(results, outdir, covariance=None, errors=None, seed=None, n_samples=0):
        calls.append(("posterior", dict(results), outdir, covariance, dict(errors), seed, n_samples))

    monkeypatch.setattr(inference, "Minimization_Algorithm", FakeMinimization)
    monkeypatch.setattr(inference.postprocess, "save_point_estimates", fake_save_point_estimates)
    monkeypatch.setattr(inference.postprocess, "save_point_estimate_posterior", fake_save_point_estimate_posterior)
    monkeypatch.setattr(inference.postprocess, "read_posterior_samples", lambda outdir: {"posterior": outdir})
    monkeypatch.setattr(inference, "minimization_railing_check", lambda results, model, outdir, tolerance=2.0: calls.append(("railing", results, outdir, tolerance)))

    parameters = {
        "I/O": {"outdir": str(tmp_path)},
        "Inference": {"method": "Minimization", "seed": 1234, "point-estimate-posterior-samples": 64},
    }

    assert inference.run_inference(parameters, FakeModel()) == {"posterior": str(tmp_path)}
    assert calls == [
        ("summary", {"x": 1.0}, str(tmp_path), {"x": 0.1}),
        ("posterior", {"x": 1.0}, str(tmp_path), "minimization-covariance", {"x": 0.1}, 1234, 64),
        ("railing", {"posterior": str(tmp_path)}, str(tmp_path), 2.0),
    ]


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
            "HypTan",
            {
                "phi_mrg": [0.0, inference.twopi],
                "c3A": [-10.0, 10.0],
                "c3p": [-10.0, 10.0],
                "c4p": [-10.0, 10.0],
            },
        ),
        (
            "TEOBPM",
            "RatExp",
            {
                "phi_mrg": [0.0, inference.twopi],
                "c3A": [-10.0, 10.0],
                "c3p": [-10.0, 10.0],
                "c4p": [-10.0, 10.0],
                "c2A": [-10.0, 10.0],
                "c2p": [-10.0, 10.0],
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


def test_point_estimate_railing_check_uses_prior_edge_tolerance(tmp_path, monkeypatch):
    output_dir = tmp_path / "Algorithm"
    output_dir.mkdir(parents=True)

    saved = {}

    def fake_savetxt(path, data, fmt="%s", header=""):
        saved_data = data.tolist() if hasattr(data, "tolist") else list(data)
        saved[os.fspath(path)] = {"data": saved_data, "header": header, "fmt": fmt}

    monkeypatch.setattr(inference.np, "savetxt", fake_savetxt)

    class FakeModel:
        names = ["near_low", "near_high", "middle"]
        bounds = [[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]]

    results = {"near_low": 0.1, "near_high": 9.9, "middle": 5.0}

    inference.point_estimate_railing_check(results, FakeModel(), str(tmp_path), tolerance=2.0)

    saved_path = os.path.join(str(tmp_path), "Algorithm", "Parameters_prior_railing.txt")
    assert saved[saved_path]["header"].strip() == "near_low_low\tnear_low_up\tnear_high_low\tnear_high_up\tmiddle_low\tmiddle_up"
    assert saved[saved_path]["data"] in ([[1, 0, 0, 1, 0, 0]], [1, 0, 0, 1, 0, 0])


def test_minimization_railing_check_uses_posterior_check_for_samples(monkeypatch, tmp_path):
    calls = []

    class FakeModel:
        names = ["param"]

    posterior = {"param": np.array([0.1, 0.2, 0.3])}

    def fake_railing_check(results_object, inference_model, outdir, nlive=None, seed=None, tolerance=2.0, check_chains=True):
        calls.append((results_object, outdir, tolerance, check_chains))

    monkeypatch.setattr(inference, "railing_check", fake_railing_check)

    inference.minimization_railing_check(posterior, FakeModel(), str(tmp_path), tolerance=3.0)

    assert calls == [(posterior, str(tmp_path), 3.0, False)]


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


def test_least_squares_parameter_errors_use_weighted_fisher_inverse():
    if not hasattr(inference.np, "linalg"):
        pytest.skip("requires real numpy linear algebra")

    class Result:
        jac = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])

    errors, covariance, _, _ = inference.estimate_least_squares_parameter_errors(["x", "y"], Result())

    assert covariance[0, 0] == pytest.approx(1.0)
    assert covariance[1, 1] == pytest.approx(0.25)
    assert errors["x"] == pytest.approx(1.0)
    assert errors["y"] == pytest.approx(0.5)


def test_kerr_linear_inversion_reports_log_amplitude_and_phase_errors():
    if not hasattr(inference.np, "linalg"):
        pytest.skip("requires real numpy linear algebra")

    mode = (2, 2, 0)
    waveform = FakeLinearKerrWaveform({mode: np.array([1.0 + 0.0j])})
    model = FakeLinearInferenceModel(
        waveform,
        np.array([2.0 + 0.0j]),
        np.array([1.0 + 1.0j]),
        ["ln_A_220", "phi_220"],
    )
    parameters = {"Inference": {"linear-inversion-eigenvalue-tol": 1e-12}}

    algorithm = inference.KerrLinearInversion_Algorithm(model, parameters)
    solution = algorithm.solve_likelihood()

    assert solution[0] == pytest.approx(np.log(2.0))
    assert solution[1] == pytest.approx(0.0)
    assert algorithm.errors["ln_A_220"] == pytest.approx(0.5)
    assert algorithm.errors["phi_220"] == pytest.approx(0.5)


def test_kerr_linear_inversion_recovers_two_complex_mode_amplitudes():
    if not hasattr(inference.np, "linalg"):
        pytest.skip("requires real numpy linear algebra")

    modes = [(2, 2, 0), (2, 2, 1)]
    basis = {
        modes[0]: np.array([1.0 + 0.2j, 0.4 + 1.1j, -0.3 + 0.7j, 0.5 - 0.8j]),
        modes[1]: np.array([0.7 - 0.4j, -1.2 + 0.3j, 0.2 + 1.4j, 0.9 + 0.6j]),
    }
    true_amplitudes = {
        modes[0]: 1.25 * np.exp(1j * 0.4),
        modes[1]: 0.70 * np.exp(1j * 2.1),
    }
    waveform = FakeLinearKerrWaveform(basis)
    data = sum(true_amplitudes[mode] * basis[mode] for mode in modes)
    error = np.array([1.0 + 1.5j, 2.0 + 2.5j, 1.2 + 1.8j, 0.8 + 1.1j])
    names = ["ln_A_220", "phi_220", "ln_A_221", "phi_221"]
    model = FakeLinearInferenceModel(waveform, data, error, names)
    parameters = {"Inference": {"linear-inversion-eigenvalue-tol": 1e-12}}

    solution = inference.KerrLinearInversion_Algorithm(model, parameters).solve_likelihood()
    recovered = dict(zip(names, solution))

    for mode, label in zip(modes, ["220", "221"]):
        complex_amplitude = np.exp(recovered[f"ln_A_{label}"]) * np.exp(1j * recovered[f"phi_{label}"])
        assert complex_amplitude == pytest.approx(true_amplitudes[mode])


def test_kerr_linear_inversion_rejects_partly_fixed_polar_amplitude():
    mode = (2, 2, 0)
    waveform = FakeLinearKerrWaveform({mode: np.array([1.0 + 0.0j, 0.0 + 1.0j])})
    model = FakeLinearInferenceModel(
        waveform,
        np.array([0.0 + 0.0j, 0.0 + 0.0j]),
        np.array([1.0 + 1.0j, 1.0 + 1.0j]),
        ["ln_A_220"],
        fixed_params={"phi_220": 0.0},
    )
    parameters = {"Inference": {"linear-inversion-eigenvalue-tol": 1e-12}}

    with pytest.raises(ValueError, match="both `ln_A_220` and `phi_220`"):
        inference.KerrLinearInversion_Algorithm(model, parameters)


def test_kerr_linear_inversion_recovers_quadratic_and_fixed_exponent_tail_amplitudes():
    if not hasattr(inference.np, "linalg"):
        pytest.skip("requires real numpy linear algebra")

    linear_mode = (2, 2, 0)
    quad_modes = ((4, 4, 0), (2, 2, 0), (2, 2, 0))
    quad_key = ("sum", quad_modes)
    tail_mode = (2, 2)
    basis = {
        linear_mode: np.array([1.0 + 0.2j, 0.4 + 1.1j, -0.3 + 0.7j, 0.5 - 0.8j, 1.2 + 0.4j, -0.7 + 0.5j]),
        ("quadratic", quad_key): np.array([0.3 + 0.6j, -0.2 + 1.0j, 0.8 - 0.4j, -1.1 + 0.2j, 0.1 + 0.9j, 0.5 - 0.6j]),
        ("tail", tail_mode, -2.0): np.array([1.1 - 0.2j, 0.6 + 0.3j, -0.4 + 0.8j, 0.9 + 0.7j, -0.5 + 1.2j, 0.2 - 1.0j]),
    }
    true_amplitudes = {
        "linear": 1.10 * np.exp(1j * 0.25),
        "quadratic": 0.35 * np.exp(1j * 1.2),
        "tail": 0.80 * np.exp(1j * 2.4),
    }
    waveform = FakeLinearKerrWaveform(
        basis,
        Kerr_modes=[linear_mode],
        quadratic_modes={"sum": [quad_modes], "diff": []},
        tail=1,
        tail_modes=[tail_mode],
    )
    data = (
        true_amplitudes["linear"] * basis[linear_mode]
        + true_amplitudes["quadratic"] * basis[("quadratic", quad_key)]
        + true_amplitudes["tail"] * basis[("tail", tail_mode, -2.0)]
    )
    error = np.array([1.0 + 1.1j, 1.2 + 1.3j, 1.4 + 1.5j, 1.6 + 1.7j, 1.8 + 1.9j, 2.0 + 2.1j])
    names = [
        "ln_A_220",
        "phi_220",
        "ln_A_sum_440_220_220",
        "phi_sum_440_220_220",
        "ln_A_tail_22",
        "phi_tail_22",
    ]
    model = FakeLinearInferenceModel(waveform, data, error, names, fixed_params={"p_tail_22": -2.0})
    parameters = {"Inference": {"linear-inversion-eigenvalue-tol": 1e-12}}

    solution = inference.KerrLinearInversion_Algorithm(model, parameters).solve_likelihood()
    recovered = dict(zip(names, solution))

    recovered_amplitudes = {
        "linear": np.exp(recovered["ln_A_220"]) * np.exp(1j * recovered["phi_220"]),
        "quadratic": np.exp(recovered["ln_A_sum_440_220_220"]) * np.exp(1j * recovered["phi_sum_440_220_220"]),
        "tail": np.exp(recovered["ln_A_tail_22"]) * np.exp(1j * recovered["phi_tail_22"]),
    }
    for key in true_amplitudes:
        assert recovered_amplitudes[key] == pytest.approx(true_amplitudes[key])


def test_kerr_linear_inversion_rejects_free_tail_exponents():
    mode = (2, 2, 0)
    waveform = FakeLinearKerrWaveform(
        {mode: np.array([1.0 + 0.0j, 0.0 + 1.0j]), ("tail", (2, 2)): np.array([0.5 + 0.0j, 0.0 + 0.5j])},
        tail=1,
        tail_modes=[(2, 2)],
    )
    model = FakeLinearInferenceModel(
        waveform,
        np.array([0.0 + 0.0j, 0.0 + 0.0j]),
        np.array([1.0 + 1.0j, 1.0 + 1.0j]),
        ["ln_A_220", "phi_220", "ln_A_tail_22", "phi_tail_22", "p_tail_22"],
    )
    parameters = {"Inference": {"linear-inversion-eigenvalue-tol": 1e-12}}

    with pytest.raises(ValueError, match="fixed `p_tail_22`"):
        inference.KerrLinearInversion_Algorithm(model, parameters)
