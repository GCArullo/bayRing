import os

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
