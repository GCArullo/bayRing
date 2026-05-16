import math
import configparser
import sys
import types
from pathlib import Path

import pytest

from bayRing import injection


def _base_parameters(**overrides):
    parameters = {
        "t_start": 0.0,
        "t_end": 120.0,
        "dt": 0.1,
        "q": 2.0,
        "Mf": 0.95,
        "af": 0.7,
    }
    parameters.update(overrides)
    return parameters


def test_split_injection_parameters_converts_legacy_kerr_amplitudes():
    times, metadata, waveform_parameters = injection.split_injection_parameters(
        _base_parameters(
            A_220=2.5,
            phi_220=0.3,
            A_22_tail=0.2,
            phi_22_tail=1.1,
            p_22_tail=-2.0,
        )
    )

    assert times == {"t_start": 0.0, "t_end": 120.0, "dt": 0.1}
    assert math.isclose(metadata["m1"], 2.0/3.0)
    assert math.isclose(metadata["m2"], 1.0/3.0)
    assert metadata["chi1"] == 0.0
    assert math.isclose(waveform_parameters["ln_A_220"], math.log(2.5))
    assert math.isclose(waveform_parameters["phi_220"], 0.3)
    assert math.isclose(waveform_parameters["ln_A_tail_22"], math.log(0.2))
    assert math.isclose(waveform_parameters["phi_tail_22"], 1.1)
    assert math.isclose(waveform_parameters["p_tail_22"], -2.0)


def test_split_injection_parameters_preserves_generic_template_parameters():
    _, metadata, waveform_parameters = injection.split_injection_parameters(
        _base_parameters(
            chi1=0.2,
            chi2=-0.1,
            A_peak_22=0.4,
            omg_peak_22=0.3,
            A_peak22dotdot=-0.01,
            ln_A_0=-3.0,
            phi_0=0.4,
            f_0=0.08,
            tau_0=12.0,
            phi=1.2,
            phi_mrg_22=0.7,
            c3A_22=1.1,
        )
    )

    assert math.isclose(metadata["chi1"], 0.2)
    assert math.isclose(metadata["A_peak_22"], 0.4)
    assert math.isclose(metadata["omg_peak_22"], 0.3)
    assert math.isclose(metadata["A_peak22dotdot"], -0.01)
    assert "A_peak_22" not in waveform_parameters
    assert math.isclose(waveform_parameters["ln_A_0"], -3.0)
    assert math.isclose(waveform_parameters["phi"], 1.2)
    assert math.isclose(waveform_parameters["phi_mrg_22"], 0.7)
    assert math.isclose(waveform_parameters["c3A_22"], 1.1)


def test_split_injection_parameters_rejects_missing_common_metadata():
    with pytest.raises(ValueError, match="Missing mandatory injection parameters"):
        injection.split_injection_parameters({"t_start": 0.0})


def test_legacy_fake_nr_catalog_name_is_accepted():
    assert injection.is_injection_catalog("injections")
    assert injection.is_injection_catalog("fake_NR")


def test_injection_example_configs_cover_available_templates():
    fake_pyRing_initialise = types.ModuleType("pyRing.initialise")
    fake_pyRing_initialise.store_git_info = lambda *args, **kwargs: None
    sys.modules.setdefault("pyRing.initialise", fake_pyRing_initialise)

    from bayRing import initialise

    initialise.pyRing_utils.print_subsection = lambda *args, **kwargs: None

    repo_root = Path(__file__).resolve().parents[1]
    config_paths = sorted((repo_root / "config_files").glob("config_injections_*_quick.ini"))

    templates = []
    for path in config_paths:
        config = configparser.ConfigParser()
        config.read(path)
        parameters = initialise.read_config(config)
        assert parameters["NR-data"]["catalog"] == "injections"
        assert parameters["Injection-data"]["parameters"] is not None
        templates.append(parameters["Model"]["template"])

    assert set(templates) == {
        "Damped-sinusoids",
        "Kerr",
        "Kerr-Damped-sinusoids",
        "KerrBinary",
        "TEOBPM",
    }
