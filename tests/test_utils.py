import io
import tarfile

import numpy as np
import pytest

from bayRing import utils


def test_get_param_override_prefers_fixed_value():
    params = {"mass": 10}
    sample = {"mass": 5, "spin": 0.1}
    assert utils.get_param_override(params, sample, "mass") == 10


def test_get_param_override_returns_sample_when_not_fixed():
    params = {"mass": 10}
    sample = {"mass": 5, "spin": 0.1}
    assert utils.get_param_override(params, sample, "spin") == 0.1


@pytest.mark.parametrize("path", [None, "", "   "])
def test_normalize_optional_path_returns_none_for_missing_values(path):
    assert utils.normalize_optional_path(path) is None


def test_normalize_optional_path_strips_config_values():
    assert utils.normalize_optional_path("  metadata.csv  ") == "metadata.csv"


def test_filter_dict_by_key_extracts_per_category():
    data = {
        "linear": {"mass": [1, 2], "spin": [3, 4]},
        "quadratic": {"mass": [5, 6]},
    }

    filtered = utils.filter_dict_by_key(data, "mass")

    assert filtered == {
        "linear": {"mass": [1, 2]},
        "quadratic": {"mass": [5, 6]},
    }


def test_find_longest_name_length():
    names = ["a", "longer", "longest_name"]
    assert utils.find_longest_name_length(names) == len("longest_name")


def test_minimisation_compatibility_check_rejects_non_kerr():
    parameters = {"template": "non-kerr", "QQNM_modes": None, "tail": 0}
    with pytest.raises(ValueError):
        utils.minimisation_compatibility_check(parameters)


def test_minimisation_compatibility_check_rejects_qqnm_modes():
    parameters = {"template": "Kerr", "QQNM_modes": [1, 2], "tail": 0}
    with pytest.raises(ValueError):
        utils.minimisation_compatibility_check(parameters)


def test_minimisation_compatibility_check_rejects_tail():
    parameters = {"template": "Kerr", "QQNM_modes": None, "tail": 1}
    with pytest.raises(ValueError):
        utils.minimisation_compatibility_check(parameters)


def test_minimisation_compatibility_check_accepts_valid_configuration():
    parameters = {"template": "Kerr", "QQNM_modes": None, "tail": 0}
    utils.minimisation_compatibility_check(parameters)


def test_diff1_central_difference_with_padding():
    xp = np.array([0.0, 1.0, 2.0, 3.0])
    yp = np.array([value * value for value in xp])

    derivative = utils.diff1(xp, yp)

    np.testing.assert_allclose(derivative, np.array([2.0, 2.0, 4.0, 4.0]))


def test_read_psi4_rit_format(tmp_path):
    tar_path = tmp_path / "psi4.tar.gz"
    asc_name = "ExtrapPsi4_l2_m2.asc"

    header = "col:Time col:Real col:Imag"
    rows = [
        "0.0 1.0 0.5",
        "1.0 2.0 1.5",
    ]
    content = "\n".join([
        "# comment",
        "# comment",
        "# comment",
        header,
        *rows,
    ])

    with tarfile.open(tar_path, "w:gz") as tar:
        data = content.encode("utf-8")
        info = tarfile.TarInfo(name=f"some/path/{asc_name}")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    result = utils.read_psi4_RIT_format(str(tar_path), asc_name)

    assert set(result.keys()) == {"Time", "Real", "Imag"}
    np.testing.assert_allclose(result["Time"], np.array([0.0, 1.0]))
    np.testing.assert_allclose(result["Real"], np.array([1.0, 2.0]))
    np.testing.assert_allclose(result["Imag"], np.array([0.5, 1.5]))
