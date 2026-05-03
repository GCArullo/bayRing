import math
import os
import sys
import tempfile
import types

os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "bayring-numba-cache"))

if "sxs" not in sys.modules:
    fake_sxs = types.ModuleType("sxs")
    fake_sxs.load = lambda *args, **kwargs: None
    sys.modules["sxs"] = fake_sxs

if "qnm" in sys.modules and not hasattr(sys.modules["qnm"], "download_data"):
    sys.modules["qnm"].download_data = lambda *args, **kwargs: None

from bayRing import NR_waveforms


def test_read_sxs_metadata_values_projects_precessing_spins():
    metadata = {
        "reference_mass_ratio": 1.6,
        "reference_dimensionless_spin1": [0.3, 0.4, -0.5],
        "reference_dimensionless_spin2": [0.0, 0.0, 0.25],
        "reference_orbital_frequency": [0.0, 0.0, 2.0],
        "reference_eccentricity": "<0.001",
        "remnant_mass": 0.95,
        "remnant_dimensionless_spin": [0.1, 0.2, 0.3],
    }

    q, chi1, chi2, tilt1, tilt2, ecc, Mf, chif = NR_waveforms._read_SXS_metadata_values(metadata)

    assert q == 1.6
    assert chi1 == -0.5
    assert chi2 == 0.25
    assert math.isclose(tilt1, math.acos(-0.5/math.sqrt(0.5)))
    assert tilt2 == 0.0
    assert ecc == 0.001
    assert Mf == 0.95
    assert math.isclose(chif, math.sqrt(0.14))


def test_read_sxs_metadata_values_uses_reference_axis_not_z_axis():
    metadata = {
        "reference_mass_ratio": 1.0,
        "reference_dimensionless_spin1": [0.3, 0.0, 0.4],
        "reference_dimensionless_spin2": [0.0, 0.0, 0.0],
        "reference_orbital_frequency": [2.0, 0.0, 0.0],
        "reference_eccentricity": 0.0,
        "remnant_mass": 0.96,
        "remnant_dimensionless_spin": [0.0, 0.0, 0.7],
    }

    _, chi1, chi2, tilt1, tilt2, _, _, chif = NR_waveforms._read_SXS_metadata_values(metadata)

    assert chi1 == 0.3
    assert chi2 == 0.0
    assert math.isclose(tilt1, math.acos(0.3/0.5))
    assert tilt2 == 0.0
    assert chif == 0.7
