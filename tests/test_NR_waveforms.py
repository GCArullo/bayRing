import numpy as np

from bayRing import NR_waveforms


def test_sxs_christodoulou_scale_prefers_reference_masses():
    metadata = {
        "reference_mass1": 0.5002,
        "reference_mass2": 0.4999,
        "initial_mass1": 0.4,
        "initial_mass2": 0.3,
    }

    assert abs(NR_waveforms.SXS_christodoulou_mass_scale(metadata) - 1.0001) < 1e-12


def test_sxs_christodoulou_scale_falls_back_to_initial_masses():
    metadata = {
        "initial_mass1": "0.5003",
        "initial_mass2": "0.4998",
    }

    assert abs(NR_waveforms.SXS_christodoulou_mass_scale(metadata) - 1.0001) < 1e-12


def test_sxs_time_and_frequency_rescaling_are_inverse():
    sim = object.__new__(NR_waveforms.NR_simulation)
    sim.christodoulou_mass_scale = 1.0001

    np.testing.assert_allclose(sim._rescale_SXS_time(np.array([0.0, 10.0])), np.array([0.0, 10.001]))
    assert sim._rescale_SXS_time(10.0) == 10.001
    assert sim._rescale_SXS_frequency(0.2) == 0.2 / 1.0001
    assert sim._rescale_SXS_second_time_derivative(0.3) == 0.3 / (1.0001**2)


def test_read_sxs_metadata_exposes_time_scale(monkeypatch):
    monkeypatch.setattr(NR_waveforms.pyRing_utils, "m1_from_m_q", lambda M, q: 0.5, raising=False)
    monkeypatch.setattr(NR_waveforms.pyRing_utils, "m2_from_m_q", lambda M, q: 0.5, raising=False)

    class Sim:
        q = 1.0
        chi1 = 0.0
        chi2 = 0.0
        tilt1 = 0.0
        tilt2 = 0.0
        ecc = 0.0
        Mf = 0.95
        af = 0.7
        M_christodoulou_total = 1.0001
        christodoulou_mass_scale = 1.0001
        A_peak_22 = None
        omg_peak_22 = None
        A_nr_error = None
        A_peak22dotdot = None

    metadata = NR_waveforms.read_NR_metadata(Sim(), "SXS")

    assert metadata["M_christodoulou_total"] == 1.0001
    assert metadata["time_scale"] == 1.0001
