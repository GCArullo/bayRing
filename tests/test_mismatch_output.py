import csv

import numpy as np
import pytest

from bayRing import postprocess


def test_mismatch_plot_dir_creates_only_requested_smoothing_dir(tmp_path):
    save_path = postprocess._mismatch_plot_dir(str(tmp_path), "above")

    mismatch_dir = tmp_path / "Algorithm" / "Mismatch"
    assert save_path == str(mismatch_dir / "Right_smoothing")
    assert not (mismatch_dir / "Left_smoothing").exists()
    assert (mismatch_dir / "Right_smoothing").exists()
    assert not (mismatch_dir / "Both_edges_smoothing").exists()


def test_mismatch_subfolder_rejects_invalid_direction():
    with pytest.raises(ValueError, match="Invalid mismatch smoothing direction"):
        postprocess._mismatch_subfolder("sideways")


def test_record_mismatch_diagnostic_merges_mismatch_and_snr(tmp_path):
    postprocess._record_mismatch_diagnostic(
        str(tmp_path), "strain_components", 60.0, 410.0, -8.1, 869565,
        0.8, 0.8, 7.0, 1.0, 1.0, direction="below-and-above",
        confidence_interval=50, strain_data="real", mismatch=0.02
    )
    postprocess._record_mismatch_diagnostic(
        str(tmp_path), "strain_components", 60.0, 410.0, -8.1, 869565,
        0.8, 0.8, 7.0, 1.0, 1.0, direction="below-and-above",
        confidence_interval=50, strain_data="real", optimal_snr=100.0
    )

    mismatch_dir = tmp_path / "Algorithm" / "Mismatch"
    diagnostics_text = (mismatch_dir / "mismatch_and_snr_diagnostics.tsv").read_text()
    parameters_text = (mismatch_dir / "mismatch_and_snr_diagnostic_parameters.tsv").read_text()
    diagnostics = list(csv.DictReader(diagnostics_text.splitlines(), delimiter="\t"))
    parameters = list(csv.DictReader(parameters_text.splitlines(), delimiter="\t"))

    assert len(diagnostics) == 1
    assert len(parameters) == 1
    assert diagnostics[0]["mismatch"] == "0.02"
    assert diagnostics[0]["optimal_snr"] == "100"
    assert parameters[0]["remnant_mass_solar_masses"] == "60"
    assert "M_60_dL_410" not in diagnostics_text


def test_compute_nr_comparison_mismatches_records_available_sxs_pairs(tmp_path):
    class FakeNRSimulation:
        NR_catalog = "SXS"
        extrap_order = 2
        res_level = 6
        t_NR_cut = np.array([0.0, 1.0])

        def read_waveform_lm_from_SXS(self, extrap_order, res_level):
            real_waveforms = {
                (2, 6): np.array([1.0, 0.0]),
                (2, 5): np.array([0.0, 1.0]),
                (3, 6): np.array([1.0, 1.0]),
            }
            imag_waveforms = {
                (2, 6): np.array([0.0, 1.0]),
                (2, 5): np.array([1.0, 0.0]),
                (3, 6): np.array([1.0, 1.0]),
            }
            try:
                real = real_waveforms[(extrap_order, res_level)]
                imag = imag_waveforms[(extrap_order, res_level)]
            except KeyError:
                raise ValueError("waveform unavailable")
            return np.array([0.0, 1.0]), real, imag

    postprocess.compute_nr_comparison_mismatches(
        FakeNRSimulation(),
        str(tmp_path),
        acf=np.array([1.0, 0.0]),
        N_FFT=2,
        M=1.0,
        dL=1.0,
        t_start_g=0.0,
        window_size_DX=0.8,
        window_size_SX=0.8,
        k=7.0,
        saturation_DX=1.0,
        saturation_SX=1.0,
        direction="below-and-above",
    )

    result_path = tmp_path / "Algorithm" / "Mismatch" / "mismatch_and_snr_diagnostics.tsv"
    rows = list(csv.DictReader(result_path.read_text().splitlines(), delimiter="\t"))

    row_by_type_and_component = {
        (row["diagnostic_type"], row["strain_data"]): row
        for row in rows
    }

    assert row_by_type_and_component[("nr_resolution_Lev6_vs_Lev5", "real")]["mismatch"] == "1"
    assert row_by_type_and_component[("nr_resolution_Lev6_vs_Lev5", "imag")]["mismatch"] == "1"
    assert float(row_by_type_and_component[("nr_extrapolation_N2_vs_N3", "real")]["mismatch"]) == pytest.approx(1.0 - 1.0 / np.sqrt(2.0))
