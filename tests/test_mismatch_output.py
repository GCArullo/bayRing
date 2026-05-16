import csv

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
