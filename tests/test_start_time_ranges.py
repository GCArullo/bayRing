import configparser
import os

import pytest

from bayRing import bayRing, initialise


def test_parse_start_time_scalar_list_and_range():
    assert initialise.parse_start_time_values("30.0") == [30.0]
    assert initialise.parse_start_time_values("20, 25, 30") == [20.0, 25.0, 30.0]
    assert initialise.parse_start_time_values("[20, 25, 30]") == [20.0, 25.0, 30.0]
    assert initialise.parse_start_time_values("20:30:5") == [20.0, 25.0, 30.0]


def test_parse_start_time_range_rejects_bad_step():
    with pytest.raises(ValueError, match="step cannot be zero"):
        initialise.parse_start_time_values("20:30:0")

    with pytest.raises(ValueError, match="step sign"):
        initialise.parse_start_time_values("20:30:-5")


def test_read_config_keeps_active_start_and_records_scan_values():
    config = configparser.ConfigParser()
    config.add_section("Inference")
    config.set("Inference", "t-start", "20:30:5")

    parameters = initialise.read_config(config)

    assert parameters["Inference"]["t-start"] == 20.0
    assert parameters["Inference"]["t-start-list"] == [20.0, 25.0, 30.0]


def test_prepare_start_time_parameters_routes_multi_start_outputs(tmp_path):
    base_parameters = {
        "I/O": {"outdir": os.fspath(tmp_path)},
        "Inference": {"t-start": 20.0, "t-start-list": [20.0, 25.0]},
    }

    run_parameters = bayRing._prepare_start_time_parameters(
        base_parameters,
        os.fspath(tmp_path),
        25.0,
        2,
        2,
    )

    assert base_parameters["Inference"]["t-start"] == 20.0
    assert run_parameters["Inference"]["t-start"] == 25.0
    assert run_parameters["I/O"]["outdir"] == os.path.join(os.fspath(tmp_path), "t_start_25M")
    assert run_parameters["I/O"]["start-time-output"] is True
    assert run_parameters["I/O"]["start-time-index"] == 2


def test_prepare_start_time_parameters_preserves_scalar_outdir(tmp_path):
    base_parameters = {
        "I/O": {"outdir": os.fspath(tmp_path)},
        "Inference": {"t-start": 20.0, "t-start-list": [20.0]},
    }

    run_parameters = bayRing._prepare_start_time_parameters(
        base_parameters,
        os.fspath(tmp_path),
        20.0,
        1,
        1,
    )

    assert run_parameters["I/O"]["outdir"] == os.fspath(tmp_path)
    assert run_parameters["I/O"]["start-time-output"] is False


def test_multi_start_output_keeps_shared_files_in_base_outdir(tmp_path, monkeypatch):
    config_path = tmp_path / "scan.ini"
    config_path.write_text("[Inference]\nt-start = 20:30:5\n", encoding="utf-8")
    base_outdir = tmp_path / "out"
    sub_outdir = base_outdir / "t_start_20M"

    def fake_store_git_info(outdir):
        with open(os.path.join(outdir, "git_info.txt"), "w", encoding="utf-8") as outfile:
            outfile.write("git metadata")

    monkeypatch.setattr(initialise, "store_git_info", fake_store_git_info)
    monkeypatch.setattr(initialise, "_is_git_repository", lambda: True)

    initialise.set_shared_output(os.fspath(base_outdir), 1, os.fspath(config_path), "full")
    initialise.set_output(
        os.fspath(sub_outdir),
        1,
        "Nested-sampler",
        os.fspath(config_path),
        "full",
        shared_files=False,
        redirect_streams=False,
    )

    assert (base_outdir / "git_info.txt").exists()
    assert (base_outdir / "scan.ini").exists()
    assert (sub_outdir / "Algorithm").exists()
    assert (sub_outdir / "Peak_quantities").exists()
    assert (sub_outdir / "Plots" / "Results").exists()
    assert not (sub_outdir / "git_info.txt").exists()
    assert not (sub_outdir / "scan.ini").exists()


def test_store_git_info_skips_when_not_in_repository(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_store_git_info(outdir):
        calls.append(outdir)

    monkeypatch.setattr(initialise, "store_git_info", fake_store_git_info)
    monkeypatch.setattr(initialise, "_is_git_repository", lambda: False)

    initialise._store_git_info(os.fspath(tmp_path))

    assert calls == []
    assert not (tmp_path / "git_info.txt").exists()
    assert "not a git repository" in capsys.readouterr().out
