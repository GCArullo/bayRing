import configparser
import math
import os
import sys

import pytest

from bayRing import bayRing, initialise


def test_parse_nr_mode_values_and_angle_ranges():
    assert initialise.parse_nr_mode_values("[(2, 2), (3, 3)]") == [(2, 2), (3, 3)]
    assert initialise.parse_nr_mode_values("22,33,4-4") == [(2, 2), (3, 3), (4, -4)]
    assert initialise.nr_mode_values_from_l_m([2, 3], [2, 3]) == [(2, 2), (3, 3)]
    values = initialise.parse_angle_values("0:pi:pi/4")
    assert len(values) == 5
    assert all(math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-11) for value, expected in zip(values, [0.0, math.pi/4.0, math.pi/2.0, 3.0*math.pi/4.0, math.pi]))


def test_read_config_records_nr_mode_scan_and_inclinations():
    config = configparser.ConfigParser()
    config.add_section("NR-data")
    config.set("NR-data", "NR-modes", "[(2, 2), (3, 3)]")
    config.add_section("Inference")
    config.set("Inference", "n-mode-workers", "2")
    config.add_section("Mismatch-GW-parameters")
    config.set("Mismatch-GW-parameters", "inclination", "0:pi/2:pi/4")

    parameters = initialise.read_config(config)

    assert parameters["NR-data"]["l-NR"] == 2
    assert parameters["NR-data"]["m"] == 2
    assert parameters["NR-data"]["NR-mode-list"] == [(2, 2), (3, 3)]
    assert parameters["Inference"]["n-mode-workers"] == 2
    assert len(parameters["Mismatch-GW-parameters"]["inclination-list"]) == 3
    assert all(math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-11) for value, expected in zip(parameters["Mismatch-GW-parameters"]["inclination-list"], [0.0, math.pi/4.0, math.pi/2.0]))


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
    config.set("Inference", "n-start-time-workers", "2")

    parameters = initialise.read_config(config)

    assert parameters["Inference"]["t-start"] == 20.0
    assert parameters["Inference"]["t-start-list"] == [20.0, 25.0, 30.0]
    assert parameters["Inference"]["n-start-time-workers"] == 2


def test_read_config_rejects_invalid_start_time_worker_count():
    config = configparser.ConfigParser()
    config.add_section("Inference")
    config.set("Inference", "n-start-time-workers", "0")

    with pytest.raises(ValueError, match="n-start-time-workers"):
        initialise.read_config(config)


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
        parallel_start_time=True,
    )

    assert base_parameters["Inference"]["t-start"] == 20.0
    assert run_parameters["Inference"]["t-start"] == 25.0
    assert run_parameters["I/O"]["outdir"] == os.path.join(os.fspath(tmp_path), "t_start_25M")
    assert run_parameters["I/O"]["start-time-output"] is True
    assert run_parameters["I/O"]["start-time-parallel"] is True
    assert run_parameters["I/O"]["start-time-index"] == 2


def test_prepare_start_time_parameters_routes_multi_mode_outputs(tmp_path):
    base_parameters = {
        "I/O": {"outdir": os.fspath(tmp_path)},
        "NR-data": {"l-NR": 2, "m": 2, "NR-mode-list": [(2, 2), (3, 3)]},
        "Inference": {"t-start": 20.0, "t-start-list": [20.0], "n-mode-workers": 2},
        "Model": {"template": "Kerr", "QNM-modes": "220"},
    }

    run_parameters = bayRing._prepare_start_time_parameters(
        base_parameters,
        os.fspath(tmp_path),
        20.0,
        1,
        1,
        parallel_start_time=True,
        nr_mode=(3, 3),
        mode_index=2,
        n_modes=2,
    )

    assert run_parameters["NR-data"]["l-NR"] == 3
    assert run_parameters["NR-data"]["m"] == 3
    assert run_parameters["I/O"]["outdir"] == os.path.join(os.fspath(tmp_path), "mode_l3_m3")
    assert run_parameters["I/O"]["mode-output"] is True
    assert run_parameters["I/O"]["mode-index"] == 2
    assert run_parameters["I/O"]["start-time-parallel"] is True


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
    assert run_parameters["I/O"]["start-time-parallel"] is False


def test_main_dispatches_multi_start_scan_to_parallel_runner(tmp_path, monkeypatch):
    config_path = tmp_path / "scan.ini"
    config_path.write_text(
        "[I/O]\noutdir = {}\n\n[Inference]\nt-start = 20:30:5\nn-start-time-workers = 2\n".format(tmp_path / "out"),
        encoding="utf-8",
    )
    captured = {}

    def fake_run_scan_jobs_parallel(config_file, run_parameters_list, start_time_workers):
        captured["config_file"] = config_file
        captured["run_parameters_list"] = run_parameters_list
        captured["start_time_workers"] = start_time_workers

    monkeypatch.setattr(sys, "argv", ["bayRing", "--config-file", os.fspath(config_path)])
    monkeypatch.setattr(initialise, "set_shared_output", lambda *args: None)
    monkeypatch.setattr(bayRing, "_run_scan_jobs_parallel", fake_run_scan_jobs_parallel)
    monkeypatch.setattr(bayRing, "_run_single_start", lambda *args: pytest.fail("serial start-time runner called"))

    bayRing.main()

    assert captured["config_file"] == os.fspath(config_path)
    assert captured["start_time_workers"] == 2
    assert [run["Inference"]["t-start"] for run in captured["run_parameters_list"]] == [20.0, 25.0, 30.0]
    assert all(run["I/O"]["start-time-parallel"] for run in captured["run_parameters_list"])


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
