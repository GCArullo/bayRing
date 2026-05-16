import csv
import json
import types
from pathlib import Path

import pytest

from scripts.teobpm import teobpm_calibration as calib


def _catalog_rows():
    return [
        {
            "sxs_id": "SXS:BBH:0001",
            "q": 1.0,
            "reference_dimensionless_spin1": [0.0, 0.0, 0.0],
            "reference_dimensionless_spin2": [0.0, 0.0, 0.0],
            "reference_eccentricity": 1.0e-4,
            "number_of_orbits": 20,
            "resolution_level": 5,
        },
        {
            "sxs_id": "SXS:BBH:0002",
            "q": 2.0,
            "reference_dimensionless_spin1": [0.0, 0.0, 0.2],
            "reference_dimensionless_spin2": [0.0, 0.0, -0.1],
            "reference_eccentricity": 2.0e-4,
            "number_of_orbits": 18,
            "resolution_level": 4,
        },
        {
            "sxs_id": "SXS:BBH:0003",
            "q": 3.0,
            "reference_dimensionless_spin1": [0.02, 0.0, 0.0],
            "reference_dimensionless_spin2": [0.0, 0.0, 0.0],
            "reference_eccentricity": 1.0e-4,
            "number_of_orbits": 18,
            "resolution_level": 4,
        },
        {
            "sxs_id": "SXS:BBH:0004",
            "q": 4.0,
            "reference_dimensionless_spin1": [0.0, 0.0, 0.0],
            "reference_dimensionless_spin2": [0.0, 0.0, 0.0],
            "reference_eccentricity": 5.0e-2,
            "number_of_orbits": 18,
            "resolution_level": 4,
        },
    ]


def test_filter_records_selects_nonspinning_nonprecessing_subset(tmp_path):
    records = [calib.normalise_catalog_entry(row) for row in _catalog_rows()]
    config = calib.CalibrationConfig(family="nonspinning", output_dir=str(tmp_path))

    selected = calib.filter_records(records, config)

    assert [record.sxs_id for record in selected] == ["SXS:BBH:0001"]


def test_aligned_spin_requires_nonspinning_base_fit(tmp_path):
    records = [calib.normalise_catalog_entry(row) for row in _catalog_rows()]
    config = calib.CalibrationConfig(family="aligned-spin", output_dir=str(tmp_path))

    with pytest.raises(ValueError, match="nonspinning"):
        calib.filter_records(records, config)


def test_prepare_campaign_writes_manifests_and_configs(tmp_path, monkeypatch):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(_catalog_rows()), encoding="utf-8")
    output_dir = tmp_path / "campaign"
    monkeypatch.setattr(calib, "_pyRing_teobpm_delta_t", lambda record, mode: 0.0 if mode == (2, 2) else 4.25)
    config = calib.CalibrationConfig(
        family="nonspinning",
        output_dir=str(output_dir),
        modes=[(2, 2), (3, 3)],
        validation_fraction=0.25,
    )

    summary = calib.prepare_campaign(str(catalog_path), config)

    assert summary["selected_records"] == 1
    assert (output_dir / "manifests" / "training_manifest.json").exists()
    run_script = output_dir / "run_local_fits.sh"
    assert run_script.exists()
    assert run_script.stat().st_mode & 0o100
    assert summary["local_fit_configs"] == 2
    assert summary["indexed_local_fit_jobs"] == 2

    expected_22 = output_dir / "local_fit_configs" / "SXS_BBH_0001_mode_22.ini"
    expected_33 = output_dir / "local_fit_configs" / "SXS_BBH_0001_mode_33.ini"
    assert run_script.read_text(encoding="utf-8").splitlines() == [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"bayRing --config-file {expected_22}",
        f"bayRing --config-file {expected_33}",
    ]
    jobs = calib.read_local_fit_index(output_dir / "local_fit_index.csv")
    assert [(job.sxs_id, job.mode, job.split) for job in jobs] == [
        ("SXS:BBH:0001", "22", "training"),
        ("SXS:BBH:0001", "33", "training"),
    ]
    assert jobs[0].config_file == str(expected_22)
    assert jobs[0].outdir == str(output_dir / "local_fits" / "SXS_BBH_0001" / "mode_22")
    expected_22_text = expected_22.read_text(encoding="utf-8")
    expected_33_text = expected_33.read_text(encoding="utf-8")
    assert "t-start = 0.0" in expected_22_text
    assert "t-peak-22 = 0.0" in expected_33_text
    assert "t-start = 4.25" in expected_33_text
    assert "[Priors]" in expected_33_text
    assert "c3p_33-min = 0.1" in expected_33_text


def test_prepare_campaign_allows_zero_validation_fraction(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(_catalog_rows()), encoding="utf-8")
    output_dir = tmp_path / "campaign"
    config = calib.CalibrationConfig(
        family="nonspinning",
        output_dir=str(output_dir),
        modes=[(2, 2)],
        validation_fraction=0.0,
    )

    summary = calib.prepare_campaign(str(catalog_path), config)

    assert summary["training_records"] == 1
    assert summary["validation_records"] == 0
    assert json.loads((output_dir / "manifests" / "validation_manifest.json").read_text(encoding="utf-8")) == []


def test_prepare_parser_defaults_to_mixed_higher_modes_and_sxs_id_subset(tmp_path):
    args = calib.build_parser().parse_args([
        "prepare",
        "--family",
        "nonspinning",
        "--output-dir",
        str(tmp_path / "campaign"),
        "--sxs-ids",
        "1,SXS_BBH_0003",
    ])

    config = calib._config_from_args(args)

    assert config.mode_mixing_modes == [(3, 2), (4, 3)]
    assert config.sxs_ids == ["SXS:BBH:0001", "SXS:BBH:0003"]
    assert config.random_fraction == 1.0
    assert config.t_start == 0.0


def test_run_local_fit_job_invokes_bayring_and_writes_logs(tmp_path, monkeypatch):
    config_file = tmp_path / "fit.ini"
    config_file.write_text("[I/O]\n", encoding="utf-8")
    job = calib.LocalFitJob(
        sxs_id="SXS:BBH:0001",
        mode="22",
        split="training",
        config_file=str(config_file),
        outdir=str(tmp_path / "out"),
        q=1.0,
        nu=0.25,
        chi1z=0.0,
        chi2z=0.0,
        chi_eff=0.0,
        chi_a=0.0,
        eccentricity=0.0,
        quality_score=1.0,
    )
    calls = []

    def fake_run(command, cwd, stdout, stderr, text, timeout):
        calls.append({
            "command": command,
            "cwd": cwd,
            "stdout": stdout.name,
            "stderr": stderr.name,
            "text": text,
            "timeout": timeout,
        })
        stdout.write("stdout text\n")
        stderr.write("stderr text\n")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(calib.subprocess, "run", fake_run)

    summary = calib.run_local_fit_job(job, bayring_executable="python -m bayRing.bayRing", timeout=12.5)

    assert summary["status"] == "completed"
    assert calls[0]["command"] == ["python", "-m", "bayRing.bayRing", "--config-file", str(config_file)]
    assert calls[0]["stdout"].endswith("campaign_logs/stdout.txt")
    assert calls[0]["stderr"].endswith("campaign_logs/stderr.txt")
    assert calls[0]["text"] is True
    assert calls[0]["timeout"] == 12.5
    assert (tmp_path / "out" / "campaign_logs" / "stdout.txt").read_text(encoding="utf-8") == "stdout text\n"
    assert (tmp_path / "out" / "campaign_logs" / "stderr.txt").read_text(encoding="utf-8") == "stderr text\n"


def test_run_local_fit_job_skips_existing_sampler_posterior(tmp_path, monkeypatch):
    config_file = tmp_path / "fit.ini"
    config_file.write_text("[Inference]\nmethod = Nested-sampler\n", encoding="utf-8")
    outdir = tmp_path / "out"
    (outdir / "Algorithm").mkdir(parents=True)
    posterior_path = outdir / "Algorithm" / "posterior.dat"
    posterior_path.write_text("c3A_22\tlogL\tlogPrior\n1.0\t-2.0\t0.0\n", encoding="utf-8")
    job = calib.LocalFitJob(
        sxs_id="SXS:BBH:0001",
        mode="22",
        split="training",
        config_file=str(config_file),
        outdir=str(outdir),
        q=1.0,
        nu=0.25,
        chi1z=0.0,
        chi2z=0.0,
        chi_eff=0.0,
        chi_a=0.0,
        eccentricity=0.0,
        quality_score=1.0,
    )

    def fail_run(*args, **kwargs):
        raise AssertionError("existing sampler posterior should skip execution")

    monkeypatch.setattr(calib.subprocess, "run", fail_run)

    summary = calib.run_local_fit_job(job)

    assert summary["status"] == "skipped-existing"
    assert summary["completion_file"] == str(posterior_path)


def test_run_local_fits_filters_requested_split(tmp_path, monkeypatch):
    campaign_dir = tmp_path / "campaign"
    jobs = []
    for split, mode in [("training", "22"), ("validation", "33")]:
        jobs.append(
            calib.LocalFitJob(
                sxs_id="SXS:BBH:0001",
                mode=mode,
                split=split,
                config_file=str(campaign_dir / f"{mode}.ini"),
                outdir=str(campaign_dir / f"out_{mode}"),
                q=1.0,
                nu=0.25,
                chi1z=0.0,
                chi2z=0.0,
                chi_eff=0.0,
                chi_a=0.0,
                eccentricity=0.0,
                quality_score=1.0,
            )
        )
    calib.write_local_fit_index(jobs, campaign_dir / "local_fit_index.csv")
    seen = []

    def fake_run_job(job, bayring_executable="bayRing", force=False, timeout=None):
        seen.append((job.mode, bayring_executable, force, timeout))
        return {
            "sxs_id": job.sxs_id,
            "mode": job.mode,
            "config_file": job.config_file,
            "outdir": job.outdir,
            "returncode": 0,
            "status": "completed",
        }

    monkeypatch.setattr(calib, "run_local_fit_job", fake_run_job)

    summary = calib.run_local_fits(campaign_dir, bayring_executable="bayRing", split="validation", modes=[(3, 3)], timeout=3.0)

    assert summary["jobs"] == 1
    assert summary["split"] == "validation"
    assert summary["modes"] == ["33"]
    assert seen == [("33", "bayRing", False, 3.0)]


def test_fill_higher_mode_inputs_adds_t_peak_and_parent_fixes(tmp_path):
    campaign_dir = tmp_path / "campaign"
    config_dir = campaign_dir / "local_fit_configs"
    config_dir.mkdir(parents=True)
    parent_config = config_dir / "SXS_BBH_0001_mode_22.ini"
    child_config = config_dir / "SXS_BBH_0001_mode_32.ini"
    parent_config.write_text("[NR-data]\nt-peak-22 = 0.0\n", encoding="utf-8")
    child_config.write_text("[NR-data]\nt-peak-22 = 0.0\n\n[Model]\nTEOB-mode-mixing = 1\n", encoding="utf-8")
    parent_out = campaign_dir / "local_fits" / "SXS_BBH_0001" / "mode_22"
    child_out = campaign_dir / "local_fits" / "SXS_BBH_0001" / "mode_32"
    (parent_out / "Algorithm").mkdir(parents=True)
    (parent_out / "Peak_quantities").mkdir(parents=True)
    child_out.mkdir(parents=True)
    (parent_out / "Peak_quantities" / "Peak_time.txt").write_text("# t_peak\n123.5\n", encoding="utf-8")
    (parent_out / "Algorithm" / "point_estimates.dat").write_text(
        "# parameter value sigma\n"
        "phi_mrg_22 1.2 0.1\n"
        "c3A_22 -0.4 0.1\n"
        "c3p_22 4.0 0.1\n"
        "c4p_22 3.0 0.1\n",
        encoding="utf-8",
    )
    jobs = [
        calib.LocalFitJob("SXS:BBH:0001", "22", "training", str(parent_config), str(parent_out), 1.0, 0.25, 0, 0, 0, 0, 0, 1),
        calib.LocalFitJob("SXS:BBH:0001", "32", "training", str(child_config), str(child_out), 1.0, 0.25, 0, 0, 0, 0, 0, 1),
    ]
    calib.write_local_fit_index(jobs, campaign_dir / "local_fit_index.csv")

    summary = calib.fill_higher_mode_inputs(campaign_dir, mode_mixing_modes=[(3, 2)])

    assert summary["configs_updated"] == 1
    text = child_config.read_text(encoding="utf-8")
    assert "t-peak-22 = 123.5" in text
    assert "fix-phi_mrg_22 = 1.2" in text
    assert "fix-c3A_22 = -0.40000000000000002" in text


def test_collect_local_fit_outputs_reads_estimates_mismatches_and_relative_phases(tmp_path):
    campaign_dir = tmp_path / "campaign"
    manifest_dir = campaign_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    record = calib.normalise_catalog_entry({
        "sxs_id": "SXS:BBH:0001",
        "q": 1.0,
        "reference_dimensionless_spin1": [0.0, 0.0, 0.0],
        "reference_dimensionless_spin2": [0.0, 0.0, 0.0],
        "A_peak_22": 0.05,
        "omg_peak_22": 0.35,
    })
    record.split = "training"
    calib.write_records([record], manifest_dir / "dataset_manifest.json")

    jobs = []
    for mode in ["22", "33"]:
        outdir = campaign_dir / "local_fits" / "SXS_BBH_0001" / f"mode_{mode}"
        jobs.append(
            calib.LocalFitJob(
                sxs_id="SXS:BBH:0001",
                mode=mode,
                split="training",
                config_file=str(campaign_dir / "local_fit_configs" / f"SXS_BBH_0001_mode_{mode}.ini"),
                outdir=str(outdir),
                q=1.0,
                nu=0.25,
                chi1z=0.0,
                chi2z=0.0,
                chi_eff=0.0,
                chi_a=0.0,
                eccentricity=0.0,
                quality_score=1.0,
            )
        )
        (outdir / "Algorithm").mkdir(parents=True)
    calib.write_local_fit_index(jobs, campaign_dir / "local_fit_index.csv")
    (Path(jobs[0].outdir) / "Algorithm" / "point_estimates.dat").write_text(
        "# parameter\tvalue\tsigma\n"
        "c3A_22\t1.0\t0.1\n"
        "c3p_22\t2.0\t0.2\n"
        "phi_mrg_22\t0.5\t0.05\n",
        encoding="utf-8",
    )
    (Path(jobs[1].outdir) / "Algorithm" / "point_estimates.dat").write_text(
        "# parameter\tvalue\tsigma\n"
        "c3A_33\t1.4\t0.1\n"
        "phi_mrg_33\t1.0\t0.05\n",
        encoding="utf-8",
    )
    mismatch_dir = Path(jobs[1].outdir) / "Algorithm" / "Mismatch"
    mismatch_dir.mkdir()
    (mismatch_dir / "Mismatch_test.txt").write_text(
        "#CI\tStrain_data\tMismatch\n"
        "50\treal\t0.02\n"
        "50\timag\t0.04\n",
        encoding="utf-8",
    )
    (mismatch_dir / "Optimal_SNR_test.txt").write_text(
        "#CI\tStrain_data\tOptimal_SNR\n"
        "50\treal\t100.0\n"
        "50\timag\t100.0\n",
        encoding="utf-8",
    )

    summary = calib.collect_local_fit_outputs(campaign_dir)

    assert summary["jobs"] == 2
    assert summary["mismatch_rows"] == 2
    rows = list(csv.DictReader((campaign_dir / "local_fit_summary.csv").open(encoding="utf-8")))
    c3a_22 = [row for row in rows if row["mode"] == "22" and row["target"] == "c3A"][0]
    delta_33 = [row for row in rows if row["mode"] == "33" and row["target"] == "delta_phi_33"][0]
    metadata_peak = [row for row in rows if row["mode"] == "22" and row["target"] == "A_peak_over_nu"][0]
    assert c3a_22["source"] == "point_estimate"
    assert abs(float(delta_33["value"]) - 0.5) < 1.0e-12
    assert "mismatch" not in delta_33
    assert abs(float(delta_33["construction_mismatch"]) - (0.02**2 + 0.04**2) ** 0.5) < 1.0e-12
    assert metadata_peak["source"] == "metadata"
    assert abs(float(metadata_peak["value"]) - 0.2) < 1.0e-12
    assert (campaign_dir / "mismatch_summary.csv").exists()
    assert (campaign_dir / "local_fit_collection_summary.json").exists()


def test_collect_local_fit_outputs_reads_sampler_posterior_when_point_estimate_missing(tmp_path):
    campaign_dir = tmp_path / "campaign"
    outdir = campaign_dir / "local_fits" / "SXS_BBH_0001" / "mode_22"
    (outdir / "Algorithm").mkdir(parents=True)
    (outdir / "Algorithm" / "posterior.dat").write_text(
        "c3A_22\tphi_mrg_22\tlogL\tlogPrior\n"
        "1.0\t0.1\t-5.0\t0.0\n"
        "2.0\t0.2\t-4.0\t0.0\n"
        "3.0\t0.3\t-3.0\t0.0\n",
        encoding="utf-8",
    )
    job = calib.LocalFitJob(
        sxs_id="SXS:BBH:0001",
        mode="22",
        split="training",
        config_file=str(campaign_dir / "fit.ini"),
        outdir=str(outdir),
        q=1.0,
        nu=0.25,
        chi1z=0.0,
        chi2z=0.0,
        chi_eff=0.0,
        chi_a=0.0,
        eccentricity=0.0,
        quality_score=1.0,
    )
    calib.write_local_fit_index([job], campaign_dir / "local_fit_index.csv")

    summary = calib.collect_local_fit_outputs(campaign_dir)

    rows = list(csv.DictReader((campaign_dir / "local_fit_summary.csv").open(encoding="utf-8")))
    c3a_rows = [row for row in rows if row["mode"] == "22" and row["target"] == "c3A"]
    assert summary["failures"] == 0
    assert len(c3a_rows) == 1
    assert c3a_rows[0]["source"] == "posterior_median"
    assert abs(float(c3a_rows[0]["value"]) - 2.0) < 1.0e-12
    assert float(c3a_rows[0]["sigma"]) > 0.0


def test_construct_global_fit_uses_nonspinning_base_for_aligned_spin(tmp_path):
    base_fit = {
        "schema": calib.FIT_SCHEMA,
        "family": "nonspinning",
        "template": "RatExp",
        "fits": {
            "22": {
                "c3A": {
                    "terms": [
                        {"name": "1", "coefficient": 1.0},
                        {"name": "nu", "coefficient": 2.0},
                    ]
                }
            }
        },
    }
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base_fit), encoding="utf-8")
    table_path = tmp_path / "local.csv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "mode", "target", "nu", "chi_eff", "chi_a", "value"])
        writer.writeheader()
        for chi_eff in [-0.2, -0.1, 0.1, 0.2]:
            nu = 0.24
            writer.writerow({
                "split": "training",
                "mode": "22",
                "target": "c3A",
                "nu": nu,
                "chi_eff": chi_eff,
                "chi_a": 0.0,
                "value": 1.0 + 2.0 * nu + 0.5 * chi_eff,
            })

    output_path = tmp_path / "aligned.json"
    payload = calib.construct_global_fit(
        table_path,
        output_path,
        "aligned-spin",
        base_nonspinning_file=str(base_path),
    )

    terms = payload["fits"]["22"]["c3A"]["terms"]
    assert payload["calibration"] == "qc"
    assert terms[0]["name"] == "1"
    assert any(term["name"] == "chi_eff" for term in terms)
    assert output_path.exists()


def test_construct_global_fit_respects_max_polynomial_degree(tmp_path):
    table_path = tmp_path / "local.csv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "mode", "target", "nu", "chi_eff", "chi_a", "value"])
        writer.writeheader()
        for nu in [0.16, 0.19, 0.22, 0.25]:
            writer.writerow({
                "split": "training",
                "mode": "22",
                "target": "c3A",
                "nu": nu,
                "chi_eff": 0.0,
                "chi_a": 0.0,
                "value": 1.0 + 2.0 * nu + 4.0 * nu * nu,
            })

    linear_payload = calib.construct_global_fit(
        table_path,
        tmp_path / "linear.json",
        "nonspinning",
        max_polynomial_degree=1,
    )
    quadratic_payload = calib.construct_global_fit(
        table_path,
        tmp_path / "quadratic.json",
        "nonspinning",
        max_polynomial_degree=2,
    )

    linear_terms = [term["name"] for term in linear_payload["fits"]["22"]["c3A"]["terms"]]
    quadratic_terms = [term["name"] for term in quadratic_payload["fits"]["22"]["c3A"]["terms"]]
    assert "nu^2" not in linear_terms
    assert "nu^2" in quadratic_terms
    assert quadratic_payload["max_polynomial_degree"] == 2


def test_prediction_rows_use_validation_split_when_available(tmp_path):
    table_path = tmp_path / "local_fit_summary.csv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "mode", "target", "q", "nu", "chi_eff", "chi_a", "value"])
        writer.writeheader()
        writer.writerow({"split": "training", "mode": "22", "target": "c3A", "q": 1.0, "nu": 0.25, "chi_eff": 0.0, "chi_a": 0.0, "value": 10.0})
        writer.writerow({"split": "validation", "mode": "22", "target": "c3A", "q": 2.0, "nu": 2.0 / 9.0, "chi_eff": 0.0, "chi_a": 0.0, "value": 1.25})
    fit_payload = {
        "fits": {
            "22": {
                "c3A": {
                    "terms": [
                        {"name": "1", "coefficient": 1.0},
                    ],
                }
            }
        }
    }

    predictions = calib._prediction_rows(table_path, fit_payload)

    assert len(predictions) == 1
    assert predictions[0]["split"] == "validation"
    assert abs(predictions[0]["residual"] - 0.25) < 1.0e-12


def test_appendix_a_prepare_writes_paper_subset_config(tmp_path, monkeypatch):
    nc_dir = tmp_path / "nc_ringdown"
    data_dir = nc_dir / "src" / "data"
    data_dir.mkdir(parents=True)
    parameter_file = data_dir / "SXS_Parameters.csv"
    with parameter_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "q", "nu", "chi1z", "chi2z", "chieff", "ecc"])
        writer.writeheader()
        writer.writerow({
            "ID": "0305",
            "q": "0.8188649427812099",
            "nu": "0.24752061408783976",
            "chi1z": "0.329976580046",
            "chi2z": "-0.43994832522",
            "chieff": "-0.016648668848199894",
            "ecc": "0.0008382",
        })
    monkeypatch.setattr(calib, "APPENDIX_A_EQUAL_MASS_IDS", ("0305",))

    summary = calib.prepare_appendix_a_campaign(
        tmp_path / "campaign",
        nc_ringdown_dir=nc_dir,
        group="equal-mass",
        template="HypTan",
        n_random_seeds=4,
    )

    assert summary["jobs"] == 1
    manifest_rows = list(csv.DictReader((tmp_path / "campaign" / "appendix_a_manifest.csv").open(encoding="utf-8")))
    assert manifest_rows[0]["x_name"] == "S_hat"
    config_text = (tmp_path / "campaign" / "local_fit_configs" / "hyptan_equal_mass" / "SXS_0305.ini").read_text(encoding="utf-8")
    assert "TEOB-template     = HypTan" in config_text
    assert f"properties-file   = {parameter_file}" in config_text


def test_appendix_a_uses_paper_degrees_and_past_hyptan_values():
    assert calib._appendix_a_degree("equal-mass", "HypTan", "c3p") == 4
    assert calib._appendix_a_degree("equal-mass", "HypTan", "c3A") == 3
    assert calib._appendix_a_degree("equal-mass", "RatExp", "c2A") == 3
    assert calib._appendix_a_degree("nonspinning", "RatExp", "c4p") == 1

    assert abs(calib._appendix_a_past_hyptan_value("equal-mass", "c3A", 0.0) + 0.366538) < 1.0e-12
    expected_c3a_nu = -1.5 + 0.75497 * 0.25 - 0.56187
    assert abs(calib._appendix_a_past_hyptan_value("nonspinning", "c3A", 0.25) - expected_c3a_nu) < 1.0e-12


def _write_rao_fixture(nc_dir: Path, order: int = 1) -> Path:
    fit_dir = nc_dir / "src" / "data" / "fits" / "nc_fits_sxs_non-spinning"
    fit_dir.mkdir(parents=True)
    path = fit_dir / f"order_fits_nu_{order}.csv"
    fieldnames = ["fit_type", "fit_order"]
    for column_base in calib.APPENDIX_A_RAO_TARGET_COLUMNS.values():
        for index in range(order + 1):
            fieldnames.append(f"{column_base}_p{index}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        row = {"fit_type": "nu", "fit_order": str(order)}
        for offset, column_base in enumerate(calib.APPENDIX_A_RAO_TARGET_COLUMNS.values()):
            row[f"{column_base}_p0"] = str(1.0 + offset)
            row[f"{column_base}_p1"] = str(0.1 + offset)
        writer.writerow(row)
    return path


def _write_current_rao_fixture(nc_dir: Path) -> Path:
    fit_dir = nc_dir / "src" / "data" / "fits" / "nc_fits_sxs_non-spinning"
    fit_dir.mkdir(parents=True)
    path = fit_dir / "order_fits_nu_1.csv"
    fieldnames = ["fit_type", "fit_order"]
    for column_base in calib.APPENDIX_A_RAO_TARGET_COLUMNS.values():
        for index in range(2):
            fieldnames.append(f"{column_base}_p{index}")
    fieldnames.extend(["norm_nu_scale", "norm_nu_shift"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        row = {
            "fit_type": "nu",
            "fit_order": "1",
            "norm_nu_scale": "1",
            "norm_nu_shift": "0",
        }
        for column_base in calib.APPENDIX_A_RAO_TARGET_COLUMNS.values():
            row[f"{column_base}_p0"] = "1"
            row[f"{column_base}_p1"] = "2"
        writer.writerow(row)
    return path


def _write_noncircular_rao_fixture(nc_dir: Path) -> Path:
    fit_dir = nc_dir / "src" / "data" / "bayesian_fit_coefficients"
    fit_dir.mkdir(parents=True)
    path = fit_dir / "order_fits_nu_emrg_bmrg_3.csv"
    fieldnames = ["fit_type", "fit_order"]
    for column_base in calib.APPENDIX_A_RAO_TARGET_COLUMNS.values():
        for index in range(4):
            fieldnames.append(f"{column_base}_p{index}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        row = {"fit_type": "nu_emrg_bmrg", "fit_order": "1"}
        for column_base in calib.APPENDIX_A_RAO_TARGET_COLUMNS.values():
            row[f"{column_base}_p0"] = "1"
            row[f"{column_base}_p1"] = "2"
            row[f"{column_base}_p2"] = "3"
            row[f"{column_base}_p3"] = "4"
        writer.writerow(row)
    return path


def test_appendix_a_loads_rao_nonspinning_fits(tmp_path):
    _write_rao_fixture(tmp_path, order=1)

    fits = calib._appendix_a_load_rao_nu_fits(tmp_path, order=1)

    assert fits["c2A"]["coefficients_high_to_low"] == [1.0, 0.1]
    assert fits["c4p"]["coefficients_high_to_low"] == [5.0, 4.1]
    assert abs(calib._appendix_a_rao_value(fits, "c2A", 0.25) - 0.35) < 1.0e-12


def test_appendix_a_loads_current_noneccentric_rao_fit(tmp_path):
    _write_current_rao_fixture(tmp_path)

    fits = calib._appendix_a_load_current_rao_fits(tmp_path)
    value = calib._appendix_a_current_rao_value(
        fits["c2A"],
        {"nu": "0.2"},
    )

    assert fits["c2A"]["reference_type"] == "current-nc"
    assert abs(value - (1.0 + 2.0 * 0.2)) < 1.0e-12


def test_appendix_a_rejects_eccentric_current_rao_fit(tmp_path):
    path = _write_noncircular_rao_fixture(tmp_path)
    with pytest.raises(ValueError, match="noncircular"):
        calib._appendix_a_load_current_rao_fits(tmp_path, fit_file=path)

    path = _write_current_rao_fixture(tmp_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys())
    rows[0]["fit_type"] = "nu_ecc"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="non-eccentric"):
        calib._appendix_a_load_current_rao_fits(tmp_path)


def test_appendix_a_compare_nonspinning_writes_reference_tables(tmp_path, monkeypatch):
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    table = campaign_dir / "appendix_a_local_fit_summary.csv"
    fieldnames = ["group", "template", "target", "value", "x_value", "sigma", "sxs_id", "mismatch"]
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for template in ["HypTan", "RatExp"]:
            for target_index, target in enumerate(calib._appendix_a_targets(template)):
                for point_index, nu in enumerate([0.2, 0.25]):
                    writer.writerow({
                        "group": "nonspinning",
                        "template": template,
                        "target": target,
                        "value": 1.0 + target_index + 0.5 * point_index,
                        "x_value": nu,
                        "sigma": "0.01",
                        "sxs_id": f"SXS:BBH:00{point_index}",
                        "mismatch": "1e-5",
                    })
    nc_dir = tmp_path / "nc_ringdown"
    _write_rao_fixture(nc_dir, order=1)

    def fake_template_plot(rows, fits, rao_fits, template, output_dir):
        path = output_dir / f"{template}.png"
        path.write_text("", encoding="utf-8")
        return path

    def fake_summary_plot(summary_rows, output_dir):
        path = output_dir / "summary.png"
        path.write_text("", encoding="utf-8")
        return path

    monkeypatch.setattr(calib, "_appendix_a_compare_template_plot", fake_template_plot)
    monkeypatch.setattr(calib, "_appendix_a_compare_summary_plot", fake_summary_plot)

    summary = calib.compare_appendix_a_nonspinning_fits(campaign_dir, nc_ringdown_dir=nc_dir, rao_reference="appendix-nu")

    assert Path(summary["point_table"]).exists()
    assert Path(summary["grid_table"]).exists()
    summary_rows = list(csv.DictReader(Path(summary["summary_table"]).open(encoding="utf-8")))
    assert {(row["template"], row["target"]) for row in summary_rows} >= {("HypTan", "c3A"), ("RatExp", "c2A")}
    assert summary["rao_fit_file"].endswith("order_fits_nu_1.csv")


def test_appendix_a_prepare_global_mismatch_uses_minimization_and_noneccentric_reference(tmp_path):
    source_campaign = tmp_path / "appendix_a"
    source_campaign.mkdir()
    table = source_campaign / "appendix_a_local_fit_summary.csv"
    fieldnames = [
        "group",
        "template",
        "target",
        "value",
        "x_name",
        "x_value",
        "sxs_id",
        "sxs_number",
        "q",
        "nu",
        "chi1z",
        "chi2z",
        "chi_eff",
        "chi_a",
        "eccentricity",
        "properties_file",
    ]
    properties_file = tmp_path / "SXS_Parameters.csv"
    properties_file.write_text("", encoding="utf-8")
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for target in calib._appendix_a_targets("RatExp"):
            writer.writerow({
                "group": "nonspinning",
                "template": "RatExp",
                "target": target,
                "value": "1.0",
                "x_name": "nu",
                "x_value": "0.25",
                "sxs_id": "SXS:BBH:0180",
                "sxs_number": "0180",
                "q": "1.0",
                "nu": "0.25",
                "chi1z": "0.0",
                "chi2z": "0.0",
                "chi_eff": "0.0",
                "chi_a": "0.0",
                "eccentricity": "0.0",
                "properties_file": str(properties_file),
            })
    fit_payload = {"schema": calib.APPENDIX_A_SCHEMA, "fits": {"nonspinning": {"RatExp": {}}}}
    for target in calib._appendix_a_targets("RatExp"):
        fit_payload["fits"]["nonspinning"]["RatExp"][target] = {
            "x_name": "nu",
            "degree": 1,
            "coefficients": [{"power": 0, "coefficient": 1.0}, {"power": 1, "coefficient": 2.0}],
        }
    (source_campaign / "appendix_a_global_fits.json").write_text(json.dumps(fit_payload), encoding="utf-8")
    nc_dir = tmp_path / "nc_ringdown"
    rao_file = _write_current_rao_fixture(nc_dir)

    summary = calib.prepare_appendix_a_global_mismatch_campaign(
        source_campaign,
        output_dir=tmp_path / "global_mismatch",
        nc_ringdown_dir=nc_dir,
        current_rao_fit_file=rao_file,
    )

    assert summary["jobs"] == 2
    existing_config = (tmp_path / "global_mismatch" / "local_fit_configs" / "existing_teobpm" / "ratexp_nonspinning" / "SXS_0180.ini").read_text(encoding="utf-8")
    new_config = (tmp_path / "global_mismatch" / "local_fit_configs" / "new_global" / "ratexp_nonspinning" / "SXS_0180.ini").read_text(encoding="utf-8")
    combined = (existing_config + "\n" + new_config).lower()
    assert "method            = minimization" in combined
    assert "teob-calibration  = qc" in combined
    assert "nested" not in combined
    assert "order_fits_nu_1.csv" in existing_config
    assert all(token not in combined for token in ["nu_emrg", "bmrg", "jmrg", "ecc"])


def test_plot_teobpm_mismatch_comparison_writes_standard_parameter_space_views(tmp_path, monkeypatch):
    table = tmp_path / "mismatch_comparison.csv"
    fieldnames = [
        "sxs_id",
        "family",
        "template",
        "nu",
        "chi_eff",
        "new_global_mismatch",
        "existing_teobpm_mismatch",
    ]
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "sxs_id": "SXS:BBH:0001",
            "family": "nonspinning",
            "template": "HypTan",
            "nu": "0.25",
            "chi_eff": "0.0",
            "new_global_mismatch": "1e-5",
            "existing_teobpm_mismatch": "2e-5",
        })
        writer.writerow({
            "sxs_id": "SXS:BBH:0002",
            "family": "spinning",
            "template": "HypTan",
            "nu": "0.22",
            "chi_eff": "0.3",
            "new_global_mismatch": "3e-5",
            "existing_teobpm_mismatch": "4e-5",
        })

    def fake_plot(rows, output_dir, prefix="teobpm_mismatch_comparison", title_prefix="TEOBPM mismatch comparison", family="all"):
        output_dir = Path(output_dir)
        nonspinning = output_dir / f"{prefix}_nonspinning.png"
        spinning = output_dir / f"{prefix}_spinning.png"
        nonspinning.write_text("", encoding="utf-8")
        spinning.write_text("", encoding="utf-8")
        return [nonspinning, spinning]

    monkeypatch.setattr(calib, "_plot_standard_mismatch_parameter_space", fake_plot)

    summary = calib.plot_teobpm_mismatch_comparison([table], tmp_path / "plots")

    point_rows = list(csv.DictReader(Path(summary["point_table"]).open(encoding="utf-8")))
    assert {(row["family"], row["label"]) for row in point_rows} == {
        ("nonspinning", "Existing TEOBPM global fit HypTan"),
        ("nonspinning", "New global fit HypTan"),
        ("spinning", "Existing TEOBPM global fit HypTan"),
        ("spinning", "New global fit HypTan"),
    }
    assert any(path.endswith("teobpm_mismatch_comparison_nonspinning.png") for path in summary["plots"])
    assert any(path.endswith("teobpm_mismatch_comparison_spinning.png") for path in summary["plots"])
    assert Path(summary["summary_table"]).exists()


def test_plot_teobpm_mismatch_comparison_rejects_nonzero_evaluation_start(tmp_path):
    table = tmp_path / "mismatch_comparison.csv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sxs_id", "family", "nu", "q", "chi_eff", "mismatch", "t_start", "tref"])
        writer.writeheader()
        writer.writerow({
            "sxs_id": "SXS:BBH:0001",
            "family": "spinning",
            "nu": "0.25",
            "q": "1.0",
            "chi_eff": "0.0",
            "mismatch": "1e-5",
            "t_start": "4.25",
            "tref": "peak22",
        })

    with pytest.raises(ValueError, match="t_start=4.25"):
        calib.plot_teobpm_mismatch_comparison([table], tmp_path / "plots")


def test_plot_local_fit_diagnostics_uses_explicit_evaluation_mismatch_table(tmp_path, monkeypatch):
    local_table = tmp_path / "local_fit_summary.csv"
    with local_table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "mode", "target", "nu", "chi_eff", "value", "construction_mismatch"])
        writer.writeheader()
        writer.writerow({
            "split": "training",
            "mode": "33",
            "target": "c3A",
            "nu": "0.25",
            "chi_eff": "0.0",
            "value": "",
            "construction_mismatch": "9e-5",
        })
    evaluation_table = tmp_path / "local_fit_evaluation_mismatch.csv"
    evaluation_table.write_text(
        "sxs_id,family,nu,q,chi_eff,mismatch,t_start,tref\n"
        "SXS:BBH:0001,nonspinning,0.25,1.0,0.0,1e-5,0,peak22\n",
        encoding="utf-8",
    )
    calls = []

    def fake_mismatch_plot(tables, output_dir, labels=None, family="auto"):
        calls.append((tables, output_dir, labels, family))
        return {"plots": [str(Path(output_dir) / "fake.png")], "rows": 1}

    monkeypatch.setattr(calib, "plot_teobpm_mismatch_comparison", fake_mismatch_plot)

    summary_without_mismatch = calib.plot_local_fit_diagnostics(local_table, tmp_path / "no_mismatch")
    assert summary_without_mismatch["mismatch_summary"] == {}
    assert calls == []

    summary = calib.plot_local_fit_diagnostics(
        local_table,
        tmp_path / "with_mismatch",
        mismatch_tables=[evaluation_table],
        mismatch_labels=["Local evaluation"],
    )

    assert summary["mismatch_summary"]["rows"] == 1
    assert calls == [([evaluation_table], tmp_path / "with_mismatch" / "mismatches", ["Local evaluation"], "auto")]


def test_validate_global_fit_uses_explicit_evaluation_mismatch_table(tmp_path, monkeypatch):
    fit_file = tmp_path / "global_fit.json"
    fit_file.write_text(
        json.dumps({"fits": {"22": {"c3A": {"terms": [{"name": "1", "coefficient": 1.0}]}}}}),
        encoding="utf-8",
    )
    validation_table = tmp_path / "validation.csv"
    with validation_table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "mode", "target", "q", "nu", "chi_eff", "chi_a", "value", "mismatch"])
        writer.writeheader()
        writer.writerow({
            "split": "validation",
            "mode": "22",
            "target": "c3A",
            "q": "1.0",
            "nu": "0.25",
            "chi_eff": "0.0",
            "chi_a": "0.0",
            "value": "1.2",
            "mismatch": "9e-5",
        })
    evaluation_table = tmp_path / "global_evaluation_mismatch.csv"
    evaluation_table.write_text(
        "sxs_id,family,nu,q,chi_eff,mismatch,t_start,tref\n"
        "SXS:BBH:0001,nonspinning,0.25,1.0,0.0,1e-5,0,peak22\n",
        encoding="utf-8",
    )
    calls = []

    def fake_validation_plot(predictions, output_dir):
        assert predictions[0]["mismatch"] == "9e-5"
        return []

    def fake_mismatch_plot(tables, output_dir, labels=None, family="auto"):
        calls.append((tables, output_dir, labels, family))
        return {"plots": [], "rows": 1}

    monkeypatch.setattr(calib, "plot_validation_diagnostics", fake_validation_plot)
    monkeypatch.setattr(calib, "plot_teobpm_mismatch_comparison", fake_mismatch_plot)

    summary = calib.validate_global_fit(
        fit_file,
        validation_table,
        tmp_path / "validation_plots",
        mismatch_tables=[evaluation_table],
        mismatch_labels=["Global evaluation"],
        family="nonspinning",
    )

    assert summary["mismatch_summary"]["rows"] == 1
    assert calls == [([evaluation_table], tmp_path / "validation_plots" / "mismatches", ["Global evaluation"], "nonspinning")]
