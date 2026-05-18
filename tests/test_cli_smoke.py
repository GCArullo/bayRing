from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.ringdown_quality.cli import main
from scripts.ringdown_quality.waveform_io import WaveformData


def _fake_waveform(sxs_id: str, lev: int, extrapolation_order: str, config) -> WaveformData:
    time = np.linspace(-10.0, 40.0, 256)
    base = np.exp(-(time**2) / 50.0) * np.exp(-1j * 0.15 * time)
    scale = 1.0
    if lev < 4:
        scale *= 0.995
    if extrapolation_order == "N3":
        scale *= 1.002
    if extrapolation_order == "N4":
        scale *= 0.998

    modes = {}
    positive_modes = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)]
    for ell, m in positive_modes:
        h_lm = scale * (0.5 ** (ell - 2)) * base * np.exp(-1j * 0.02 * m * time)
        modes[(ell, m)] = h_lm
        modes[(ell, -m)] = (-1) ** ell * np.conjugate(h_lm)
    modes[(2, 0)] = scale * np.real(base)

    return WaveformData(
        sxs_id=sxs_id,
        lev=lev,
        extrapolation_order=extrapolation_order,
        time=time,
        modes=modes,
        ell_min=2,
        ell_max=4,
        metadata={},
    )


def test_cli_smoke_writes_required_outputs(tmp_path, monkeypatch):
    import scripts.ringdown_quality.catalog as catalog_module
    import scripts.ringdown_quality.reporting as reporting_module
    import scripts.ringdown_quality.waveform_io as waveform_io_module

    catalog_df = pd.DataFrame(
        [
            {
                "deprecated": False,
                "reference_chi1_perp": 0.0,
                "reference_chi2_perp": 0.0,
                "reference_eccentricity": 1.0e-4,
                "keywords": "Nonprecessing",
                "reference_mass_ratio": 1.0,
                "reference_chi_eff": 0.0,
                "number_of_orbits": 20,
            }
        ],
        index=["SXS:BBH:0001"],
    )

    monkeypatch.setattr(reporting_module, "load_catalog", lambda config: catalog_df)
    monkeypatch.setattr(catalog_module, "discover_available_levs", lambda sxs_id, config=None: [3, 4])
    monkeypatch.setattr(waveform_io_module, "load_waveform", _fake_waveform)

    output_dir = tmp_path / "ringdown_quality"
    rc = main(["run", "--config", "configs/default.yml", "--output", str(output_dir), "--max-simulations", "1"])

    assert rc == 0
    for filename in [
        "run_config_resolved.yml",
        "candidate_catalog.csv",
        "rejected_catalog.csv",
        "per_mode_metrics.csv",
        "simulation_scores.csv",
        "selected_simulations.csv",
        "selected_per_mode_metrics.csv",
        "run_summary.txt",
    ]:
        assert (output_dir / filename).exists()

    selected = pd.read_csv(output_dir / "selected_simulations.csv")
    selected_metrics = pd.read_csv(output_dir / "selected_per_mode_metrics.csv")
    assert list(selected["sxs_id"]) == ["SXS:BBH:0001"]
    assert set(selected_metrics["sxs_id"]) == {"SXS:BBH:0001"}
