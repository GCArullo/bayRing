from __future__ import annotations

import math

import numpy as np
import pandas as pd

from scripts.ringdown_quality.config import Config
from scripts.ringdown_quality.metrics import mode_quality_score
from scripts.ringdown_quality.scoring import (
    add_mode_percentiles,
    selected_per_mode_metrics,
    select_simulations,
    score_candidate,
    simulation_scores,
)
from scripts.ringdown_quality.waveform_io import WaveformData


def test_e_mode_uses_default_weights_exactly():
    weights = {"resolution": 0.7, "extrapolation": 0.1, "floor": 0.1, "symmetry": 0.1}
    e_mode, q_mode = mode_quality_score(
        delta_res=2.0,
        delta_ext=3.0,
        delta_floor=4.0,
        delta_sym=5.0,
        weights=weights,
    )
    expected = math.sqrt(0.7 * 2.0**2 + 0.1 * 3.0**2 + 0.1 * 4.0**2 + 0.1 * 5.0**2)
    assert e_mode == expected
    assert q_mode == 1.0 / (expected + 1.0e-12)


def test_worse_metrics_produce_lower_raw_score():
    weights = {"resolution": 0.7, "extrapolation": 0.1, "floor": 0.1, "symmetry": 0.1}
    _, good = mode_quality_score(delta_res=0.1, delta_ext=0.1, delta_floor=0.1, delta_sym=0.1, weights=weights)
    _, bad = mode_quality_score(delta_res=0.2, delta_ext=0.1, delta_floor=0.1, delta_sym=0.1, weights=weights)
    assert bad < good


def _per_mode_df():
    return pd.DataFrame(
        [
            {"sxs_id": "SXS:BBH:0001", "ell": 2, "m": 2, "Q_mode": 100.0, "E_mode": 0.01, "valid_mode": True, "flags": ""},
            {"sxs_id": "SXS:BBH:0001", "ell": 3, "m": 3, "Q_mode": 1.0, "E_mode": 1.00, "valid_mode": True, "flags": ""},
            {"sxs_id": "SXS:BBH:0002", "ell": 2, "m": 2, "Q_mode": 50.0, "E_mode": 0.02, "valid_mode": True, "flags": ""},
            {"sxs_id": "SXS:BBH:0002", "ell": 3, "m": 3, "Q_mode": 50.0, "E_mode": 0.02, "valid_mode": True, "flags": ""},
            {"sxs_id": "SXS:BBH:0003", "ell": 2, "m": 2, "Q_mode": 1.0, "E_mode": 1.00, "valid_mode": True, "flags": ""},
            {"sxs_id": "SXS:BBH:0003", "ell": 3, "m": 3, "Q_mode": 100.0, "E_mode": 0.01, "valid_mode": True, "flags": ""},
        ]
    )


def test_percentile_ranking_is_independent_per_mode():
    ranked = add_mode_percentiles(_per_mode_df())
    row = ranked[(ranked["sxs_id"] == "SXS:BBH:0001") & (ranked["ell"] == 2)].iloc[0]
    other = ranked[(ranked["sxs_id"] == "SXS:BBH:0001") & (ranked["ell"] == 3)].iloc[0]
    assert row["P_mode"] == 1.0
    assert other["P_mode"] == 1.0 / 3.0


def test_common_selection_penalizes_one_bad_mode():
    config = Config()
    config.modes.target_modes = [(2, 2), (3, 3)]
    config.selection.top_k = 1
    per_mode = add_mode_percentiles(_per_mode_df())
    candidates = pd.DataFrame(
        [
            {"sxs_id": "SXS:BBH:0001", "lev_high": 4, "lev_low": 3},
            {"sxs_id": "SXS:BBH:0002", "lev_high": 4, "lev_low": 3},
            {"sxs_id": "SXS:BBH:0003", "lev_high": 4, "lev_low": 3},
        ]
    )

    scores = simulation_scores(per_mode, candidates, config)
    selected = select_simulations(scores, config)
    selected_metrics = selected_per_mode_metrics(per_mode, selected)

    assert list(selected["sxs_id"]) == ["SXS:BBH:0002"]
    assert set(selected_metrics["sxs_id"]) == {"SXS:BBH:0002"}
    assert set(zip(selected_metrics["ell"], selected_metrics["m"])) == {(2, 2), (3, 3)}


def test_score_candidate_can_use_zero_weight_optional_extrapolation(monkeypatch):
    import scripts.ringdown_quality.waveform_io as waveform_io_module

    def fake_waveform(sxs_id: str, lev: int, extrapolation_order: str, config: Config) -> WaveformData:
        time = np.linspace(-10.0, 40.0, 256)
        scale = 1.0 if lev == 4 else 0.99
        base = scale * np.exp(-(time**2) / 60.0) * np.exp(-1j * 0.1 * time)
        modes = {
            (2, 2): base,
            (2, -2): np.conjugate(base),
        }
        return WaveformData(
            sxs_id=sxs_id,
            lev=lev,
            extrapolation_order=extrapolation_order,
            time=time,
            modes=modes,
            ell_min=2,
            ell_max=2,
            metadata={},
        )

    config = Config()
    config.modes.target_modes = [(2, 2)]
    config.waveform.comparison_extrapolation_orders = []
    config.filters.require_extrapolation_comparison = False
    config.metrics.weights = {
        "resolution": 7.0 / 9.0,
        "extrapolation": 0.0,
        "floor": 1.0 / 9.0,
        "symmetry": 1.0 / 9.0,
    }
    candidate = pd.Series({"sxs_id": "SXS:BBH:0001", "lev_low": 3, "lev_high": 4})

    monkeypatch.setattr(waveform_io_module, "load_waveform", fake_waveform)

    rows = score_candidate(candidate, config)

    assert len(rows) == 1
    assert rows[0]["valid_mode"] is True
    assert rows[0]["delta_ext"] == 0.0


def test_score_candidate_marks_waveform_load_failures(monkeypatch):
    import scripts.ringdown_quality.waveform_io as waveform_io_module

    config = Config()
    config.modes.target_modes = [(2, 2)]
    config.waveform.comparison_extrapolation_orders = []
    config.filters.require_extrapolation_comparison = False
    candidate = pd.Series({"sxs_id": "SXS:BBH:0001", "lev_low": 3, "lev_high": 4})

    monkeypatch.setattr(waveform_io_module, "extrapolation_order_available", lambda *args, **kwargs: True)

    def fail_load(*args, **kwargs):
        raise RuntimeError("download failed")

    monkeypatch.setattr(waveform_io_module, "load_waveform", fail_load)

    rows = score_candidate(candidate, config)

    assert len(rows) == 1
    assert rows[0]["valid_mode"] is False
    assert rows[0]["flags"] == "waveform_load_failed"
    assert "failed to load required waveform" in rows[0]["rejection_reason"]
