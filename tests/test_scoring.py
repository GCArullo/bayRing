from __future__ import annotations

import math

import pandas as pd

from scripts.ringdown_quality.config import Config
from scripts.ringdown_quality.metrics import mode_quality_score
from scripts.ringdown_quality.scoring import (
    add_mode_percentiles,
    selected_per_mode_metrics,
    select_simulations,
    simulation_scores,
)


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
