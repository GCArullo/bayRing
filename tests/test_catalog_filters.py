from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.ringdown_quality.catalog import filter_catalog
from scripts.ringdown_quality.config import Config


def _config_without_lev_filter() -> Config:
    config = Config()
    config.filters.require_two_resolutions = False
    return config


def test_catalog_filters_reject_bad_catalog_rows():
    df = pd.DataFrame(
        [
            {
                "deprecated": False,
                "reference_chi1_perp": 0.0,
                "reference_chi2_perp": 0.0,
                "reference_eccentricity": 1.0e-4,
                "keywords": "Nonprecessing",
            },
            {
                "deprecated": True,
                "reference_chi1_perp": 0.0,
                "reference_chi2_perp": 0.0,
                "reference_eccentricity": 1.0e-4,
                "keywords": "",
            },
            {
                "deprecated": False,
                "reference_chi1_perp": 2.0e-3,
                "reference_chi2_perp": 0.0,
                "reference_eccentricity": 1.0e-4,
                "keywords": "",
            },
            {
                "deprecated": False,
                "reference_chi1_perp": 0.0,
                "reference_chi2_perp": 0.0,
                "reference_eccentricity": 2.0e-3,
                "keywords": "",
            },
            {
                "deprecated": False,
                "reference_chi1_perp": np.nan,
                "reference_chi2_perp": 0.0,
                "reference_eccentricity": 1.0e-4,
                "keywords": "",
            },
            {
                "deprecated": False,
                "reference_chi1_perp": 0.0,
                "reference_chi2_perp": 0.0,
                "reference_eccentricity": 1.0e-4,
                "keywords": "",
            },
        ],
        index=[
            "SXS:BBH:0001",
            "SXS:BBH:0002",
            "SXS:BBH:0003",
            "SXS:BBH:0004",
            "SXS:BBH:0005",
            "SXS:NSNS:0006",
        ],
    )

    candidates, rejected = filter_catalog(df, _config_without_lev_filter())

    assert list(candidates["sxs_id"]) == ["SXS:BBH:0001"]
    reasons = rejected.set_index("sxs_id")["rejection_reason"].to_dict()
    assert "deprecated" in reasons["SXS:BBH:0002"]
    assert "chi1_perp_above_threshold" in reasons["SXS:BBH:0003"]
    assert "reference_eccentricity_above_threshold" in reasons["SXS:BBH:0004"]
    assert "missing_or_nonfinite_perpendicular_spin" in reasons["SXS:BBH:0005"]
    assert "not_bbh" in reasons["SXS:NSNS:0006"]


def test_catalog_filters_reject_missing_required_metadata():
    df = pd.DataFrame(
        [
            {
                "deprecated": np.nan,
                "reference_chi1_perp": 0.0,
                "reference_chi2_perp": 0.0,
                "reference_eccentricity": 1.0e-4,
            },
            {
                "deprecated": False,
                "reference_chi1_perp": 0.0,
                "reference_chi2_perp": 0.0,
                "reference_eccentricity": np.nan,
            },
        ],
        index=["SXS:BBH:0100", "SXS:BBH:0101"],
    )

    candidates, rejected = filter_catalog(df, _config_without_lev_filter())

    assert candidates.empty
    reasons = rejected.set_index("sxs_id")["rejection_reason"].to_dict()
    assert "missing_deprecated_metadata" in reasons["SXS:BBH:0100"]
    assert "missing_or_nonfinite_reference_eccentricity" in reasons["SXS:BBH:0101"]


def test_catalog_filter_allows_missing_extrapolation_when_disabled(monkeypatch):
    import scripts.ringdown_quality.catalog as catalog_module

    df = pd.DataFrame(
        [
            {
                "deprecated": False,
                "reference_chi1_perp": 0.0,
                "reference_chi2_perp": 0.0,
                "reference_eccentricity": 1.0e-4,
                "keywords": "",
            }
        ],
        index=["SXS:BBH:0102"],
    )
    config = Config()
    config.filters.require_extrapolation_comparison = False

    monkeypatch.setattr(catalog_module, "discover_available_levs", lambda sxs_id, config=None: [3, 4])
    monkeypatch.setattr(
        catalog_module,
        "discover_available_extrapolation_orders",
        lambda sxs_id, lev, config=None: ["N2"],
    )

    candidates, rejected = filter_catalog(df, config)

    assert list(candidates["sxs_id"]) == ["SXS:BBH:0102"]
    assert rejected.empty
