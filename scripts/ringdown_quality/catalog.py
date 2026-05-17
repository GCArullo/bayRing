from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .waveform_io import (
    configure_sxs,
    discover_available_extrapolation_orders,
    discover_available_levs,
    extrapolation_order_available,
    extrapolation_order_label,
)


def load_catalog(config: Config) -> pd.DataFrame:
    import sxs

    configure_sxs(config)
    df = sxs.load("dataframe", tag=config.catalog.tag)
    if isinstance(df, pd.DataFrame):
        return df.copy()
    if hasattr(df, "to_dataframe"):
        return df.to_dataframe().copy()
    return pd.DataFrame(df)


def _normalise_catalog_index(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "sxs_id" not in result.columns:
        result.insert(0, "sxs_id", result.index.astype(str))
    result["sxs_id"] = result["sxs_id"].astype(str)
    result = result.set_index("sxs_id", drop=False)
    return result.sort_index()


def _column(row: pd.Series, names: list[str]) -> Any:
    for name in names:
        if name in row.index:
            return row[name]
    return None


def _numeric(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 1:
            return _numeric(value[0])
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if match:
            return float(match.group(0))
    return float("nan")


def _vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().strip("[]()")
        parts = re.split(r"[,\s]+", text)
        values = [_numeric(part) for part in parts if part]
    elif isinstance(value, (list, tuple, np.ndarray)):
        values = [_numeric(item) for item in value]
    else:
        return None
    if len(values) >= 2 and all(math.isfinite(x) for x in values[:2]):
        return values
    return None


def _spin_perp(row: pd.Series, body: int) -> float:
    direct = _column(
        row,
        [
            f"reference_chi{body}_perp",
            f"reference-chi{body}-perp",
            f"reference_dimensionless_spin{body}_perp",
            f"reference-dimensionless-spin{body}-perp",
        ],
    )
    direct_value = _numeric(direct)
    if math.isfinite(direct_value):
        return direct_value

    for x_name, y_name in [
        (f"reference_chi{body}x", f"reference_chi{body}y"),
        (f"reference_chi{body}_x", f"reference_chi{body}_y"),
        (f"reference_dimensionless_spin{body}x", f"reference_dimensionless_spin{body}y"),
        (f"reference_dimensionless_spin{body}_x", f"reference_dimensionless_spin{body}_y"),
        (f"reference-dimensionless-spin{body}x", f"reference-dimensionless-spin{body}y"),
    ]:
        x = _numeric(_column(row, [x_name]))
        y = _numeric(_column(row, [y_name]))
        if math.isfinite(x) and math.isfinite(y):
            return float(math.hypot(x, y))

    vector = _vector(
        _column(
            row,
            [
                f"reference_dimensionless_spin{body}",
                f"reference-dimensionless-spin{body}",
                f"reference_chi{body}",
            ],
        )
    )
    if vector is not None:
        return float(math.hypot(vector[0], vector[1]))
    return float("nan")


def _keyword_tokens(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, float) and math.isnan(value):
        return set()
    raw_items = value if isinstance(value, (list, tuple, set)) else [value]
    tokens: set[str] = set()
    for item in raw_items:
        text = str(item).lower()
        tokens.add(text.strip())
        tokens.update(part.strip() for part in re.split(r"[,;]", text) if part.strip())
        tokens.update(part.strip() for part in text.split() if part.strip())
    return tokens


def _has_keyword(value: Any, labels: set[str]) -> bool:
    tokens = _keyword_tokens(value)
    return any(label.lower() in tokens for label in labels)


def _has_deprecated_keyword(value: Any) -> bool:
    return "deprecated" in _keyword_tokens(value)


def _object_is_black_hole(value: Any) -> bool:
    text = str(value).lower().replace("_", " ").replace("-", " ")
    return text in {"bh", "black hole", "blackhole"}


def _objects_are_black_holes(row: pd.Series) -> bool:
    object_columns = [
        ("object1", "object2"),
        ("object_1", "object_2"),
        ("object1_type", "object2_type"),
        ("object_1_type", "object_2_type"),
    ]
    for first, second in object_columns:
        if first in row.index and second in row.index:
            return _object_is_black_hole(row[first]) and _object_is_black_hole(row[second])
    return True


def _deprecated_value(row: pd.Series) -> bool | None:
    value = _column(row, ["deprecated", "is_deprecated"])
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "deprecated"}
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return bool(value)


def _reference_eccentricity(row: pd.Series) -> float:
    return _numeric(
        _column(
            row,
            [
                "reference_eccentricity",
                "reference-eccentricity",
                "reference_ecc",
                "reference-ecc",
            ],
        )
    )


def _choose_levs(levs: list[int]) -> tuple[int | None, int | None]:
    if len(levs) < 2:
        return None, None
    return int(levs[-2]), int(levs[-1])


def _unique_extrapolation_orders(config: Config) -> list[str]:
    raw_orders = [
        extrapolation_order_label(config.waveform.reference_extrapolation_order),
        *[extrapolation_order_label(order) for order in config.waveform.comparison_extrapolation_orders],
    ]
    return list(dict.fromkeys(raw_orders))


def filter_catalog(df: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    catalog = _normalise_catalog_index(df)
    reasons_by_id: dict[str, list[str]] = {sxs_id: [] for sxs_id in catalog.index}
    filter_counts: list[dict[str, int]] = []
    current_ids = set(catalog.index)

    def record_filter(name: str, passing: dict[str, bool]) -> None:
        nonlocal current_ids
        current_ids = {sxs_id for sxs_id in current_ids if passing.get(sxs_id, False)}
        filter_counts.append({"filter": name, "retained": len(current_ids)})

    if config.filters.require_bbh:
        passing = {}
        for sxs_id, row in catalog.iterrows():
            ok = str(sxs_id).startswith("SXS:BBH:") and _objects_are_black_holes(row)
            passing[sxs_id] = ok
            if not ok:
                reasons_by_id[sxs_id].append("not_bbh")
        record_filter("bbh", passing)

    if config.filters.require_non_deprecated:
        passing = {}
        for sxs_id, row in catalog.iterrows():
            deprecated = _deprecated_value(row)
            has_keyword = _has_deprecated_keyword(_column(row, ["keywords", "keyword"]))
            ok = deprecated is False and not has_keyword
            passing[sxs_id] = ok
            if deprecated is None:
                reasons_by_id[sxs_id].append("missing_deprecated_metadata")
            elif deprecated:
                reasons_by_id[sxs_id].append("deprecated")
            if has_keyword:
                reasons_by_id[sxs_id].append("deprecated_keyword")
        record_filter("non_deprecated", passing)

    if config.filters.require_nonprecessing:
        passing = {}
        for sxs_id, row in catalog.iterrows():
            chi1_perp = _spin_perp(row, 1)
            chi2_perp = _spin_perp(row, 2)
            finite = math.isfinite(chi1_perp) and math.isfinite(chi2_perp)
            ok = (
                finite
                and chi1_perp <= config.filters.max_chi1_perp
                and chi2_perp <= config.filters.max_chi2_perp
            )
            if config.filters.keyword_sanity_filter and _has_keyword(
                _column(row, ["keywords", "keyword"]), {"precessing", "precessing-spins"}
            ):
                ok = False
                reasons_by_id[sxs_id].append("precessing_keyword")
            if not finite:
                reasons_by_id[sxs_id].append("missing_or_nonfinite_perpendicular_spin")
            elif chi1_perp > config.filters.max_chi1_perp:
                reasons_by_id[sxs_id].append("chi1_perp_above_threshold")
            elif chi2_perp > config.filters.max_chi2_perp:
                reasons_by_id[sxs_id].append("chi2_perp_above_threshold")
            passing[sxs_id] = ok
        record_filter("nonprecessing", passing)

    if config.filters.require_noneccentric:
        passing = {}
        for sxs_id, row in catalog.iterrows():
            eccentricity = _reference_eccentricity(row)
            ok = math.isfinite(eccentricity) and eccentricity <= config.filters.max_reference_eccentricity
            if config.filters.keyword_sanity_filter and _has_keyword(
                _column(row, ["keywords", "keyword"]), {"eccentric"}
            ):
                ok = False
                reasons_by_id[sxs_id].append("eccentric_keyword")
            if not math.isfinite(eccentricity):
                reasons_by_id[sxs_id].append("missing_or_nonfinite_reference_eccentricity")
            elif eccentricity > config.filters.max_reference_eccentricity:
                reasons_by_id[sxs_id].append("reference_eccentricity_above_threshold")
            passing[sxs_id] = ok
        record_filter("noneccentric", passing)

    catalog = catalog.copy()
    catalog["reference_chi1_perp"] = [_spin_perp(row, 1) for _, row in catalog.iterrows()]
    catalog["reference_chi2_perp"] = [_spin_perp(row, 2) for _, row in catalog.iterrows()]
    catalog["reference_eccentricity"] = [_reference_eccentricity(row) for _, row in catalog.iterrows()]

    if config.filters.require_two_resolutions:
        passing = {}
        available_levs = {}
        available_extrapolation_orders = {}
        required_orders = _unique_extrapolation_orders(config)
        for sxs_id in catalog.index:
            if sxs_id not in current_ids:
                available_levs[sxs_id] = []
                available_extrapolation_orders[sxs_id] = []
                passing[sxs_id] = False
                continue
            levs = discover_available_levs(str(sxs_id), config=config)
            available_levs[sxs_id] = levs
            ok = len(levs) >= 2
            passing[sxs_id] = ok
            available_extrapolation_orders[sxs_id] = []
            if not ok:
                reasons_by_id[sxs_id].append("insufficient_levs")
            else:
                _, lev_high = _choose_levs(levs)
                if lev_high is not None:
                    discovered_orders = discover_available_extrapolation_orders(str(sxs_id), lev_high, config=config)
                    if discovered_orders is not None:
                        available_extrapolation_orders[sxs_id] = discovered_orders
                        available_set = {extrapolation_order_label(order) for order in discovered_orders}
                        if len(available_set) < 2:
                            ok = False
                            reasons_by_id[sxs_id].append("insufficient_extrapolation_orders")
                        elif any(order not in available_set for order in required_orders):
                            ok = False
                            reasons_by_id[sxs_id].append("missing_required_extrapolation_order")
                    else:
                        availability = {
                            order: extrapolation_order_available(sxs_id, lev_high, order, config=config)
                            for order in required_orders
                        }
                        if any(value is False for value in availability.values()):
                            ok = False
                            reasons_by_id[sxs_id].append("missing_required_extrapolation_order")
                        elif all(value is not None for value in availability.values()) and len(
                            [value for value in availability.values() if value is True]
                        ) < 2:
                            ok = False
                            reasons_by_id[sxs_id].append("insufficient_extrapolation_orders")
                        else:
                            available_extrapolation_orders[sxs_id] = [
                                order for order, value in availability.items() if value is True
                            ]
                else:
                    ok = False
                    reasons_by_id[sxs_id].append("insufficient_levs")
            passing[sxs_id] = ok
        record_filter("two_resolutions", passing)
        catalog["available_levs"] = [" ".join(str(lev) for lev in available_levs[sxs_id]) for sxs_id in catalog.index]
        catalog["available_extrapolation_orders"] = [
            " ".join(order for order in available_extrapolation_orders[sxs_id]) for sxs_id in catalog.index
        ]
        lev_pairs = [_choose_levs(available_levs[sxs_id]) for sxs_id in catalog.index]
        catalog["lev_low"] = [pair[0] for pair in lev_pairs]
        catalog["lev_high"] = [pair[1] for pair in lev_pairs]

    catalog["rejection_reason"] = [";".join(reasons_by_id[sxs_id]) for sxs_id in catalog.index]
    candidates = catalog.loc[[sxs_id for sxs_id in catalog.index if sxs_id in current_ids]].copy()
    rejected = catalog.loc[[sxs_id for sxs_id in catalog.index if sxs_id not in current_ids]].copy()
    candidates.attrs["filter_counts"] = filter_counts
    rejected.attrs["filter_counts"] = filter_counts
    return candidates, rejected
