from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .catalog import filter_catalog, load_catalog
from .config import Config, dump_config
from .scoring import (
    selected_per_mode_metrics,
    select_simulations,
    score_candidates,
    simulation_scores,
)


REQUIRED_OUTPUT_FILES = [
    "run_config_resolved.yml",
    "candidate_catalog.csv",
    "rejected_catalog.csv",
    "per_mode_metrics.csv",
    "simulation_scores.csv",
    "selected_simulations.csv",
    "selected_per_mode_metrics.csv",
    "run_summary.txt",
]


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _selection_rule(config: Config) -> str:
    if config.selection.top_k is not None:
        return f"top_k={config.selection.top_k}"
    if config.selection.top_fraction is not None:
        return f"top_fraction={config.selection.top_fraction}"
    if config.selection.min_common_score is not None:
        return f"min_common_score={config.selection.min_common_score}"
    return "top_k=50"


def _filter_counts_text(candidates: pd.DataFrame, rejected: pd.DataFrame) -> list[str]:
    counts = candidates.attrs.get("filter_counts") or rejected.attrs.get("filter_counts") or []
    return [f"  {item['filter']}: {item['retained']}" for item in counts]


def build_run_summary(
    *,
    config: Config,
    catalog_rows_loaded: int,
    candidates: pd.DataFrame,
    rejected: pd.DataFrame,
    per_mode_metrics: pd.DataFrame,
    simulation_scores_df: pd.DataFrame,
    selected: pd.DataFrame,
) -> str:
    valid_all = int(simulation_scores_df["valid_all_modes"].sum()) if "valid_all_modes" in simulation_scores_df else 0
    lines = [
        f"Catalog tag: {config.catalog.tag}",
        f"Number of catalog rows loaded: {catalog_rows_loaded}",
        "Number retained after each filter:",
        *_filter_counts_text(candidates, rejected),
        "Target modes: " + " ".join(f"({ell},{m})" for ell, m in config.modes.target_modes),
        "Ringdown window: [0, 30]M",
        "Metric weights: "
        + ", ".join(f"{key}={value}" for key, value in config.metrics.weights.items()),
        f"Number of simulations with all modes valid: {valid_all}",
        f"Aggregation method: {config.selection.aggregate_method}",
        f"Selection rule: {_selection_rule(config)}",
        f"Number selected: {len(selected)}",
        "Top 10 selected SXS IDs with S_common:",
    ]
    for _, row in selected.head(10).iterrows():
        lines.append(f"  {row['sxs_id']}: {row['S_common']:.6g}")

    valid_counts = (
        per_mode_metrics[per_mode_metrics["valid_mode"].astype(bool)]
        .groupby(["ell", "m"])
        .size()
        .to_dict()
        if not per_mode_metrics.empty
        else {}
    )
    for mode in config.modes.target_modes:
        count = int(valid_counts.get(mode, 0))
        if count < 20:
            lines.append(
                f"WARNING: target mode ({mode[0]},{mode[1]}) has only {count} valid simulations; "
                "percentile rankings are unstable."
            )
    return "\n".join(lines) + "\n"


def write_outputs(
    *,
    config: Config,
    catalog_rows_loaded: int,
    candidates: pd.DataFrame,
    rejected: pd.DataFrame,
    per_mode_metrics: pd.DataFrame,
    simulation_scores_df: pd.DataFrame,
    selected: pd.DataFrame,
) -> dict[str, Path]:
    output_dir = Path(config.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_metrics = selected_per_mode_metrics(per_mode_metrics, selected)

    paths: dict[str, Path] = {}
    paths["run_config_resolved"] = output_dir / "run_config_resolved.yml"
    dump_config(config, paths["run_config_resolved"])

    csv_outputs = {
        "candidate_catalog": candidates.reset_index(drop=True),
        "rejected_catalog": rejected.reset_index(drop=True),
        "per_mode_metrics": per_mode_metrics,
        "simulation_scores": simulation_scores_df,
        "selected_simulations": selected,
        "selected_per_mode_metrics": selected_metrics,
    }
    for key, df in csv_outputs.items():
        paths[key] = output_dir / f"{key}.csv"
        _write_csv(df, paths[key])

    paths["run_summary"] = output_dir / "run_summary.txt"
    paths["run_summary"].write_text(
        build_run_summary(
            config=config,
            catalog_rows_loaded=catalog_rows_loaded,
            candidates=candidates,
            rejected=rejected,
            per_mode_metrics=per_mode_metrics,
            simulation_scores_df=simulation_scores_df,
            selected=selected,
        ),
        encoding="utf-8",
    )
    return paths


def run_pipeline(config: Config) -> dict[str, Any]:
    catalog = load_catalog(config)
    candidates, rejected = filter_catalog(catalog, config)
    per_mode = score_candidates(candidates, config)
    scores = simulation_scores(per_mode, candidates, config)
    selected = select_simulations(scores, config)
    paths = write_outputs(
        config=config,
        catalog_rows_loaded=len(catalog),
        candidates=candidates,
        rejected=rejected,
        per_mode_metrics=per_mode,
        simulation_scores_df=scores,
        selected=selected,
    )
    return {
        "catalog": catalog,
        "candidates": candidates,
        "rejected": rejected,
        "per_mode_metrics": per_mode,
        "simulation_scores": scores,
        "selected_simulations": selected,
        "paths": paths,
    }
