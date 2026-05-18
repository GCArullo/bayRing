from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml


DEFAULT_WEIGHTS = {
    "resolution": 0.7,
    "extrapolation": 0.1,
    "floor": 0.1,
    "symmetry": 0.1,
}
DEFAULT_RINGDOWN_WINDOW = [0.0, 30.0]
FORBIDDEN_OPTION_KEYS = {
    "qnm",
    "qnmfit",
    "qnmfits",
    "qnm_fit",
    "qnm_fits",
    "fit_qnm",
    "ringdown_fit",
    "ringdown_fitting",
    "fit_remnant",
    "remnant_mass",
    "remnant_spin",
    "remnant_parameters",
}


@dataclass
class CatalogConfig:
    tag: str = "3.0.0"
    auto_download: bool = True
    cache: bool = True


@dataclass
class FiltersConfig:
    require_bbh: bool = True
    require_non_deprecated: bool = True
    require_nonprecessing: bool = True
    require_noneccentric: bool = True
    max_chi1_perp: float = 1.0e-3
    max_chi2_perp: float = 1.0e-3
    max_reference_eccentricity: float = 1.0e-3
    keyword_sanity_filter: bool = True
    require_two_resolutions: bool = True
    require_extrapolation_comparison: bool = True


@dataclass
class WaveformConfig:
    data_type: str = "h"
    reference_extrapolation_order: str = "N2"
    comparison_extrapolation_orders: list[str] = field(default_factory=lambda: ["N3", "N4"])
    ringdown_window_M: list[float] = field(default_factory=lambda: list(DEFAULT_RINGDOWN_WINDOW))
    merger_time_definition: str = "ell2_norm_peak"
    remove_memory: bool = False
    interpolation: str = "cubic"
    interpolation_fallback: str = "linear"
    min_samples_in_window: int = 64


@dataclass
class ModesConfig:
    target_modes: list[tuple[int, int]] = field(
        default_factory=lambda: [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)]
    )


@dataclass
class MetricsConfig:
    norm_floor: float = 1.0e-14
    pointwise_error_floor: float = 1.0e-15
    signal_to_error_quantile: float = 0.10
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))


@dataclass
class SelectionConfig:
    aggregate_method: str = "min_percentile"
    top_k: int | None = 50
    top_fraction: float | None = None
    min_common_score: float | None = None
    require_all_modes_valid: bool = True


@dataclass
class OutputConfig:
    directory: str = "outputs/ringdown_quality"
    write_per_mode_metrics: bool = True
    write_simulation_scores: bool = True
    write_selected_metrics: bool = True
    file_format: str = "csv"


@dataclass
class RuntimeConfig:
    checkpoint_every: int = 25
    max_simulations: int | None = None
    random_seed: int = 1234
    verbose: bool = True


@dataclass
class Config:
    catalog: CatalogConfig = field(default_factory=CatalogConfig)
    filters: FiltersConfig = field(default_factory=FiltersConfig)
    waveform: WaveformConfig = field(default_factory=WaveformConfig)
    modes: ModesConfig = field(default_factory=ModesConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["modes"]["target_modes"] = [list(mode) for mode in self.modes.target_modes]
        return data


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalise_modes(raw_modes: Any) -> list[tuple[int, int]]:
    modes = []
    for mode in raw_modes or []:
        if isinstance(mode, str):
            parts = mode.replace(",", " ").split()
        else:
            parts = list(mode)
        if len(parts) != 2:
            raise ValueError(f"Mode entries must contain exactly two integers; got {mode!r}.")
        modes.append((int(parts[0]), int(parts[1])))
    return sorted(modes, key=lambda item: (item[0], item[1]))


def _dataclass_from_dict(cls: type[Any], values: dict[str, Any]) -> Any:
    return cls(**values)


def config_from_dict(raw: dict[str, Any] | None) -> Config:
    data = _merge_dict(Config().to_dict(), raw or {})
    data["modes"]["target_modes"] = _normalise_modes(data["modes"].get("target_modes"))
    data["waveform"]["ringdown_window_M"] = [float(x) for x in data["waveform"]["ringdown_window_M"]]
    data["metrics"]["weights"] = {str(key): float(value) for key, value in data["metrics"]["weights"].items()}
    return Config(
        catalog=_dataclass_from_dict(CatalogConfig, data["catalog"]),
        filters=_dataclass_from_dict(FiltersConfig, data["filters"]),
        waveform=_dataclass_from_dict(WaveformConfig, data["waveform"]),
        modes=_dataclass_from_dict(ModesConfig, data["modes"]),
        metrics=_dataclass_from_dict(MetricsConfig, data["metrics"]),
        selection=_dataclass_from_dict(SelectionConfig, data["selection"]),
        output=_dataclass_from_dict(OutputConfig, data["output"]),
        runtime=_dataclass_from_dict(RuntimeConfig, data["runtime"]),
    )


def _iter_keys(raw: Any, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    if isinstance(raw, dict):
        keys: list[tuple[str, ...]] = []
        for key, value in raw.items():
            path = prefix + (str(key),)
            keys.append(path)
            keys.extend(_iter_keys(value, path))
        return keys
    if isinstance(raw, list):
        keys = []
        for index, value in enumerate(raw):
            keys.extend(_iter_keys(value, prefix + (str(index),)))
        return keys
    return []


def _validate_no_forbidden_options(raw: dict[str, Any] | None) -> None:
    for path in _iter_keys(raw or {}):
        key = path[-1].lower().replace("-", "_")
        if key in FORBIDDEN_OPTION_KEYS:
            dotted = ".".join(path)
            raise ValueError(
                f"Forbidden option `{dotted}` was provided. This selector must not run QNM or "
                "ringdown-fitting diagnostics."
            )
        if "qnm" in key:
            dotted = ".".join(path)
            raise ValueError(
                f"Forbidden option `{dotted}` was provided. This selector must not use QNM fitting options."
            )


def validate_config(
    config: Config,
    raw: dict[str, Any] | None = None,
    *,
    allow_nonstandard_window: bool = False,
    allow_nonstandard_weights: bool = False,
) -> None:
    _validate_no_forbidden_options(raw)
    if not allow_nonstandard_window and config.waveform.ringdown_window_M != DEFAULT_RINGDOWN_WINDOW:
        raise ValueError(
            "`waveform.ringdown_window_M` must be exactly [0.0, 30.0] unless "
            "`--allow-nonstandard-window` is passed."
        )

    if not allow_nonstandard_weights:
        weights = config.metrics.weights
        if set(weights) != set(DEFAULT_WEIGHTS):
            raise ValueError("Metric weights must contain resolution, extrapolation, floor, and symmetry.")
        for key, expected in DEFAULT_WEIGHTS.items():
            if weights[key] != expected:
                raise ValueError(
                    "Metric weights must be exactly "
                    "resolution=0.7, extrapolation=0.1, floor=0.1, symmetry=0.1 unless "
                    "`--allow-nonstandard-weights` is passed."
                )
        if sum(Decimal(str(value)) for value in weights.values()) != Decimal("1.0"):
            raise ValueError("Default metric weights must sum to exactly 1.0.")

    if config.waveform.merger_time_definition != "ell2_norm_peak":
        raise ValueError("Only `ell2_norm_peak` merger-time definition is implemented.")
    if config.waveform.data_type != "h":
        raise ValueError("Only strain `h` waveforms are supported by this selector.")
    if config.selection.aggregate_method not in {"min_percentile", "geometric_mean_percentile"}:
        raise ValueError("Unsupported aggregate_method; use min_percentile or geometric_mean_percentile.")
    if config.output.file_format != "csv":
        raise ValueError("Only CSV output is currently implemented.")


def load_config(
    path: str | Path,
    *,
    allow_nonstandard_window: bool = False,
    allow_nonstandard_weights: bool = False,
) -> Config:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    config = config_from_dict(raw)
    validate_config(
        config,
        raw,
        allow_nonstandard_window=allow_nonstandard_window,
        allow_nonstandard_weights=allow_nonstandard_weights,
    )
    return config


def dump_config(config: Config, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
