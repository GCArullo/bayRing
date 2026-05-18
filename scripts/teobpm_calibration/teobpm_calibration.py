"""TEOBPM calibration workflow helpers.

This module intentionally keeps the expensive parts of the campaign explicit:
SXS downloads, local bayRing fits, and validation waveform comparisons can be
run by generated commands, while the metadata manifests and global-fit files are
plain JSON/CSV artifacts that can be inspected and versioned.
"""

from __future__ import annotations

import argparse
import configparser
import concurrent.futures
import csv
import itertools
import json
import math
import os
import shlex
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_MODES = [(2, 2), (2, 1), (3, 3), (3, 2), (3, 1), (4, 4), (4, 3), (4, 2), (4, 1), (5, 5)]
TEOB_MODE_MIXING_PARENTS = {
    (3, 1): (2, 1),
    (3, 2): (2, 2),
    (4, 1): (2, 1),
    (4, 2): (2, 2),
    (4, 3): (3, 3),
}
DEFAULT_MODE_MIXING_MODES = sorted(TEOB_MODE_MIXING_PARENTS)
DEFAULT_COUNTER_ROTATING_MODES = [(2, 1)]
QQNM_TERMS_BY_MODE = {(5, 5): 'Px220x330'}
REFERENCE_FIT_TARGETS = (
    "A_ref_over_nu",
    "A_refdot_over_nu",
    "A_refdotdot_over_nu",
    "omg_ref",
    "phi_mrg",
    "DeltaT",
)
LOCAL_FIT_TARGETS_HYPTAN = ("A_peak_over_nu", "omg_peak", "c3A", "c3p", "c4p") + REFERENCE_FIT_TARGETS
LOCAL_FIT_TARGETS_RATEXP = ("A_peak_over_nu", "A_peakdotdot_over_nu", "omg_peak", "c2A", "c3A", "c2p", "c3p", "c4p") + REFERENCE_FIT_TARGETS
LOCAL_FIT_TARGETS_SEOBNRV5 = ("A_peak_over_nu", "omg_peak", "c2A", "c2p", "c3A", "c3p") + REFERENCE_FIT_TARGETS
TEOB_44_WINDOW_TARGETS = (
    "quad44_window_delay",
    "quad44_window_width",
    "quad44_window_steepness",
)
PHASE_TARGET_PREFIX = "delta_phi_"
FIT_SCHEMA = "bayRing.teobpm.global-fit.v1"
APPENDIX_A_SCHEMA = "bayRing.teobpm.appendix-a.v1"
APPENDIX_A_EQUAL_MASS_IDS = ("1477", "0225", "0219", "0305", "0224", "0148", "2086", "0156", "0172", "3628", "3627", "0160")
APPENDIX_A_NONSPINNING_IDS = ("0180", "0007", "0169", "0030", "0167", "0295", "0056", "0296", "0297", "0298", "0299", "0063", "0300", "0301", "0185", "0303")
APPENDIX_A_TARGETS_HYPTAN = ("c3A", "c3p", "c4p")
APPENDIX_A_TARGETS_RATEXP = ("c2A", "c3A", "c2p", "c3p", "c4p")
APPENDIX_A_RAO_TARGET_COLUMNS = {
    "c2A": "c_2_A",
    "c3A": "c_3_A",
    "c2p": "c_2_phi",
    "c3p": "c_3_phi",
    "c4p": "c_4_phi",
}
APPENDIX_A_CURRENT_RAO_FIT = "src/data/fits/nc_fits_sxs_non-spinning/order_fits_nu_1.csv"
APPENDIX_A_RAO_VARIABLE_COLUMNS = {
    "nu": ("nu",),
}
APPENDIX_A_NONCIRCULAR_VARIABLES = {"ecc", "e0", "eccentricity", "emrg", "bmrg", "jmrg"}
APPENDIX_A_PAST_HYPTAN_EQUAL_MASS = {
    "c3A": (0.0, 0.0, 0.044763, -0.138980, -0.366538),
    "c3p": (-1.174263, -0.9099211, 1.690678, 2.866629, 4.188784),
    "c4p": (0.0, 0.0, 2.925663, 4.362706, 2.462696),
}
APPENDIX_A_PAST_HYPTAN_NONSPINNING = {
    "c3A": {"offset": -1.5, "num": (0.75497, -0.56187), "den": (1.0,)},
    "c3p": {"offset": 3.5, "num": (296.64, -63.107, 4.4414), "den": (69.129, 13.299, 1.0)},
    "c4p": {"offset": 3.25, "num": (-109.47, 7.1508), "den": (287.42, 556.34, 1.0)},
}


@dataclass
class SimulationRecord:
    """Normalised SXS metadata used by the calibration workflow."""

    sxs_id: str
    q: float
    nu: float
    chi1z: float
    chi2z: float
    chi_eff: float
    chi_a: float
    eccentricity: float
    spin1_perp: float
    spin2_perp: float
    number_of_orbits: float
    resolution_level: int
    quality_score: float
    split: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CalibrationConfig:
    """Configuration controlling dataset and command generation."""

    family: str
    output_dir: str
    modes: list[tuple[int, int]] = field(default_factory=lambda: list(DEFAULT_MODES))
    mode_mixing_modes: list[tuple[int, int]] = field(default_factory=lambda: list(DEFAULT_MODE_MIXING_MODES))
    counter_rotating_modes: list[tuple[int, int]] = field(default_factory=lambda: list(DEFAULT_COUNTER_ROTATING_MODES))
    template: str = "RatExp"
    teob_calibration: str = "qc"
    validation_fraction: float = 0.33
    seed: int = 1234
    sxs_ids: list[str] = field(default_factory=list)
    random_fraction: float = 1.0
    max_eccentricity: float = 1.0e-3
    max_transverse_spin: float = 1.0e-6
    nonspinning_spin_tolerance: float = 1.0e-6
    equal_mass_tolerance: float = 1.0e-6
    min_quality_score: float | None = None
    bayring_executable: str = "bayRing"
    nr_data_dir: str = ""
    properties_file: str = ""
    extrap_order: int = 2
    resolution_level: int = -1
    t_start: float = 0.0
    t_end: float = 140.0
    method: str = "Minimization"
    min_method: str = "trf"
    n_random_seeds: int = 16
    n_mode_workers: int = 1
    base_nonspinning_fit: str = ""
    notes: str = ""


@dataclass
class LocalFitJob:
    """One generated bayRing local-fit job."""

    sxs_id: str
    mode: str
    split: str
    config_file: str
    outdir: str
    q: float
    nu: float
    chi1z: float
    chi2z: float
    chi_eff: float
    chi_a: float
    eccentricity: float
    quality_score: float


def _teobpm_requires_merger_peak_data(template: str) -> bool:
    """TEOBPM calibration jobs use NR-computed merger data instead of pyRing's built-in peak fits."""

    if template in {"HypTan", "RatExp", "SEOBNRv5"}:
        return True
    raise ValueError(f"Unknown TEOBPM template `{template}`.")


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    return bool(_as_float(value, default=0.0))


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _first_present(raw: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in raw and raw[key] not in (None, ""):
            return raw[key]
    return default


def _vector_component(raw: Any, index: int) -> float:
    if raw is None or raw == "":
        return 0.0
    if isinstance(raw, str):
        cleaned = raw.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
        parts = [part for part in cleaned.replace(",", " ").split() if part]
        if len(parts) > index:
            return _as_float(parts[index])
        return 0.0
    if isinstance(raw, (list, tuple)) and len(raw) > index:
        return _as_float(raw[index])
    return 0.0


def _spin_components(raw: dict[str, Any], body: int) -> tuple[float, float, float]:
    vector = _first_present(
        raw,
        [
            f"reference_dimensionless_spin{body}",
            f"relaxed_dimensionless_spin{body}",
            f"initial_dimensionless_spin{body}",
            f"chi{body}",
            f"spin{body}",
        ],
    )
    x = _as_float(_first_present(raw, [f"chi{body}x", f"spin{body}x"]), _vector_component(vector, 0))
    y = _as_float(_first_present(raw, [f"chi{body}y", f"spin{body}y"]), _vector_component(vector, 1))
    z = _as_float(_first_present(raw, [f"chi{body}z", f"spin{body}z"]), _vector_component(vector, 2))
    return x, y, z


def _normalise_sxs_id(raw_id: Any) -> str:
    text = str(raw_id).strip()
    text = text.replace("SXS_BBH_", "SXS:BBH:").replace("SXS-BBH-", "SXS:BBH:")
    if text.startswith("SXS:BBH:"):
        prefix, number = text.rsplit(":", 1)
        return f"{prefix}:{int(number):04d}" if number.isdigit() else text
    if text.startswith("BBH:"):
        return "SXS:" + text
    if text.isdigit():
        return "SXS:BBH:" + text.zfill(4)
    return text


def _default_worker_count() -> int:
    return max(1, os.cpu_count() or 1)


def _mode_label(mode: tuple[int, int]) -> str:
    l_value, m_value = mode
    return f"{int(l_value)}{int(m_value)}"


def _mode_from_label(label: str) -> tuple[int, int]:
    text = str(label).strip()
    if len(text) < 2:
        raise ValueError(f"Invalid mode label `{label}`.")
    return int(text[0]), int(text[1:])


def parse_modes(raw_modes: str | Iterable[Any] | None) -> list[tuple[int, int]]:
    if raw_modes in (None, ""):
        return list(DEFAULT_MODES)
    if isinstance(raw_modes, str):
        text = raw_modes.strip()
        if text.startswith("["):
            raw_modes = json.loads(text.replace("(", "[").replace(")", "]"))
        else:
            raw_modes = [item.strip() for item in text.split(",") if item.strip()]

    modes: list[tuple[int, int]] = []
    for raw_mode in raw_modes:
        if isinstance(raw_mode, str):
            token = raw_mode.strip().replace("(", "").replace(")", "")
            if " " in token or "," in token:
                pieces = [piece for piece in token.replace(",", " ").split() if piece]
                if len(pieces) != 2:
                    raise ValueError(f"Invalid mode token `{raw_mode}`.")
                modes.append((int(pieces[0]), int(pieces[1])))
            else:
                if len(token) < 2:
                    raise ValueError(f"Invalid compact mode `{raw_mode}`.")
                l_value = int(token[0])
                m_value = int(token[1:])
                modes.append((l_value, m_value))
        else:
            if len(raw_mode) != 2:
                raise ValueError(f"Invalid mode entry `{raw_mode}`.")
            modes.append((int(raw_mode[0]), int(raw_mode[1])))

    if len(set(modes)) != len(modes):
        raise ValueError(f"Repeated TEOBPM modes requested: {modes}.")
    return modes


def parse_sxs_ids(raw_ids: str | Iterable[str] | None) -> list[str]:
    if raw_ids in (None, ""):
        return []
    if isinstance(raw_ids, str):
        tokens = raw_ids.replace("\n", ",").split(",")
    else:
        tokens = list(raw_ids)
    ids = [_normalise_sxs_id(token) for token in tokens]
    ids = [sxs_id for sxs_id in ids if sxs_id]
    return sorted(set(ids))


def read_sxs_id_file(path: str | Path | None) -> list[str]:
    if not path:
        return []
    tokens = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if text:
            tokens.extend(item.strip() for item in text.split(",") if item.strip())
    return parse_sxs_ids(tokens)


def _normalise_family(family: str) -> str:
    if family == "aligned-spin":
        return "spinning"
    return family


def _is_spin_family(family: str) -> bool:
    return _normalise_family(family) in {"equal-mass-spinning", "spinning"}


def symmetric_mass_ratio_from_q(q: float) -> float:
    return q / (1.0 + q) ** 2


def normalise_catalog_entry(raw: dict[str, Any]) -> SimulationRecord | None:
    raw_id = _first_present(raw, ["sxs_id", "id", "simulation", "simulation_id", "object"], None)
    if raw_id is None:
        return None

    m1 = _as_float(_first_present(raw, ["reference_mass1", "mass1", "m1"]), 0.0)
    m2 = _as_float(_first_present(raw, ["reference_mass2", "mass2", "m2"]), 0.0)
    q = _as_float(_first_present(raw, ["q", "mass_ratio", "reference_mass_ratio"]), 0.0)
    if q <= 0.0 and m1 > 0.0 and m2 > 0.0:
        q = max(m1, m2) / min(m1, m2)
    if q <= 0.0:
        return None

    nu = symmetric_mass_ratio_from_q(q)
    spin1x, spin1y, chi1z = _spin_components(raw, 1)
    spin2x, spin2y, chi2z = _spin_components(raw, 2)
    spin1_perp = math.hypot(spin1x, spin1y)
    spin2_perp = math.hypot(spin2x, spin2y)
    chi_eff = (q * chi1z + chi2z) / (1.0 + q)
    chi_a = 0.5 * (chi1z - chi2z)
    eccentricity = abs(
        _as_float(
            _first_present(
                raw,
                ["reference_eccentricity", "initial_eccentricity", "eccentricity", "e0"],
                0.0,
            )
        )
    )
    number_of_orbits = _as_float(_first_present(raw, ["number_of_orbits", "num_orbits", "Norb"], 0.0))
    resolution_level = _as_int(_first_present(raw, ["resolution_level", "Lev", "lev", "highest_lev"], 0), 0)
    quality_score = (
        100.0 * max(resolution_level, 0)
        + min(number_of_orbits, 100.0)
        - 1.0e4 * eccentricity
        - 100.0 * (spin1_perp + spin2_perp)
    )

    return SimulationRecord(
        sxs_id=_normalise_sxs_id(raw_id),
        q=q,
        nu=nu,
        chi1z=chi1z,
        chi2z=chi2z,
        chi_eff=chi_eff,
        chi_a=chi_a,
        eccentricity=eccentricity,
        spin1_perp=spin1_perp,
        spin2_perp=spin2_perp,
        number_of_orbits=number_of_orbits,
        resolution_level=resolution_level,
        quality_score=quality_score,
        metadata=dict(raw),
    )


def _rows_from_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [dict(item) for item in data]
    if isinstance(data, dict):
        if "simulations" in data and isinstance(data["simulations"], list):
            return [dict(item) for item in data["simulations"]]
        rows = []
        for key, value in data.items():
            if isinstance(value, dict):
                row = dict(value)
                row.setdefault("sxs_id", key)
                rows.append(row)
        return rows
    raise ValueError(f"Unsupported catalog JSON structure in `{path}`.")


def _rows_from_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _rows_from_sxs_catalog() -> list[dict[str, Any]]:
    import sxs

    catalog = sxs.load("catalog")
    table = getattr(catalog, "table", None)
    if table is not None:
        try:
            return [dict(row) for _, row in table.iterrows()]
        except AttributeError:
            pass
    simulations = getattr(catalog, "simulations", None)
    if isinstance(simulations, dict):
        rows = []
        for key, value in simulations.items():
            row = dict(value)
            row.setdefault("sxs_id", key)
            rows.append(row)
        return rows
    if isinstance(catalog, dict):
        return [dict(value, sxs_id=key) for key, value in catalog.items() if isinstance(value, dict)]
    raise RuntimeError("Could not normalise the object returned by `sxs.load('catalog')`.")


def load_catalog(catalog_file: str | None = None) -> list[SimulationRecord]:
    if catalog_file:
        path = Path(catalog_file)
        rows = _rows_from_json(path) if path.suffix.lower() == ".json" else _rows_from_csv(path)
    else:
        rows = _rows_from_sxs_catalog()

    records = []
    for row in rows:
        record = normalise_catalog_entry(row)
        if record is not None:
            records.append(record)

    records.sort(key=lambda item: item.sxs_id)
    return records


def filter_records(records: Iterable[SimulationRecord], config: CalibrationConfig) -> list[SimulationRecord]:
    family = _normalise_family(config.family)
    if family not in {"nonspinning", "equal-mass-spinning", "spinning"}:
        raise ValueError("`family` must be one of: nonspinning, equal-mass-spinning, spinning.")
    if _is_spin_family(config.family) and not config.base_nonspinning_fit:
        raise ValueError("Spinning TEOBPM calibration requires `base_nonspinning_fit`.")

    selected_ids = set(parse_sxs_ids(config.sxs_ids))
    filtered = []
    for record in records:
        if selected_ids and record.sxs_id not in selected_ids:
            continue
        if record.eccentricity > config.max_eccentricity:
            continue
        if record.spin1_perp > config.max_transverse_spin or record.spin2_perp > config.max_transverse_spin:
            continue
        is_nonspinning = (
            abs(record.chi1z) <= config.nonspinning_spin_tolerance
            and abs(record.chi2z) <= config.nonspinning_spin_tolerance
        )
        if family == "nonspinning":
            if abs(record.chi1z) > config.nonspinning_spin_tolerance:
                continue
            if abs(record.chi2z) > config.nonspinning_spin_tolerance:
                continue
        elif family == "equal-mass-spinning":
            if abs(record.q - 1.0) > config.equal_mass_tolerance and abs(record.nu - 0.25) > config.equal_mass_tolerance:
                continue
            if is_nonspinning:
                continue
        elif family == "spinning":
            if is_nonspinning:
                continue
        if config.min_quality_score is not None and record.quality_score < config.min_quality_score:
            continue
        filtered.append(record)

    return filtered


def apply_random_fraction(records: Iterable[SimulationRecord], random_fraction: float, seed: int) -> list[SimulationRecord]:
    records = list(records)
    if not (0.0 < random_fraction <= 1.0):
        raise ValueError("`random_fraction` must be greater than 0 and at most 1.")
    if random_fraction >= 1.0 or not records:
        return records
    n_selected = max(1, int(math.ceil(len(records) * random_fraction)))
    rng = np.random.default_rng(seed)
    indices = sorted(int(index) for index in rng.choice(len(records), size=n_selected, replace=False))
    return [records[index] for index in indices]


def _bin_key(record: SimulationRecord) -> tuple[int, int]:
    q_bin = int(math.floor(min(record.q, 20.0) / 2.0))
    chi_bin = int(math.floor((record.chi_eff + 1.0) / 0.25))
    return q_bin, chi_bin


def split_train_validation(records: Iterable[SimulationRecord], validation_fraction: float, seed: int) -> tuple[list[SimulationRecord], list[SimulationRecord]]:
    if not (0.0 <= validation_fraction < 1.0):
        raise ValueError("`validation_fraction` must be between 0 and 1, inclusive of 0.")

    if validation_fraction == 0.0:
        training = []
        for record in records:
            copied = SimulationRecord(**asdict(record))
            copied.split = "training"
            training.append(copied)
        return (
            sorted(training, key=lambda item: (item.q, item.chi_eff, item.sxs_id)),
            [],
        )

    grouped: dict[tuple[int, int], list[SimulationRecord]] = {}
    for record in records:
        grouped.setdefault(_bin_key(record), []).append(record)

    training: list[SimulationRecord] = []
    validation: list[SimulationRecord] = []
    for key in sorted(grouped):
        bucket = sorted(grouped[key], key=lambda item: (-item.quality_score, item.q, item.chi_eff, item.sxs_id))
        n_validation = max(1, int(round(len(bucket) * validation_fraction))) if len(bucket) > 1 else 0
        offset = seed % max(len(bucket), 1)
        validation_indices = set()
        if n_validation:
            span = max(len(bucket) - 1, 1)
            for index in np.linspace(1, span, n_validation):
                validation_indices.add((int(round(index)) + offset) % len(bucket))
            if 0 in validation_indices and len(bucket) > n_validation:
                validation_indices.remove(0)
                validation_indices.add(len(bucket) - 1)

        for index, record in enumerate(bucket):
            copied = SimulationRecord(**asdict(record))
            if index in validation_indices:
                copied.split = "validation"
                validation.append(copied)
            else:
                copied.split = "training"
                training.append(copied)

    return (
        sorted(training, key=lambda item: (item.q, item.chi_eff, item.sxs_id)),
        sorted(validation, key=lambda item: (item.q, item.chi_eff, item.sxs_id)),
    )


def write_records(records: Iterable[SimulationRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for record in records]
    path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")


def read_records(path: str | Path) -> list[SimulationRecord]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [SimulationRecord(**row) for row in data]


def _properties_sxs_key(row: dict[str, Any]) -> str:
    raw_id = str(row.get("sxs_id") or row.get("SXS_ID") or row.get("ID") or "").strip()
    if raw_id.startswith("SXS:BBH:"):
        return normalise_sxs_id(raw_id)
    if raw_id.startswith("SXS"):
        return normalise_sxs_id(raw_id)
    return f"SXS:BBH:{raw_id.zfill(4)}"


def _mode_metadata_aliases(column: str, value: Any) -> dict[str, Any]:
    aliases: dict[str, Any] = {column: value}
    if column.startswith("A_peak") and column.endswith("dotdot"):
        suffix = column[len("A_peak") : -len("dotdot")]
        if len(suffix) == 2 and suffix.isdigit():
            aliases[f"A_peak{suffix}dotdot"] = value
        return aliases

    for prefix, canonical_prefixes in (
        ("A_peak", ("A_peak_",)),
        ("omega_peak", ("omega_peak_", "omg_peak_")),
    ):
        if not column.startswith(prefix):
            continue
        suffix = column[len(prefix):]
        if len(suffix) == 2 and suffix.isdigit():
            for canonical_prefix in canonical_prefixes:
                aliases[f"{canonical_prefix}{suffix}"] = value
        return aliases
    return aliases


def attach_properties_metadata(records: Iterable[SimulationRecord], properties_file: str | Path) -> None:
    properties_path = Path(properties_file)
    rows_by_id: dict[str, dict[str, Any]] = {}
    with properties_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            metadata: dict[str, Any] = {}
            for key, value in row.items():
                metadata.update(_mode_metadata_aliases(str(key), value))
            rows_by_id[_properties_sxs_key(row)] = metadata

    missing = []
    for record in records:
        metadata = rows_by_id.get(record.sxs_id)
        if metadata is None:
            missing.append(record.sxs_id)
            continue
        record.metadata.update(metadata)
    if missing:
        examples = ", ".join(missing[:10])
        raise ValueError(f"`{properties_path}` is missing merger metadata for {len(missing)} SXS IDs, e.g. {examples}.")


def _local_prior_text(mode: tuple[int, int], template: str) -> str:
    mode_label = _mode_label(mode)
    if template == "HypTan" and mode == (2, 2):
        bounds = {
            "c3A": (-0.5, -0.1),
            "c3p": (2.0, 7.0),
            "c4p": (0.1, 10.0),
        }
    elif template == "HypTan":
        bounds = {
            "c3A": (-5.0, 1.0),
            "c3p": (0.1, 10.0),
            "c4p": (0.1, 10.0),
        }
    elif template == "RatExp":
        bounds = {
            "c2A": (0.1, 10.0),
            "c3A": (-5.0, 1.0),
            "c2p": (0.1, 10.0),
            "c3p": (0.1, 10.0),
            "c4p": (0.1, 10.0),
        }
    elif template == "SEOBNRv5":
        bounds = {
            "c2A": (1.0e-4, 5.0),
            "c3A": (-5.0, 1.0),
            "c2p": (1.0e-4, 5.0),
            "c3p": (0.1, 10.0),
        }
    else:
        raise ValueError(f"Unknown TEOBPM template `{template}`.")

    lines = []
    for name, (lower, upper) in bounds.items():
        lines.append(f"{name}_{mode_label}-min = {lower:.16g}")
        lines.append(f"{name}_{mode_label}-max = {upper:.16g}")
    return "\n".join(lines)


def _local_counter_rotating_prior_text(mode: tuple[int, int], template: str) -> str:
    counter_mode_label = _mode_label((mode[0], -mode[1]))
    if template == "HypTan":
        bounds = {
            "ln_A_counter_scale": (-8.0, 0.0),
            "phi_mrg_counter": (0.0, 2.0*math.pi),
            "c3A_counter": (-5.0, 1.0),
            "c3p_counter": (0.1, 10.0),
            "c4p_counter": (0.1, 10.0),
        }
    elif template == "RatExp":
        bounds = {
            "ln_A_counter_scale": (-8.0, 0.0),
            "phi_mrg_counter": (0.0, 2.0*math.pi),
            "c2A_counter": (0.1, 10.0),
            "c3A_counter": (-5.0, 1.0),
            "c2p_counter": (0.1, 10.0),
            "c3p_counter": (0.1, 10.0),
            "c4p_counter": (0.1, 10.0),
        }
    elif template == "SEOBNRv5":
        bounds = {
            "ln_A_counter_scale": (-8.0, 0.0),
            "phi_mrg_counter": (0.0, 2.0*math.pi),
            "c2A_counter": (1.0e-4, 5.0),
            "c3A_counter": (-5.0, 1.0),
            "c2p_counter": (1.0e-4, 5.0),
            "c3p_counter": (0.1, 10.0),
        }
    else:
        raise ValueError(f"Unknown TEOBPM template `{template}`.")

    lines = []
    for name, (lower, upper) in bounds.items():
        lines.append(f"{name}_{counter_mode_label}-min = {lower:.16g}")
        lines.append(f"{name}_{counter_mode_label}-max = {upper:.16g}")
    return "\n".join(lines)


def _local_fit_t_start(record: SimulationRecord, mode: tuple[int, int], config: CalibrationConfig) -> float:
    return float(config.t_start)


def local_config_text(record: SimulationRecord, mode: tuple[int, int], config: CalibrationConfig) -> str:
    mode_label = _mode_label(mode)
    outdir = Path(config.output_dir) / "local_fits" / record.sxs_id.replace(":", "_") / f"mode_{mode_label}"
    merger_data = 1 if _teobpm_requires_merger_peak_data(config.template) else 0
    global_fit = 0
    mode_mixing = 1 if mode in config.mode_mixing_modes else 0
    counter_rotating = 1 if mode in config.counter_rotating_modes else 0
    t_start = _local_fit_t_start(record, mode, config)
    prior_text = _local_prior_text(mode, config.template)
    quadratic_modes = QQNM_TERMS_BY_MODE.get(mode, '')
    counter_prior_text = _local_counter_rotating_prior_text(mode, config.template)
    if counter_prior_text:
        prior_text = f"{prior_text}\n{counter_prior_text}"
    return f"""[I/O]
outdir = {outdir}
screen-output = 0
run-type = full

[NR-data]
catalog = SXS
ID = {record.sxs_id.split(":")[-1]}
dir = {config.nr_data_dir}
download = 1
properties-file = {config.properties_file}
extrap-order = {config.extrap_order}
res-level = {config.resolution_level}
l-NR = {mode[0]}
m = {mode[1]}
t-peak-22 = 0.0
error = align-with-mismatch-all
error-t-min = 0.0
error-t-max = 30.0

[Model]
template = TEOBPM
TEOB-template = {config.template}
TEOB-calibration = {config.teob_calibration}
TEOB-global-fit = {global_fit}
TEOB-merger-data = {merger_data}
TEOB-mode-mixing = {mode_mixing}
TEOB-counter-rotating = {counter_rotating}
{ 'QQNM-modes = ' + quadratic_modes if quadratic_modes else '' }

[Priors]
{prior_text}

[Inference]
method = {config.method}
min-method = {config.min_method}
n-random-seeds = {config.n_random_seeds}
t-start = {t_start}
t-end = {config.t_end}
n-mode-workers = {config.n_mode_workers}
"""


def generate_local_fit_configs(records: Iterable[SimulationRecord], config: CalibrationConfig) -> list[Path]:
    config_dir = Path(config.output_dir) / "local_fit_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for record in records:
        for mode in config.modes:
            path = config_dir / f"{record.sxs_id.replace(':', '_')}_mode_{_mode_label(mode)}.ini"
            path.write_text(local_config_text(record, mode, config), encoding="utf-8")
            paths.append(path)
    return paths


def local_fit_jobs(records: Iterable[SimulationRecord], config: CalibrationConfig) -> list[LocalFitJob]:
    jobs = []
    for record in records:
        for mode in config.modes:
            mode_label = _mode_label(mode)
            config_path = Path(config.output_dir) / "local_fit_configs" / f"{record.sxs_id.replace(':', '_')}_mode_{mode_label}.ini"
            outdir = Path(config.output_dir) / "local_fits" / record.sxs_id.replace(":", "_") / f"mode_{mode_label}"
            jobs.append(
                LocalFitJob(
                    sxs_id=record.sxs_id,
                    mode=mode_label,
                    split=record.split or "training",
                    config_file=str(config_path),
                    outdir=str(outdir),
                    q=record.q,
                    nu=record.nu,
                    chi1z=record.chi1z,
                    chi2z=record.chi2z,
                    chi_eff=record.chi_eff,
                    chi_a=record.chi_a,
                    eccentricity=record.eccentricity,
                    quality_score=record.quality_score,
                )
            )
    return jobs


def write_local_fit_index(jobs: Iterable[LocalFitJob], path: Path) -> None:
    rows = [asdict(job) for job in jobs]
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(LocalFitJob.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_local_fit_index(path: str | Path) -> list[LocalFitJob]:
    rows = _read_table(path)
    return [
        LocalFitJob(
            sxs_id=row["sxs_id"],
            mode=row["mode"],
            split=row.get("split", "training"),
            config_file=row["config_file"],
            outdir=row["outdir"],
            q=_as_float(row.get("q")),
            nu=_as_float(row.get("nu")),
            chi1z=_as_float(row.get("chi1z")),
            chi2z=_as_float(row.get("chi2z")),
            chi_eff=_as_float(row.get("chi_eff")),
            chi_a=_as_float(row.get("chi_a")),
            eccentricity=_as_float(row.get("eccentricity")),
            quality_score=_as_float(row.get("quality_score")),
        )
        for row in rows
    ]


def write_run_script(config_paths: Iterable[Path], config: CalibrationConfig) -> Path:
    script = Path(config.output_dir) / "run_local_fits.sh"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for path in config_paths:
        lines.append(f"{config.bayring_executable} --config-file {shlex.quote(str(path))}")
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script.chmod(0o755)
    return script


def prepare_campaign(catalog_file: str | None, config: CalibrationConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_catalog(catalog_file)
    filtered = filter_records(records, config)
    n_before_random_fraction = len(filtered)
    filtered = apply_random_fraction(filtered, config.random_fraction, config.seed)
    if config.properties_file:
        attach_properties_metadata(filtered, config.properties_file)
    training, validation = split_train_validation(filtered, config.validation_fraction, config.seed)
    all_split = training + validation
    jobs = local_fit_jobs(all_split, config)
    config_paths = generate_local_fit_configs(all_split, config)
    run_script = write_run_script(config_paths, config)

    write_records(filtered, output_dir / "manifests" / "filtered_manifest.json")
    write_records(training, output_dir / "manifests" / "training_manifest.json")
    write_records(validation, output_dir / "manifests" / "validation_manifest.json")
    write_records(all_split, output_dir / "manifests" / "dataset_manifest.json")
    write_local_fit_index(jobs, output_dir / "local_fit_index.csv")
    (output_dir / "campaign_config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8")

    return {
        "total_catalog_records": len(records),
        "filtered_records_before_random_fraction": n_before_random_fraction,
        "random_fraction": config.random_fraction,
        "selected_records": len(filtered),
        "training_records": len(training),
        "validation_records": len(validation),
        "local_fit_configs": len(config_paths),
        "indexed_local_fit_jobs": len(jobs),
        "run_script": str(run_script),
    }


def _read_table(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_rows_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _median(values: Iterable[float]) -> float:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return math.nan
    middle = len(finite) // 2
    if len(finite) % 2:
        return float(finite[middle])
    return float(0.5 * (finite[middle - 1] + finite[middle]))


def _campaign_config(campaign_dir: str | Path) -> dict[str, Any]:
    path = Path(campaign_dir) / "campaign_config.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _local_fit_completion_sentinel(outdir: str | Path) -> Path | None:
    algorithm_dir = Path(outdir) / "Algorithm"
    for filename in ("point_estimates.dat", "posterior.dat"):
        path = algorithm_dir / filename
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def run_local_fit_job(job: LocalFitJob, bayring_executable: str = "bayRing", force: bool = False, timeout: float | None = None) -> dict[str, Any]:
    outdir = Path(job.outdir)
    completion_sentinel = _local_fit_completion_sentinel(outdir)
    if completion_sentinel is not None and not force:
        print(f"* Skipping existing local fit {job.sxs_id} mode {job.mode}: found {completion_sentinel.name}.", flush=True)
        return {
            "sxs_id": job.sxs_id,
            "mode": job.mode,
            "config_file": job.config_file,
            "outdir": job.outdir,
            "returncode": 0,
            "status": "skipped-existing",
            "completion_file": str(completion_sentinel),
        }

    outdir.mkdir(parents=True, exist_ok=True)
    log_dir = outdir / "campaign_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    command = [*shlex.split(bayring_executable), "--config-file", job.config_file]
    print(f"* Running local fit {job.sxs_id} mode {job.mode}.", flush=True)
    stdout_path = log_dir / "stdout.txt"
    stderr_path = log_dir / "stderr.txt"
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
            completed = subprocess.run(
                command,
                cwd=os.getcwd(),
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                timeout=timeout,
            )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        if stdout:
            with stdout_path.open("a", encoding="utf-8") as stdout_file:
                stdout_file.write(stdout)
        with stderr_path.open("a", encoding="utf-8") as stderr_file:
            if stderr:
                stderr_file.write(stderr)
            stderr_file.write(f"\nTimed out after {timeout} seconds.\n")
        print(f"* Finished local fit {job.sxs_id} mode {job.mode}: timeout.", flush=True)
        return {
            "sxs_id": job.sxs_id,
            "mode": job.mode,
            "config_file": job.config_file,
            "outdir": job.outdir,
            "returncode": -1,
            "status": "timeout",
        }
    result = {
        "sxs_id": job.sxs_id,
        "mode": job.mode,
        "config_file": job.config_file,
        "outdir": job.outdir,
        "returncode": completed.returncode,
        "status": "completed" if completed.returncode == 0 else "failed",
    }
    print(f"* Finished local fit {job.sxs_id} mode {job.mode}: {result['status']}.", flush=True)
    return result


def _select_jobs(jobs: Iterable[LocalFitJob], split: str = "all", modes: Iterable[tuple[int, int]] | None = None) -> list[LocalFitJob]:
    selected = list(jobs)
    if split not in {"training", "validation"}:
        if split != "all":
            raise ValueError("`split` must be one of: all, training, validation.")
    else:
        selected = [job for job in selected if job.split == split]
    if modes is not None:
        mode_labels = {_mode_label(mode) for mode in modes}
        selected = [job for job in selected if job.mode in mode_labels]
    return selected


def _progress_bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + "-" * width + "] 0/0"
    filled = int(round(width * done / total))
    filled = min(width, max(0, filled))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {done}/{total}"


def _print_local_fit_progress(done: int, total: int, results: list[dict[str, Any]]) -> None:
    counts = {
        "completed": sum(1 for row in results if row.get("status") == "completed"),
        "failed": sum(1 for row in results if row.get("status") == "failed"),
        "timeout": sum(1 for row in results if row.get("status") == "timeout"),
        "skipped": sum(1 for row in results if row.get("status") == "skipped-existing"),
    }
    print(
        "* Progress {bar} completed={completed} skipped={skipped} failed={failed} timeout={timeout}".format(
            bar=_progress_bar(done, total),
            **counts,
        ),
        flush=True,
    )


def run_local_fits(
    campaign_dir: str | Path,
    workers: int = 1,
    bayring_executable: str | None = None,
    force: bool = False,
    timeout: float | None = None,
    split: str = "all",
    modes: Iterable[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    campaign_dir = Path(campaign_dir)
    config = _campaign_config(campaign_dir)
    executable = bayring_executable or config.get("bayring_executable", "bayRing")
    jobs = _select_jobs(read_local_fit_index(campaign_dir / "local_fit_index.csv"), split=split, modes=modes)
    if workers < 1:
        raise ValueError("`workers` must be at least 1.")

    print(f"* Local-fit campaign: {len(jobs)} jobs, split={split}, workers={workers}.", flush=True)
    if not jobs:
        _print_local_fit_progress(0, 0, [])

    results: list[dict[str, Any]] = []
    if workers == 1:
        for index, job in enumerate(jobs, start=1):
            results.append(run_local_fit_job(job, executable, force=force, timeout=timeout))
            _print_local_fit_progress(index, len(jobs), results)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_local_fit_job, job, executable, force, timeout) for job in jobs]
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                results.append(future.result())
                _print_local_fit_progress(index, len(jobs), results)
        results.sort(key=lambda row: (row["sxs_id"], row["mode"]))

    summary_path = campaign_dir / "local_fit_run_summary.csv"
    _write_rows_csv(results, summary_path)
    summary = {
        "jobs": len(results),
        "split": split,
        "modes": sorted({_mode_label(mode) for mode in modes}) if modes is not None else "all",
        "completed": sum(1 for row in results if row["status"] == "completed"),
        "failed": sum(1 for row in results if row["status"] == "failed"),
        "timeout": sum(1 for row in results if row["status"] == "timeout"),
        "skipped_existing": sum(1 for row in results if row["status"] == "skipped-existing"),
        "summary_file": str(summary_path),
    }
    (campaign_dir / "local_fit_run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def read_point_estimates(path: str | Path) -> dict[str, dict[str, float]]:
    estimates: dict[str, dict[str, float]] = {}
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pieces = line.split()
            if len(pieces) < 2:
                continue
            parameter = pieces[0]
            estimates[parameter] = {
                "value": _as_float(pieces[1], default=math.nan),
                "sigma": _as_float(pieces[2], default=math.nan) if len(pieces) > 2 else math.nan,
            }
    return estimates


def read_posterior_estimates(path: str | Path) -> dict[str, dict[str, float]]:
    path = Path(path)
    delimiter = None
    header: list[str] = []
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8") as posterior_file:
        for line in posterior_file:
            stripped = line.strip()
            if stripped:
                stripped = stripped.lstrip("#").strip()
                delimiter = "," if "," in stripped else None
                header = [item.strip() for item in stripped.split(delimiter) if item.strip()] if delimiter else stripped.split()
                break

        for line in posterior_file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            rows.append([item.strip() for item in stripped.split(delimiter) if item.strip()] if delimiter else stripped.split())

    estimates: dict[str, dict[str, float]] = {}
    skipped_names = {"logL", "logPrior", "logL_birth", "log_w", "log_wt", "logwt"}
    for index, name in enumerate(header):
        if name in skipped_names:
            continue
        values = [
            _as_float(row[index], default=math.nan)
            for row in rows
            if index < len(row)
        ]
        values = [value for value in values if math.isfinite(value)]
        if not values:
            continue
        sigma = math.nan
        if len(values) > 1:
            mean = sum(values) / len(values)
            sigma = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
        estimates[name] = {
            "value": _median(values),
            "sigma": sigma,
        }
    return estimates


def read_local_fit_estimates(outdir: str | Path) -> tuple[dict[str, dict[str, float]], str]:
    algorithm_dir = Path(outdir) / "Algorithm"
    point_path = algorithm_dir / "point_estimates.dat"
    if point_path.exists() and point_path.stat().st_size > 0:
        return read_point_estimates(point_path), "point_estimate"
    posterior_path = algorithm_dir / "posterior.dat"
    if posterior_path.exists() and posterior_path.stat().st_size > 0:
        return read_posterior_estimates(posterior_path), "posterior_median"
    return {}, "missing"


def _read_peak_time(path: str | Path) -> float | None:
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            value = _as_float(line, default=math.nan)
            if math.isfinite(value):
                return float(value)
    return None


def _load_generated_ini(path: str | Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(path)
    return parser


def _write_generated_ini(parser: configparser.ConfigParser, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        parser.write(handle)


def fill_higher_mode_inputs(
    campaign_dir: str | Path,
    mode_mixing_modes: Iterable[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Fill generated HM configs with t-peak-22 and parent local-fit values.

    Higher-mode TEOBPM fits need the 22 peak time. Mixed modes also need
    fixed parent-mode phase and local coefficients, because pyRing internally
    constructs the parent multipole before projecting the requested spherical
    mode.
    """

    campaign_dir = Path(campaign_dir)
    jobs = read_local_fit_index(campaign_dir / "local_fit_index.csv")
    mode_mixing_set = set(mode_mixing_modes or [])
    peak_times: dict[str, float] = {}
    estimates_by_job: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for job in jobs:
        peak_time = _read_peak_time(Path(job.outdir) / "Peak_quantities" / "Peak_time.txt")
        if job.mode == "22" and peak_time is not None:
            peak_times[job.sxs_id] = peak_time
        estimates, _ = read_local_fit_estimates(job.outdir)
        if estimates:
            estimates_by_job[(job.sxs_id, job.mode)] = estimates

    configs_updated = 0
    parent_fixes_added = 0
    missing_t_peak = []
    missing_parent_fits = []
    for job in jobs:
        mode = _mode_from_label(job.mode)
        if mode == (2, 2):
            continue
        config_path = Path(job.config_file)
        parser = _load_generated_ini(config_path)
        changed = False

        t_peak_22 = peak_times.get(job.sxs_id)
        if t_peak_22 is None:
            missing_t_peak.append({"sxs_id": job.sxs_id, "mode": job.mode, "config_file": job.config_file})
        else:
            if not parser.has_section("NR-data"):
                parser.add_section("NR-data")
            current = _as_float(parser.get("NR-data", "t-peak-22", fallback="0.0"), default=0.0)
            if abs(current - t_peak_22) > 1.0e-12:
                parser.set("NR-data", "t-peak-22", f"{t_peak_22:.17g}")
                changed = True

        model_uses_parent_mode = (
            mode in mode_mixing_set
            or (mode == (4, 4) and _as_bool(parser.get("Model", "TEOB-quadratic-44", fallback="0")))
        )
        if model_uses_parent_mode:
            parent = TEOB_MODE_MIXING_PARENTS.get(mode)
            if parent is None and mode == (4, 4):
                parent = (2, 2)
            if parent is None:
                missing_parent_fits.append({"sxs_id": job.sxs_id, "mode": job.mode, "reason": "unsupported parent mode"})
            else:
                parent_label = _mode_label(parent)
                parent_estimates = estimates_by_job.get((job.sxs_id, parent_label))
                if not parent_estimates:
                    missing_parent_fits.append({"sxs_id": job.sxs_id, "mode": job.mode, "parent_mode": parent_label})
                else:
                    if not parser.has_section("Priors"):
                        parser.add_section("Priors")
                    for parameter, estimate in sorted(parent_estimates.items()):
                        if not parameter.endswith(f"_{parent_label}"):
                            continue
                        prefix = parameter[: -len(f"_{parent_label}")]
                        if prefix not in {"phi_mrg", "c2A", "c3A", "c2p", "c3p", "c4p"}:
                            continue
                        value = _as_float(estimate.get("value"), default=math.nan)
                        if not math.isfinite(value):
                            continue
                        option = f"fix-{parameter}"
                        current = parser.get("Priors", option, fallback=None)
                        if current is None or abs(_as_float(current, default=math.nan) - value) > 1.0e-12:
                            parser.set("Priors", option, f"{value:.17g}")
                            parent_fixes_added += 1
                            changed = True

        if changed:
            _write_generated_ini(parser, config_path)
            configs_updated += 1

    summary = {
        "campaign_dir": str(campaign_dir),
        "configs_seen": len(jobs),
        "configs_updated": configs_updated,
        "t_peak_22_values": len(peak_times),
        "missing_t_peak": len(missing_t_peak),
        "missing_t_peak_examples": missing_t_peak[:20],
        "parent_fixes_added": parent_fixes_added,
        "missing_parent_fits": len(missing_parent_fits),
        "missing_parent_fit_examples": missing_parent_fits[:20],
    }
    (campaign_dir / "hm_input_fill_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


TEOB_44_ADDITIVE_TARGETS = {
    "ln_A_tapered_441",
    "phi_tapered_441",
    *TEOB_44_WINDOW_TARGETS,
}


def _target_from_parameter(parameter: str, mode: str) -> str | None:
    if mode == "44" and parameter in TEOB_44_ADDITIVE_TARGETS:
        return parameter
    suffix = f"_{mode}"
    if not parameter.endswith(suffix):
        return None
    target = parameter[: -len(suffix)]
    if target in {"c2A", "c3A", "c2p", "c3p", "c4p", *REFERENCE_FIT_TARGETS}:
        return target
    return None


def _wrap_phase(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _local_fit_targets_for_template(template: str) -> tuple[str, ...]:
    if template == "HypTan":
        return LOCAL_FIT_TARGETS_HYPTAN + TEOB_44_WINDOW_TARGETS
    if template == "RatExp":
        return LOCAL_FIT_TARGETS_RATEXP + TEOB_44_WINDOW_TARGETS
    if template == "SEOBNRv5":
        return LOCAL_FIT_TARGETS_SEOBNRV5 + TEOB_44_WINDOW_TARGETS
    raise ValueError(f"Unknown TEOBPM template `{template}`.")


def _record_metadata_targets(record: SimulationRecord | None, job: LocalFitJob, template: str) -> list[dict[str, Any]]:
    if record is None:
        return []
    rows: list[dict[str, Any]] = []
    metadata = record.metadata or {}
    mode = job.mode
    target_keys = {
        "omg_peak": [f"omg_peak_{mode}", f"omega_peak_{mode}", f"peak-omega-l{mode[0]}-m{mode[1:]}"],
        "A_peak_over_nu": [f"A_peak_{mode}", f"A_peak{mode}", f"peak-ampl-l{mode[0]}-m{mode[1:]}"],
        "A_peakdotdot_over_nu": [f"A_peak{mode}dotdot"],
        "DeltaT": [f"DeltaT_{mode}", f"DeltaT{mode}"],
    }
    allowed_targets = set(_local_fit_targets_for_template(template))
    for target, keys in target_keys.items():
        if target not in allowed_targets:
            continue
        raw_value = _first_present(metadata, keys, None)
        if raw_value is None:
            continue
        value = _as_float(raw_value, default=math.nan)
        if target.endswith("_over_nu") and job.nu != 0.0:
            value = value / job.nu
        if math.isfinite(value):
            rows.append(_local_fit_row(job, target, value, math.nan, source="metadata"))
    return rows


def _computed_merger_targets_from_job(job: LocalFitJob, template: str) -> list[dict[str, Any]]:
    path = Path(job.outdir) / "Peak_quantities" / "Merger_metadata.json"
    if not path.exists():
        return []
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    mode = job.mode
    target_keys = {
        "omg_peak": [f"omg_peak_{mode}", f"omega_peak_{mode}"],
        "A_peak_over_nu": [f"A_peak_over_nu_{mode}", f"A_peak_{mode}", f"A_peak{mode}"],
        "A_peakdot_over_nu": [f"A_peakdot_over_nu_{mode}", f"A_peak{mode}dot"],
        "A_peakdotdot_over_nu": [f"A_peakdotdot_over_nu_{mode}", f"A_peak{mode}dotdot"],
        "DeltaT": [f"DeltaT_{mode}", f"DeltaT{mode}"],
    }
    allowed_targets = set(_local_fit_targets_for_template(template))
    rows: list[dict[str, Any]] = []
    for target, keys in target_keys.items():
        if target not in allowed_targets:
            continue
        raw_value = _first_present(metadata, keys, None)
        if raw_value is None:
            continue
        value = _as_float(raw_value, default=math.nan)
        if target.endswith("_over_nu") and keys[0] not in metadata and job.nu != 0.0:
            value = value / job.nu
        if math.isfinite(value):
            rows.append(_local_fit_row(job, target, value, math.nan, source="NR_sim"))
    return rows


def _fixed_prior_targets_from_config(job: LocalFitJob, template: str) -> list[dict[str, Any]]:
    config_path = Path(job.config_file)
    if not config_path.exists():
        return []
    parser = _load_generated_ini(config_path)
    if not parser.has_section("Priors"):
        return []

    rows: list[dict[str, Any]] = []
    allowed_targets = set(_local_fit_targets_for_template(template))
    for target in REFERENCE_FIT_TARGETS:
        if target not in allowed_targets:
            continue
        value = _as_float(parser.get("Priors", f"fix-{target}_{job.mode}", fallback=None), default=math.nan)
        if math.isfinite(value):
            rows.append(_local_fit_row(job, target, value, math.nan, source="fixed_prior"))
    if job.mode == "44":
        for target in TEOB_44_WINDOW_TARGETS:
            if target not in allowed_targets:
                continue
            value = _as_float(parser.get("Priors", f"fix-{target}", fallback=None), default=math.nan)
            if math.isfinite(value):
                rows.append(_local_fit_row(job, target, value, math.nan, source="fixed_prior"))
    return rows


def _local_fit_row(
    job: LocalFitJob,
    target: str,
    value: float,
    sigma: float,
    source: str,
    construction_mismatch: float | None = None,
) -> dict[str, Any]:
    row = {
        "sxs_id": job.sxs_id,
        "split": job.split,
        "mode": job.mode,
        "target": target,
        "value": value,
        "sigma": sigma,
        "source": source,
        "q": job.q,
        "nu": job.nu,
        "chi1z": job.chi1z,
        "chi2z": job.chi2z,
        "chi_eff": job.chi_eff,
        "chi_a": job.chi_a,
        "eccentricity": job.eccentricity,
        "quality_score": job.quality_score,
        "outdir": job.outdir,
    }
    if construction_mismatch is not None:
        row["construction_mismatch"] = construction_mismatch
    return row


def read_mismatch_file(path: str | Path, job: LocalFitJob | None = None) -> list[dict[str, Any]]:
    path = Path(path)
    rows = []
    header: list[str] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                header = [item.strip() for item in line[1:].split()]
                continue
            pieces = line.split()
            if len(pieces) < 2:
                continue
            row: dict[str, Any] = {"file": str(path)}
            if job is not None:
                row.update({"sxs_id": job.sxs_id, "mode": job.mode, "split": job.split})
            if header and len(header) == len(pieces):
                for key, value in zip(header, pieces):
                    row[key] = value
            else:
                row["CI"] = pieces[0]
                row["quantity"] = pieces[1] if len(pieces) > 2 else ""
                row["Mismatch"] = pieces[-1]
            if "Mismatch" in row and row.get("Mismatch") not in (None, ""):
                row["mismatch"] = _as_float(row.get("Mismatch"), default=math.nan)
            elif not header:
                row["mismatch"] = _as_float(pieces[-1], default=math.nan)
            else:
                row["mismatch"] = math.nan
            row["CI"] = _as_float(row.get("CI", pieces[0]), default=math.nan)
            rows.append(row)
    return rows


def read_mismatch_diagnostics_file(path: str | Path, job: LocalFitJob | None = None) -> list[dict[str, Any]]:
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for source_row in reader:
            if source_row.get("mismatch") in (None, ""):
                continue
            row: dict[str, Any] = {"file": str(path), **source_row}
            if job is not None:
                row.update({"sxs_id": job.sxs_id, "mode": job.mode, "split": job.split})
            row["CI"] = _as_float(source_row.get("confidence_interval"), default=math.nan)
            row["Strain_data"] = source_row.get("strain_data", "")
            row["Mismatch"] = source_row.get("mismatch", "")
            row["mismatch"] = _as_float(source_row.get("mismatch"), default=math.nan)
            rows.append(row)
    return rows


def collect_mismatch_rows(job: LocalFitJob) -> list[dict[str, Any]]:
    mismatch_dir = Path(job.outdir) / "Algorithm" / "Mismatch"
    if not mismatch_dir.exists():
        return []
    rows = []
    diagnostics_path = mismatch_dir / "mismatch_and_snr_diagnostics.tsv"
    if diagnostics_path.exists():
        rows.extend(read_mismatch_diagnostics_file(diagnostics_path, job=job))
    for path in sorted(mismatch_dir.glob("*.txt")):
        rows.extend(read_mismatch_file(path, job=job))
    return [row for row in rows if math.isfinite(_as_float(row.get("mismatch"), default=math.nan))]


def representative_mismatch(rows: list[dict[str, Any]]) -> float | None:
    ci50_rows = [
        row
        for row in rows
        if math.isfinite(_as_float(row.get("CI"), default=math.nan))
        and abs(_as_float(row.get("CI"), default=math.nan) - 50.0) < 1.0e-9
        and math.isfinite(_as_float(row.get("mismatch"), default=math.nan))
    ]
    by_strain = {
        str(row.get("Strain_data") or row.get("quantity") or "").lower(): _as_float(row.get("mismatch"), default=math.nan)
        for row in ci50_rows
    }
    if math.isfinite(by_strain.get("real", math.nan)) and math.isfinite(by_strain.get("imag", math.nan)):
        return float(math.hypot(by_strain["real"], by_strain["imag"]))

    values = [_as_float(row.get("mismatch"), default=math.nan) for row in ci50_rows]
    if not values:
        values = [_as_float(row.get("mismatch"), default=math.nan) for row in rows if math.isfinite(_as_float(row.get("mismatch"), default=math.nan))]
    if not values:
        return None
    return float(sum(values) / len(values))


def collect_local_fit_outputs(campaign_dir: str | Path, output_table: str | Path | None = None, split: str = "all") -> dict[str, Any]:
    campaign_dir = Path(campaign_dir)
    output_table = Path(output_table) if output_table else campaign_dir / "local_fit_summary.csv"
    template = _campaign_config(campaign_dir).get("template", "RatExp")
    jobs = _select_jobs(read_local_fit_index(campaign_dir / "local_fit_index.csv"), split=split)
    manifest_path = campaign_dir / "manifests" / "dataset_manifest.json"
    records_by_id = {record.sxs_id: record for record in read_records(manifest_path)} if manifest_path.exists() else {}

    rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    phases: dict[tuple[str, str], dict[str, Any]] = {}

    for job in jobs:
        job_mismatch_rows = collect_mismatch_rows(job)
        mismatch_rows.extend(job_mismatch_rows)
        mismatch = representative_mismatch(job_mismatch_rows)
        estimates, estimate_source = read_local_fit_estimates(job.outdir)
        fixed_prior_rows = _fixed_prior_targets_from_config(job, template)
        computed_merger_rows = _computed_merger_targets_from_job(job, template)
        computed_merger_targets = {row["target"] for row in computed_merger_rows}

        if not estimates:
            failures.append({"sxs_id": job.sxs_id, "mode": job.mode, "outdir": job.outdir, "reason": "missing point_estimates.dat or posterior.dat"})
            rows.extend(row for row in fixed_prior_rows if row["target"] not in computed_merger_targets)
            rows.extend(computed_merger_rows)
            rows.extend(row for row in _record_metadata_targets(records_by_id.get(job.sxs_id), job, template) if row["target"] not in computed_merger_targets)
            continue

        estimated_targets: set[str] = set()
        for parameter, estimate in estimates.items():
            target = _target_from_parameter(parameter, job.mode)
            if target is None:
                continue
            if target == "phi_mrg":
                phases[(job.sxs_id, job.mode)] = {"job": job, **estimate, "construction_mismatch": mismatch}
            else:
                estimated_targets.add(target)
                rows.append(
                    _local_fit_row(
                        job,
                        target,
                        estimate["value"],
                        estimate["sigma"],
                        source=estimate_source,
                        construction_mismatch=mismatch,
                    )
                )
        rows.extend(row for row in fixed_prior_rows if row["target"] not in estimated_targets and row["target"] not in computed_merger_targets)
        rows.extend(computed_merger_rows)
        rows.extend(row for row in _record_metadata_targets(records_by_id.get(job.sxs_id), job, template) if row["target"] not in estimated_targets and row["target"] not in computed_merger_targets)

    for (sxs_id, mode), phase_data in sorted(phases.items()):
        if mode == "22":
            continue
        reference = phases.get((sxs_id, "22"))
        if reference is None:
            failures.append({"sxs_id": sxs_id, "mode": mode, "outdir": phase_data["job"].outdir, "reason": "missing reference phi_mrg_22"})
            continue
        delta = _wrap_phase(phase_data["value"] - reference["value"])
        sigma = math.nan
        if math.isfinite(_as_float(phase_data.get("sigma"), default=math.nan)) and math.isfinite(_as_float(reference.get("sigma"), default=math.nan)):
            sigma = math.sqrt(phase_data["sigma"] ** 2 + reference["sigma"] ** 2)
        rows.append(
            _local_fit_row(
                phase_data["job"],
                f"{PHASE_TARGET_PREFIX}{mode}",
                delta,
                sigma,
                source="relative_phase",
                construction_mismatch=phase_data.get("construction_mismatch"),
            )
        )

    _write_rows_csv(rows, output_table)
    _write_rows_csv(mismatch_rows, campaign_dir / "mismatch_summary.csv")
    _write_rows_csv(failures, campaign_dir / "local_fit_collection_failures.csv")
    summary = {
        "jobs": len(jobs),
        "split": split,
        "rows": len(rows),
        "mismatch_rows": len(mismatch_rows),
        "failures": len(failures),
        "output_table": str(output_table),
        "mismatch_table": str(campaign_dir / "mismatch_summary.csv"),
        "failures_table": str(campaign_dir / "local_fit_collection_failures.csv"),
    }
    (campaign_dir / "local_fit_collection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _term_power_name(variable: str, power: int) -> str:
    return variable if power == 1 else f"{variable}^{power}"


def _evaluate_term(term_name: str, row: dict[str, Any]) -> float:
    if term_name == "1":
        return 1.0
    variables = {
        "nu": _as_float(row.get("nu")),
        "chi_eff": _as_float(row.get("chi_eff")),
        "chi_a": _as_float(row.get("chi_a")),
    }
    value = 1.0
    for factor in term_name.split("*"):
        if "^" in factor:
            variable, raw_power = factor.split("^", 1)
            power = int(raw_power)
        else:
            variable = factor
            power = 1
        value *= variables[variable] ** power
    return value


def _term_values(term_names: Iterable[str], row: dict[str, Any]) -> list[float]:
    return [_evaluate_term(term, row) for term in term_names]


def evaluate_terms(terms: Iterable[dict[str, float]], row: dict[str, Any]) -> float:
    return sum(float(term["coefficient"]) * _term_values([term["name"]], row)[0] for term in terms)


def _candidate_bases(family: str, max_polynomial_degree: int = 3) -> list[list[str]]:
    max_degree = max(0, int(max_polynomial_degree))
    if _normalise_family(family) == "nonspinning":
        bases = []
        for degree in range(max_degree + 1):
            terms = ["1"]
            terms.extend(_term_power_name("nu", power) for power in range(1, degree + 1))
            bases.append(terms)
        return bases

    spin_terms_by_degree = [
        (1, "chi_eff"),
        (1, "chi_a"),
        (2, "nu*chi_eff"),
        (2, "nu*chi_a"),
        (2, "chi_eff^2"),
        (2, "chi_a^2"),
    ]
    bases = []
    for degree in range(1, max_degree + 1):
        terms = [term for term_degree, term in spin_terms_by_degree if term_degree <= degree]
        if terms and terms not in bases:
            bases.append(terms)
    return bases


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    augmented = [list(matrix[row]) + [vector[row]] for row in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1.0e-14:
            augmented[pivot][col] = 1.0e-14
        if pivot != col:
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        pivot_value = augmented[col][col]
        for item in range(col, n + 1):
            augmented[col][item] /= pivot_value
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            for item in range(col, n + 1):
                augmented[row][item] -= factor * augmented[col][item]
    return [augmented[row][n] for row in range(n)]


def _least_squares(design: list[list[float]], values: list[float]) -> list[float]:
    n_terms = len(design[0])
    ata = [[0.0 for _ in range(n_terms)] for _ in range(n_terms)]
    atb = [0.0 for _ in range(n_terms)]
    for row, value in zip(design, values):
        for i in range(n_terms):
            atb[i] += row[i] * value
            for j in range(n_terms):
                ata[i][j] += row[i] * row[j]
    for i in range(n_terms):
        ata[i][i] += 1.0e-12
    return _solve_linear_system(ata, atb)


def _fit_basis(rows: list[dict[str, Any]], target: str, term_names: list[str], base_terms: list[dict[str, float]] | None = None) -> dict[str, Any]:
    y = [_as_float(row["value"]) for row in rows]
    if base_terms:
        offset = [evaluate_terms(base_terms, row) for row in rows]
        y_fit = [value - base for value, base in zip(y, offset)]
    else:
        y_fit = y
    design = [_term_values(term_names, row) for row in rows]
    coeffs = _least_squares(design, y_fit)
    predictions = [sum(row[i] * coeffs[i] for i in range(len(coeffs))) for row in design]
    residual = [value - prediction for value, prediction in zip(y_fit, predictions)]
    n = max(len(y), 1)
    k = len(term_names)
    rss = float(sum(value * value for value in residual))
    sigma2 = max(rss / n, 1.0e-300)
    bic = n * math.log(sigma2) + k * math.log(n)
    aic = n * math.log(sigma2) + 2 * k
    terms = [{"name": name, "coefficient": float(value)} for name, value in zip(term_names, coeffs)]
    if base_terms:
        terms = list(base_terms) + terms
    return {
        "target": target,
        "terms": terms,
        "spin_correction_terms": [{"name": name, "coefficient": float(value)} for name, value in zip(term_names, coeffs)] if base_terms else [],
        "rss": rss,
        "aic": float(aic),
        "bic": float(bic),
        "n": int(n),
    }


def _load_global_fit(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _base_terms_for(base_fit: dict[str, Any] | None, mode_label: str, target: str) -> list[dict[str, float]]:
    if base_fit is None:
        return []
    try:
        return list(base_fit["fits"][mode_label][target]["terms"])
    except KeyError:
        return []


def construct_global_fit(
    local_fit_table: str | Path,
    output_file: str | Path,
    family: str,
    template: str = "RatExp",
    teob_calibration: str = "qc",
    base_nonspinning_file: str | None = None,
    max_polynomial_degree: int = 3,
) -> dict[str, Any]:
    if _is_spin_family(family) and not base_nonspinning_file:
        raise ValueError("Spinning global fits require a nonspinning base fit file.")
    base_fit = _load_global_fit(base_nonspinning_file) if base_nonspinning_file else None
    rows = _read_table(local_fit_table)
    allowed_targets = set(_local_fit_targets_for_template(template))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("split", "training") != "training":
            continue
        mode = row.get("mode") or row.get("lm") or row.get("mode_label")
        target = row.get("target") or row.get("parameter")
        if target not in allowed_targets and not str(target).startswith(PHASE_TARGET_PREFIX):
            continue
        if not mode or not target or row.get("value") in (None, ""):
            continue
        grouped.setdefault((str(mode), str(target)), []).append(row)

    fit_payload: dict[str, Any] = {
        "schema": FIT_SCHEMA,
        "family": family,
        "template": template,
        "calibration": teob_calibration,
        "base_nonspinning_file": str(base_nonspinning_file or ""),
        "max_polynomial_degree": int(max_polynomial_degree),
        "fits": {},
        "relative_phase_reference": "22",
        "notes": "Spinning fits are constructed as spin corrections on top of the nonspinning terms when a base file is provided.",
    }
    for (mode_label, target), target_rows in sorted(grouped.items()):
        if len(target_rows) < 2:
            continue
        base_terms = _base_terms_for(base_fit, mode_label, target)
        candidates = []
        for term_names in _candidate_bases(family, max_polynomial_degree=max_polynomial_degree):
            if len(target_rows) < len(term_names):
                continue
            candidates.append(_fit_basis(target_rows, target, term_names, base_terms=base_terms))
        if not candidates:
            continue
        best = min(candidates, key=lambda item: (item["bic"], item["aic"], len(item["terms"])))
        fit_payload["fits"].setdefault(mode_label, {})[target] = best

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fit_payload, indent=2, sort_keys=True), encoding="utf-8")
    return fit_payload


def _prediction_rows(validation_table: str | Path, fit_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _read_table(validation_table)
    has_validation_split = any(row.get("split") == "validation" for row in rows)
    predictions = []
    for row in rows:
        if has_validation_split and row.get("split") != "validation":
            continue
        mode = str(row.get("mode") or row.get("lm") or row.get("mode_label"))
        target = str(row.get("target") or row.get("parameter"))
        try:
            fit = fit_payload["fits"][mode][target]
        except KeyError:
            continue
        prediction = evaluate_terms(fit["terms"], row)
        value = _as_float(row.get("value"))
        predictions.append({
            **row,
            "mode": mode,
            "target": target,
            "value": value,
            "prediction": prediction,
            "residual": value - prediction,
            "abs_residual": abs(value - prediction),
        })
    return predictions


def _write_prediction_csv(rows: list[dict[str, Any]], path: Path) -> None:
    _write_rows_csv(rows, path)


MISMATCH_VALUE_COLUMNS = ("mismatch", "Mismatch", "representative_mismatch", "waveform_mismatch")
MISMATCH_WIDE_COLUMNS = (
    ("new_global_mismatch", "New global fit"),
    ("new_mismatch", "New global fit"),
    ("existing_teobpm_mismatch", "Existing TEOBPM global fit"),
    ("current_teobpm_mismatch", "Existing TEOBPM global fit"),
    ("current_mismatch", "Existing TEOBPM global fit"),
    ("reference_mismatch", "Reference fit"),
    ("past_mismatch", "Past fit"),
    ("local_mismatch", "Local fit"),
)
MISMATCH_T_START_COLUMNS = ("t_start", "t-start", "comparison_t_start", "analysis_t_start", "window_start")
MISMATCH_REFERENCE_COLUMNS = ("tref", "t_ref", "reference_time", "reference_peak", "comparison_reference")
MISMATCH_PEAK22_REFERENCE_VALUES = {
    "0",
    "0.0",
    "22peak",
    "peak22",
    "peak-22",
    "peak_22",
    "22-peak",
    "22_peak",
    "tpeak22",
    "t-peak-22",
    "t_peak_22",
    "mode22-peak",
    "mode22_peak",
}
MISMATCH_HISTOGRAM_TEMPLATE_COLORS = {
    "hyptan_ratexp_nr": "#6f4bb1",
    "hyptan": "#6f4bb1",
    "ratexp": "#d55e00",
    "seobnrv5": "#0072b2",
}
MISMATCH_HISTOGRAM_FAMILY_ORDER = {
    "nonspinning": 0,
    "equal_mass_spinning": 1,
    "generic_spinning": 2,
}
MISMATCH_HISTOGRAM_TEMPLATE_ORDER = {
    "hyptan_ratexp_nr": 0,
    "hyptan": 0,
    "ratexp": 1,
    "seobnrv5": 2,
}


def _first_finite(row: dict[str, Any], columns: Iterable[str], default: float = math.nan) -> float:
    for column in columns:
        value = row.get(column)
        if value in (None, ""):
            continue
        parsed = _as_float(value, default=math.nan)
        if math.isfinite(parsed):
            return parsed
    return default


def _normalised_reference_text(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "").replace("__", "_")


def _check_peak22_mismatch_window(table_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    t_start_rows = 0
    reference_rows = 0
    for row in rows:
        for column in MISMATCH_T_START_COLUMNS:
            raw_value = row.get(column)
            if raw_value in (None, ""):
                continue
            value = _as_float(raw_value, default=math.nan)
            if not math.isfinite(value):
                normalised = _normalised_reference_text(raw_value)
                if normalised in MISMATCH_PEAK22_REFERENCE_VALUES:
                    t_start_rows += 1
                    break
                raise ValueError(
                    f"Mismatch table `{table_path}` declares nonnumeric {column}={raw_value!r}. "
                    "Evaluation comparisons must declare the 22-peak reference or numeric t_start=0."
                )
            t_start_rows += 1
            if abs(value) > 1.0e-9:
                raise ValueError(
                    f"Mismatch table `{table_path}` declares {column}={value}. "
                    "Evaluation comparisons must start at the 22 peak, so use t_start=0."
                )
            break

        for column in MISMATCH_REFERENCE_COLUMNS:
            raw_value = row.get(column)
            if raw_value in (None, ""):
                continue
            value = _as_float(raw_value, default=math.nan)
            if math.isfinite(value):
                if abs(value) > 1.0e-9:
                    raise ValueError(
                        f"Mismatch table `{table_path}` declares {column}={value}. "
                        "Evaluation comparisons must use tref=peak22 or numeric tref=0."
                    )
                reference_rows += 1
                break
            normalised = _normalised_reference_text(raw_value)
            if normalised not in MISMATCH_PEAK22_REFERENCE_VALUES:
                raise ValueError(
                    f"Mismatch table `{table_path}` declares {column}={raw_value!r}. "
                    "Evaluation comparisons must use the 22-peak reference."
                )
            reference_rows += 1
            break

    status = "verified" if (t_start_rows or reference_rows) else "unverified"
    return {
        "table": str(table_path),
        "expected_reference": "peak22",
        "expected_t_start": 0.0,
        "rows": len(rows),
        "rows_with_t_start_metadata": t_start_rows,
        "rows_with_reference_metadata": reference_rows,
        "status": status,
    }


def _nu_from_row(row: dict[str, Any]) -> float:
    nu = _first_finite(row, ("nu", "symmetric_mass_ratio", "eta"))
    if math.isfinite(nu):
        return nu
    x_name = str(row.get("x_name", "")).strip().lower()
    if x_name == "nu":
        return _as_float(row.get("x_value"), default=math.nan)
    q = _as_float(row.get("q"), default=math.nan)
    if math.isfinite(q) and q > 0.0:
        return symmetric_mass_ratio_from_q(q)
    return math.nan


def q_from_symmetric_mass_ratio(nu: float) -> float:
    if not math.isfinite(nu) or nu <= 0.0 or nu > 0.25:
        return math.nan
    discriminant = max(1.0 - 4.0 * nu, 0.0)
    return (1.0 - 2.0 * nu + math.sqrt(discriminant)) / (2.0 * nu)


def _q_from_row(row: dict[str, Any]) -> float:
    q = _first_finite(row, ("q", "mass_ratio", "reference_mass_ratio"))
    if math.isfinite(q) and q > 0.0:
        return q
    x_name = str(row.get("x_name", "")).strip().lower()
    if x_name in {"q", "mass_ratio"}:
        return _as_float(row.get("x_value"), default=math.nan)
    return q_from_symmetric_mass_ratio(_nu_from_row(row))


def _chi_eff_from_row(row: dict[str, Any]) -> float:
    chi_eff = _first_finite(row, ("chi_eff", "chieff", "chi-effective"))
    if math.isfinite(chi_eff):
        return chi_eff
    x_name = str(row.get("x_name", "")).strip().lower().replace("-", "_")
    if x_name in {"chi_eff", "chieff", "s_hat", "shat"}:
        return _as_float(row.get("x_value"), default=math.nan)
    q = _as_float(row.get("q"), default=math.nan)
    chi1z = _as_float(row.get("chi1z"), default=math.nan)
    chi2z = _as_float(row.get("chi2z"), default=math.nan)
    if math.isfinite(q) and q > 0.0 and math.isfinite(chi1z) and math.isfinite(chi2z):
        return (q * chi1z + chi2z) / (1.0 + q)
    return math.nan


def _chi_a_from_row(row: dict[str, Any]) -> float:
    chi_a = _first_finite(row, ("chi_a", "chia", "chi-antisymmetric"))
    if math.isfinite(chi_a):
        return chi_a
    x_name = str(row.get("x_name", "")).strip().lower().replace("-", "_")
    if x_name in {"chi_a", "chia", "a_12", "a12"}:
        return _as_float(row.get("x_value"), default=math.nan)
    chi1z = _as_float(row.get("chi1z"), default=math.nan)
    chi2z = _as_float(row.get("chi2z"), default=math.nan)
    if math.isfinite(chi1z) and math.isfinite(chi2z):
        return 0.5 * (chi1z - chi2z)
    return math.nan


def _mismatch_family_from_row(row: dict[str, Any]) -> str:
    raw_family = str(row.get("family") or row.get("group") or "").strip().lower().replace("_", "-")
    if raw_family in {"nonspinning", "non-spinning"}:
        return "nonspinning"
    if raw_family in {"aligned-spin", "spinning", "equal-mass", "aligned"}:
        return "spinning"

    spin_values = [
        _as_float(row.get("chi_eff"), default=math.nan),
        _as_float(row.get("chieff"), default=math.nan),
        _as_float(row.get("chi_a"), default=math.nan),
        _as_float(row.get("chi1z"), default=math.nan),
        _as_float(row.get("chi2z"), default=math.nan),
    ]
    finite_spins = [abs(value) for value in spin_values if math.isfinite(value)]
    if finite_spins and max(finite_spins) <= 1.0e-6:
        return "nonspinning"
    return "spinning"


def _mismatch_series_label(row: dict[str, Any], source_label: str) -> str:
    label = str(
        row.get("comparison_label")
        or row.get("model_label")
        or row.get("fit_label")
        or row.get("model")
        or row.get("fit")
        or source_label
    ).strip()
    if not label:
        label = "Mismatch"
    template = str(row.get("template") or "").strip()
    if template and template.lower() not in label.lower():
        label = f"{label} {template}"
    return label


def _expanded_mismatch_rows(row: dict[str, Any], source_label: str) -> list[dict[str, Any]]:
    wide_rows = []
    for column, label in MISMATCH_WIDE_COLUMNS:
        value = _as_float(row.get(column), default=math.nan)
        if math.isfinite(value):
            wide_rows.append({**row, "mismatch": value, "comparison_label": label})
    if wide_rows:
        return wide_rows

    value = _first_finite(row, MISMATCH_VALUE_COLUMNS)
    if not math.isfinite(value):
        return []
    return [{**row, "mismatch": value, "comparison_label": row.get("comparison_label") or source_label}]


def _representative_tabular_mismatch(rows: list[dict[str, Any]]) -> float:
    ci50_rows = [
        row for row in rows
        if abs(_as_float(row.get("CI"), default=math.nan) - 50.0) < 1.0e-9
        and math.isfinite(_as_float(row.get("mismatch"), default=math.nan))
    ]
    by_strain = {
        str(row.get("Strain_data") or row.get("quantity") or "").strip().lower(): _as_float(row.get("mismatch"), default=math.nan)
        for row in ci50_rows
    }
    if math.isfinite(by_strain.get("real", math.nan)) and math.isfinite(by_strain.get("imag", math.nan)):
        return math.sqrt(by_strain["real"] ** 2 + by_strain["imag"] ** 2)
    if ci50_rows:
        return _median(_as_float(row.get("mismatch"), default=math.nan) for row in ci50_rows)
    return _median(_as_float(row.get("mismatch"), default=math.nan) for row in rows)


def _normalise_mismatch_rows(rows: list[dict[str, Any]], source_label: str, family_hint: str | None = None) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str, str, str, str, str], list[dict[str, Any]]] = {}
    representative_metadata: dict[tuple[str, str, str, str, str, str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        for expanded in _expanded_mismatch_rows(row, source_label):
            nu = _nu_from_row(expanded)
            q = _q_from_row(expanded)
            chi_eff = _chi_eff_from_row(expanded)
            chi_a = _chi_a_from_row(expanded)
            family = family_hint or _mismatch_family_from_row(expanded)
            label = _mismatch_series_label(expanded, source_label)
            if not math.isfinite(nu) or not math.isfinite(q):
                continue
            if family == "spinning" and not math.isfinite(chi_eff):
                continue
            if family == "nonspinning" and not math.isfinite(chi_eff):
                chi_eff = 0.0
            if family == "nonspinning" and not math.isfinite(chi_a):
                chi_a = 0.0
            key = (
                label,
                family,
                str(expanded.get("template") or ""),
                str(expanded.get("sxs_id") or expanded.get("simulation") or expanded.get("simulation_id") or ""),
                str(expanded.get("mode") or expanded.get("lm") or ""),
                f"{nu:.14g}",
                f"{q:.14g}",
                f"{chi_eff:.14g}",
                f"{chi_a:.14g}",
                str(expanded.get("split") or ""),
            )
            grouped.setdefault(key, []).append(expanded)
            representative_metadata.setdefault(key, expanded)

    normalised = []
    for key, group_rows in sorted(grouped.items()):
        label, family, template, sxs_id, mode, nu_text, q_text, chi_eff_text, chi_a_text, split = key
        mismatch = _representative_tabular_mismatch(group_rows)
        if not math.isfinite(mismatch) or mismatch <= 0.0:
            continue
        first = representative_metadata[key]
        normalised.append({
            "label": label,
            "family": family,
            "template": template,
            "sxs_id": sxs_id,
            "mode": mode,
            "split": split,
            "nu": float(nu_text),
            "q": float(q_text),
            "chi_eff": float(chi_eff_text),
            "chi_a": float(chi_a_text),
            "mismatch": float(mismatch),
            "n_aggregated_rows": len(group_rows),
            "source": source_label,
            "group": first.get("group", ""),
        })
    return normalised


def _mismatch_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["family"], row["label"]), []).append(row)
    summary_rows = []
    for (family, label), label_rows in sorted(grouped.items()):
        mismatches = [float(row["mismatch"]) for row in label_rows]
        nu_values = [float(row["nu"]) for row in label_rows]
        q_values = [float(row["q"]) for row in label_rows]
        chi_values = [float(row["chi_eff"]) for row in label_rows]
        chi_a_values = [float(row["chi_a"]) for row in label_rows if math.isfinite(float(row["chi_a"]))]
        summary_rows.append({
            "family": family,
            "label": label,
            "n": len(label_rows),
            "mismatch_min": min(mismatches),
            "mismatch_median": _median(mismatches),
            "mismatch_max": max(mismatches),
            "nu_min": min(nu_values),
            "nu_max": max(nu_values),
            "q_min": min(q_values),
            "q_max": max(q_values),
            "chi_eff_min": min(chi_values),
            "chi_eff_max": max(chi_values),
            "chi_a_min": min(chi_a_values) if chi_a_values else math.nan,
            "chi_a_max": max(chi_a_values) if chi_a_values else math.nan,
        })
    return summary_rows


def _percentile(values: Iterable[float], percentile: float) -> float:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return math.nan
    if len(finite) == 1:
        return finite[0]
    rank = (len(finite) - 1) * percentile / 100.0
    lower_index = int(math.floor(rank))
    upper_index = int(math.ceil(rank))
    if lower_index == upper_index:
        return finite[lower_index]
    lower_weight = upper_index - rank
    upper_weight = rank - lower_index
    return float(finite[lower_index] * lower_weight + finite[upper_index] * upper_weight)


def _finite_mismatch_limits(rows: list[dict[str, Any]]) -> tuple[float, float]:
    values = [float(row["mismatch"]) for row in rows if math.isfinite(float(row["mismatch"])) and float(row["mismatch"]) > 0.0]
    if not values:
        return 1.0e-8, 1.0e-3
    lower = min(values)
    upper = max(values)
    if abs(upper - lower) < 1.0e-15:
        return lower * 0.85, upper * 1.15
    return lower, upper


def _pad_axis(values: Iterable[float], pad_fraction: float = 0.08) -> tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return -1.0, 1.0
    lower = min(finite)
    upper = max(finite)
    if abs(upper - lower) < 1.0e-12:
        pad = max(abs(upper) * pad_fraction, 1.0e-3)
    else:
        pad = (upper - lower) * pad_fraction
    return lower - pad, upper + pad


def _plot_standard_mismatch_parameter_space(
    rows: list[dict[str, Any]],
    output_dir: str | Path,
    prefix: str = "teobpm_mismatch_comparison",
    title_prefix: str = "TEOBPM mismatch comparison",
    family: str = "all",
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_paths: list[Path] = []
    requested = {family}
    if family in {"auto", "all"}:
        requested = {"nonspinning", "spinning"}

    plt.rcParams.update({
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    })

    nonspinning_rows = [row for row in rows if row["family"] == "nonspinning"]
    if "nonspinning" in requested and nonspinning_rows:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        markers = ["o", "s", "^", "D", "P", "X", "v"]
        for index, label in enumerate(sorted({row["label"] for row in nonspinning_rows})):
            label_rows = sorted([row for row in nonspinning_rows if row["label"] == label], key=lambda item: item["nu"])
            x = [float(row["nu"]) for row in label_rows]
            y = [float(row["mismatch"]) for row in label_rows]
            ax.plot(x, y, lw=0.85, alpha=0.45)
            ax.scatter(x, y, marker=markers[index % len(markers)], s=44, edgecolor="black", linewidth=0.25, label=label)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\nu$")
        ax.set_ylabel("Mismatch")
        ax.set_title(f"{title_prefix}: nonspinning")
        ax.legend(loc="best", frameon=False, fontsize=8)
        fig.tight_layout()
        path = output_path / f"{prefix}_nonspinning.png"
        fig.savefig(path, dpi=240)
        plt.close(fig)
        plot_paths.append(path)

    spinning_rows = [row for row in rows if row["family"] == "spinning"]
    if "spinning" in requested and spinning_rows:
        projection_rows = [
            row for row in spinning_rows
            if math.isfinite(float(row.get("nu", math.nan))) and math.isfinite(float(row.get("chi_eff", math.nan)))
        ]
        if projection_rows:
            labels = sorted({row["label"] for row in projection_rows})
            ncols = len(labels)
            fig, axes = plt.subplots(1, ncols, figsize=(max(5.0, 4.4 * ncols), 4.8), squeeze=False, sharex=True, sharey=True)
            vmin, vmax = _finite_mismatch_limits(projection_rows)
            norm = LogNorm(vmin=vmin, vmax=vmax)
            scatter = None
            xlim = _pad_axis(row["nu"] for row in projection_rows)
            ylim = _pad_axis(row["chi_eff"] for row in projection_rows)
            for column, label in enumerate(labels):
                ax = axes[0][column]
                label_rows = [row for row in projection_rows if row["label"] == label]
                scatter = ax.scatter(
                    [float(row["nu"]) for row in label_rows],
                    [float(row["chi_eff"]) for row in label_rows],
                    c=[float(row["mismatch"]) for row in label_rows],
                    norm=norm,
                    cmap="magma_r",
                    s=52,
                    edgecolor="black",
                    linewidth=0.25,
                )
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
                ax.set_xlabel(r"$\nu$")
                ax.set_title(label, fontsize=10)
                if column == 0:
                    ax.set_ylabel(r"$\chi_\mathrm{eff}$")
            if scatter is not None:
                cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), fraction=0.035, pad=0.025)
                cbar.set_label("Mismatch")
            fig.suptitle(f"{title_prefix}: spinning", y=0.995, fontsize=11)
            fig.subplots_adjust(top=0.82, wspace=0.22)
            legacy_path = output_path / f"{prefix}_spinning.png"
            chi_eff_path = output_path / f"{prefix}_spinning_chi_eff.png"
            fig.savefig(legacy_path, dpi=240)
            fig.savefig(chi_eff_path, dpi=240)
            plt.close(fig)
            plot_paths.extend([legacy_path, chi_eff_path])

    return plot_paths


def plot_teobpm_mismatch_comparison(
    mismatch_tables: list[str | Path],
    output_dir: str | Path,
    labels: list[str] | None = None,
    family: str = "auto",
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if labels is not None and len(labels) != len(mismatch_tables):
        raise ValueError("The number of labels must match the number of mismatch tables.")

    rows: list[dict[str, Any]] = []
    input_tables = []
    window_checks = []
    for index, table in enumerate(mismatch_tables):
        table_path = Path(table)
        source_label = labels[index] if labels is not None else table_path.stem.replace("_", " ")
        input_tables.append(str(table_path))
        if not table_path.exists():
            raise FileNotFoundError(f"Mismatch table `{table_path}` does not exist.")
        family_hint = family if family in {"nonspinning", "spinning"} else None
        table_rows = _read_table(table_path)
        window_checks.append(_check_peak22_mismatch_window(table_path, table_rows))
        rows.extend(_normalise_mismatch_rows(table_rows, source_label, family_hint=family_hint))

    if family not in {"auto", "all"}:
        rows = [row for row in rows if row["family"] == family]

    point_path = output_path / "teobpm_mismatch_comparison_points.csv"
    summary_path = output_path / "teobpm_mismatch_comparison_summary.csv"
    _write_rows_csv(rows, point_path)
    summary_rows = _mismatch_summary_rows(rows)
    _write_rows_csv(summary_rows, summary_path)
    plot_paths = _plot_standard_mismatch_parameter_space(rows, output_path, family=family)
    payload = {
        "input_tables": input_tables,
        "point_table": str(point_path),
        "summary_table": str(summary_path),
        "plots": [str(path) for path in plot_paths],
        "rows": len(rows),
        "families": sorted({row["family"] for row in rows}),
        "notes": "Nonspinning mismatch plots use nu. Spinning mismatch plots use nu-chi_eff.",
        "comparison_window": {
            "expected_reference": "peak22",
            "expected_t_start": 0.0,
            "tables": window_checks,
        },
    }
    (output_path / "teobpm_mismatch_comparison_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _discover_campaign_mismatch_tables(campaign_roots: Iterable[str | Path] | None) -> list[Path]:
    tables: list[Path] = []
    for root in campaign_roots or []:
        root_path = Path(root)
        tables.extend(sorted(root_path.glob("*/*/mismatch_summary.csv")))
    return tables


def _case_label_from_mismatch_table(table_path: str | Path, campaign_roots: Iterable[str | Path] | None = None) -> str:
    path = Path(table_path)
    for root in campaign_roots or []:
        root_path = Path(root)
        try:
            relative = path.resolve().relative_to(root_path.resolve())
        except ValueError:
            continue
        parts = relative.parts
        if len(parts) >= 3 and parts[-1] == "mismatch_summary.csv":
            template, family = parts[0], parts[1]
            return f"{family}-{template}"
    if path.name == "mismatch_summary.csv" and len(path.parts) >= 3:
        family = path.parent.name
        template = path.parent.parent.name
        return f"{family}-{template}"
    return path.stem.replace("_", " ")


def _mismatch_histogram_label_parts(label: Any) -> tuple[str, str, str]:
    text = str(label)
    source, separator, suffix = text.partition(":")
    family = ""
    template = ""
    if "-" in source:
        family, template = source.rsplit("-", 1)
    return family, template, suffix.strip() if separator else ""


def _mismatch_histogram_display_label(label: Any) -> str:
    family, template, suffix = _mismatch_histogram_label_parts(label)
    family_names = {
        "nonspinning": "nonspinning",
        "equal_mass_spinning": "equal-mass spinning",
        "generic_spinning": "generic spinning",
    }
    template_names = {
        "hyptan_ratexp_nr": "HypTan (RatExp NR)",
        "hyptan": "HypTan",
        "ratexp": "RatExp",
        "seobnrv5": "SEOBNRv5",
    }
    if family and template:
        display = f"{template_names.get(template, template.replace('_', ' '))} / {family_names.get(family, family.replace('_', ' '))}"
    else:
        display = str(label).replace("_", " ")
    if suffix:
        display = f"{display}: {suffix}"
    return display


def _mismatch_histogram_sort_key(label: Any) -> tuple[int, int, str]:
    family, template, suffix = _mismatch_histogram_label_parts(label)
    return (
        MISMATCH_HISTOGRAM_FAMILY_ORDER.get(family, 99),
        MISMATCH_HISTOGRAM_TEMPLATE_ORDER.get(template, 99),
        suffix or str(label),
    )


def _mismatch_histogram_color(label: Any, index: int) -> str:
    _, template, _ = _mismatch_histogram_label_parts(label)
    if template in MISMATCH_HISTOGRAM_TEMPLATE_COLORS:
        return MISMATCH_HISTOGRAM_TEMPLATE_COLORS[template]
    palette = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#56b4e9", "#e69f00", "#6f4bb1", "#4d4d4d")
    return palette[index % len(palette)]


def _mismatch_histogram_linestyle(label: Any) -> str:
    family, _, _ = _mismatch_histogram_label_parts(label)
    return {
        "nonspinning": "-",
        "equal_mass_spinning": "--",
        "generic_spinning": "-.",
    }.get(family, "-")


def _mismatch_histogram_points(
    mismatch_tables: list[str | Path],
    labels: list[str] | None = None,
    campaign_roots: Iterable[str | Path] | None = None,
) -> list[dict[str, Any]]:
    if labels is not None and len(labels) != len(mismatch_tables):
        raise ValueError("The number of labels must match the number of mismatch tables.")

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    source_by_group: dict[tuple[str, str, str, str], str] = {}
    input_by_group: dict[tuple[str, str, str, str], str] = {}
    for table_index, table in enumerate(mismatch_tables):
        table_path = Path(table)
        if not table_path.exists():
            raise FileNotFoundError(f"Mismatch table `{table_path}` does not exist.")
        source_label = labels[table_index] if labels is not None else _case_label_from_mismatch_table(table_path, campaign_roots)
        for row_index, row in enumerate(_read_table(table_path)):
            for expanded in _expanded_mismatch_rows(row, source_label):
                comparison_label = _mismatch_series_label(expanded, source_label)
                case_label = source_label
                if comparison_label and comparison_label != source_label:
                    case_label = f"{source_label}: {comparison_label}"
                mode = str(expanded.get("mode") or expanded.get("lm") or "all").strip() or "unknown"
                sxs_id = str(
                    expanded.get("sxs_id")
                    or expanded.get("simulation")
                    or expanded.get("simulation_id")
                    or expanded.get("run_id")
                    or f"row_{row_index}"
                ).strip()
                split = str(expanded.get("split") or "").strip()
                key = (case_label, mode, sxs_id, split)
                grouped.setdefault(key, []).append(expanded)
                source_by_group[key] = source_label
                input_by_group[key] = str(table_path)

    points: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        label, mode, sxs_id, split = key
        mismatch = _representative_tabular_mismatch(rows)
        if not math.isfinite(mismatch) or mismatch <= 0.0:
            continue
        points.append({
            "input_table": input_by_group[key],
            "label": label,
            "mode": mode,
            "mismatch": float(mismatch),
            "n_aggregated_rows": len(rows),
            "source": source_by_group[key],
            "split": split,
            "sxs_id": sxs_id,
        })
    return points


def _mismatch_histogram_summary_rows(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for point in points:
        grouped.setdefault((str(point["label"]), str(point["mode"])), []).append(point)
        grouped.setdefault((str(point["label"]), "all"), []).append(point)

    summary_rows: list[dict[str, Any]] = []
    for (label, mode), rows in sorted(grouped.items()):
        values = [float(row["mismatch"]) for row in rows]
        summary_rows.append({
            "label": label,
            "mode": mode,
            "n": len(values),
            "mismatch_min": min(values),
            "mismatch_p05": _percentile(values, 5.0),
            "mismatch_median": _median(values),
            "mismatch_p95": _percentile(values, 95.0),
            "mismatch_max": max(values),
        })
    return summary_rows


def _mismatch_histogram_bins(values: list[float], bins: int) -> np.ndarray:
    finite = [float(value) for value in values if math.isfinite(float(value)) and float(value) > 0.0]
    if not finite:
        return np.linspace(0.0, 1.0, max(2, bins + 1))
    lower = min(finite)
    upper = max(finite)
    if abs(upper - lower) < max(1.0e-16, abs(upper) * 1.0e-12):
        lower *= 0.85
        upper *= 1.15
    lower = max(lower, np.nextafter(0.0, 1.0))
    return np.logspace(math.log10(lower), math.log10(upper), max(2, bins + 1))


def _plot_mismatch_histogram_series(
    series: list[tuple[str, list[float]]],
    output_file: str | Path,
    title: str,
    bins: int,
) -> Path | None:
    import matplotlib.pyplot as plt

    cleaned = [
        (label, [float(value) for value in values if math.isfinite(float(value)) and float(value) > 0.0])
        for label, values in series
    ]
    cleaned = [(label, values) for label, values in cleaned if values]
    if not cleaned:
        return None

    all_values = [value for _, values in cleaned for value in values]
    bin_edges = _mismatch_histogram_bins(all_values, bins)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "figure.facecolor": "white",
    })
    if len(cleaned) == 1:
        fig, ax = plt.subplots(figsize=(7.4, 4.6))
    else:
        fig_height = max(7.2, 1.4 + 0.9 * len(cleaned))
        fig, ax = plt.subplots(figsize=(10.4, fig_height))
    if isinstance(ax, (list, tuple, np.ndarray)):
        ax = np.asarray(ax, dtype=object).flat[0]
    if len(cleaned) == 1:
        label, values = cleaned[0]
        color = _mismatch_histogram_color(label, 0)
        median = _median(values)
        p05 = _percentile(values, 5.0)
        p95 = _percentile(values, 95.0)
        ax.hist(values, bins=bin_edges, color=color, edgecolor="white", linewidth=0.8, alpha=0.78)
        ax.axvspan(p05, p95, color=color, alpha=0.12, lw=0)
        ax.axvline(median, color="#202124", lw=1.7, label=f"median = {median:.2g}")
        ax.axvline(p05, color=color, lw=1.1, ls="--", alpha=0.85)
        ax.axvline(p95, color=color, lw=1.1, ls="--", alpha=0.85, label="5-95%")
        ax.text(
            0.025,
            0.96,
            f"n = {len(values)}\nmedian = {median:.2g}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.32", "fc": "white", "ec": "#d0d7de", "alpha": 0.92},
        )
        ax.legend(loc="upper right", frameon=False, fontsize=8)
        ax.set_ylabel("Count")
    else:
        row_step = 1.25
        row_amplitude = 0.92
        y_positions = []
        display_labels = []
        tick_colors = []
        for index, (label, values) in enumerate(cleaned):
            baseline = (len(cleaned) - index - 1) * row_step
            color = _mismatch_histogram_color(label, index)
            counts, _ = np.histogram(values, bins=bin_edges)
            counts = counts.astype(float)
            if counts.max() > 0:
                counts = counts / counts.max() * row_amplitude
            x_values = np.repeat(bin_edges, 2)[1:-1]
            y_values = baseline + np.repeat(counts, 2)
            ax.fill_between(x_values, baseline, y_values, color=color, alpha=0.22, lw=0)
            ax.plot(x_values, y_values, color=color, lw=2.0)
            median = _median(values)
            ax.vlines(median, baseline, baseline + row_amplitude, color=color, lw=1.45)
            ax.text(
                bin_edges[-1],
                baseline + 0.13,
                f"n={len(values)}",
                ha="right",
                va="bottom",
                fontsize=8,
                color="#4b5563",
            )
            y_positions.append(baseline + 0.42 * row_step)
            display_labels.append(_mismatch_histogram_display_label(label))
            tick_colors.append(color)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(display_labels, fontsize=9)
        for tick_label, color in zip(ax.get_yticklabels(), tick_colors):
            tick_label.set_color(color)
        ax.set_ylim(-0.25, len(cleaned) * row_step - 0.05)
        ax.set_ylabel("")
    ax.set_xscale("log")
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.set_xlabel("Mismatch")
    ax.set_title(title)
    ax.grid(axis="x", which="major", alpha=0.28)
    ax.grid(axis="y", alpha=0.08)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def _plot_mismatch_cdf_series(
    series: list[tuple[str, list[float]]],
    output_file: str | Path,
    title: str,
) -> Path | None:
    import matplotlib.pyplot as plt

    cleaned = [
        (label, sorted(float(value) for value in values if math.isfinite(float(value)) and float(value) > 0.0))
        for label, values in series
    ]
    cleaned = [(label, values) for label, values in cleaned if values]
    if not cleaned:
        return None

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_values = [value for _, values in cleaned for value in values]
    x_edges = _mismatch_histogram_bins(all_values, 32)

    plt.rcParams.update({
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "figure.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    if isinstance(ax, (list, tuple, np.ndarray)):
        ax = np.asarray(ax, dtype=object).flat[0]

    for index, (label, values) in enumerate(cleaned):
        cumulative = np.arange(1, len(values) + 1, dtype=float) / float(len(values))
        color = _mismatch_histogram_color(label, index)
        ax.step(
            values,
            cumulative,
            where="post",
            color=color,
            linestyle=_mismatch_histogram_linestyle(label),
            linewidth=2.0,
            label=f"{_mismatch_histogram_display_label(label)} (n={len(values)})",
        )

    ax.set_xscale("log")
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mismatch")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(title)
    ax.grid(axis="x", which="major", alpha=0.28)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_mismatch_histograms(
    mismatch_tables: list[str | Path] | None,
    output_dir: str | Path,
    labels: list[str] | None = None,
    campaign_roots: list[str | Path] | None = None,
    bins: int = 24,
) -> dict[str, Any]:
    tables = [Path(table) for table in (mismatch_tables or [])]
    tables.extend(_discover_campaign_mismatch_tables(campaign_roots))
    tables = list(dict.fromkeys(tables))
    if not tables:
        raise ValueError("Provide at least one --mismatch-table or --campaign-root with mismatch_summary.csv files.")
    if labels is not None and len(labels) != len(tables):
        raise ValueError("The number of labels must match the number of discovered and explicit mismatch tables.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    points = _mismatch_histogram_points(tables, labels=labels, campaign_roots=campaign_roots)
    points_table = output_path / "mismatch_histogram_points.csv"
    summary_table = output_path / "mismatch_histogram_summary.csv"
    _write_rows_csv(points, points_table)
    summary_rows = _mismatch_histogram_summary_rows(points)
    _write_rows_csv(summary_rows, summary_table)

    plot_paths: list[Path] = []
    histogram_paths: list[Path] = []
    cdf_paths: list[Path] = []
    labels_seen = sorted({str(point["label"]) for point in points}, key=_mismatch_histogram_sort_key)
    modes_seen = sorted({str(point["mode"]) for point in points})
    for label in labels_seen:
        label_points = [point for point in points if point["label"] == label]
        case_dir = output_path / "cases" / _safe_filename_token(label)
        path = _plot_mismatch_histogram_series(
            [(label, [float(point["mismatch"]) for point in label_points])],
            case_dir / "mismatch_histogram_all_modes.png",
            f"Mismatch histogram: {_mismatch_histogram_display_label(label)}, all modes",
            bins,
        )
        if path is not None:
            plot_paths.append(path)
            histogram_paths.append(path)
        for mode in modes_seen:
            mode_points = [point for point in label_points if point["mode"] == mode]
            if not mode_points:
                continue
            path = _plot_mismatch_histogram_series(
                [(label, [float(point["mismatch"]) for point in mode_points])],
                case_dir / "modes" / f"mismatch_histogram_mode_{_safe_filename_token(mode)}.png",
                f"Mismatch histogram: {_mismatch_histogram_display_label(label)}, mode {mode}",
                bins,
            )
            if path is not None:
                plot_paths.append(path)
                histogram_paths.append(path)

    combined_dir = output_path / "combined"
    path = _plot_mismatch_histogram_series(
        [
            (label, [float(point["mismatch"]) for point in points if point["label"] == label])
            for label in labels_seen
        ],
        combined_dir / "mismatch_histogram_all_modes.png",
        "Mismatch histograms by case: all modes",
        bins,
    )
    if path is not None:
        plot_paths.append(path)
        histogram_paths.append(path)
    for mode in modes_seen:
        path = _plot_mismatch_histogram_series(
            [
                (label, [float(point["mismatch"]) for point in points if point["label"] == label and point["mode"] == mode])
                for label in labels_seen
            ],
            combined_dir / f"mismatch_histogram_mode_{_safe_filename_token(mode)}.png",
            f"Mismatch histograms by case: mode {mode}",
            bins,
        )
        if path is not None:
            plot_paths.append(path)
            histogram_paths.append(path)

    cdf_dir = output_path / "cdfs" / "combined"
    path = _plot_mismatch_cdf_series(
        [
            (label, [float(point["mismatch"]) for point in points if point["label"] == label])
            for label in labels_seen
        ],
        cdf_dir / "mismatch_cdf_all_modes.png",
        "Mismatch CDFs by case: all modes",
    )
    if path is not None:
        plot_paths.append(path)
        cdf_paths.append(path)
    for mode in modes_seen:
        path = _plot_mismatch_cdf_series(
            [
                (label, [float(point["mismatch"]) for point in points if point["label"] == label and point["mode"] == mode])
                for label in labels_seen
            ],
            cdf_dir / f"mismatch_cdf_mode_{_safe_filename_token(mode)}.png",
            f"Mismatch CDFs by case: mode {mode}",
        )
        if path is not None:
            plot_paths.append(path)
            cdf_paths.append(path)

    payload = {
        "bins": bins,
        "cdf_plots": [str(path) for path in cdf_paths],
        "histogram_plots": [str(path) for path in histogram_paths],
        "input_tables": [str(table) for table in tables],
        "labels": labels_seen,
        "modes": modes_seen,
        "plots": [str(path) for path in plot_paths],
        "points": len(points),
        "points_table": str(points_table),
        "summary_table": str(summary_table),
    }
    (output_path / "mismatch_histogram_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _safe_filename_token(text: Any) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(text))


def plot_local_fit_diagnostics(
    local_fit_table: str | Path,
    output_dir: str | Path,
    family: str = "auto",
    mismatch_tables: list[str | Path] | None = None,
    mismatch_labels: list[str] | None = None,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    coefficient_dir = output_path / "coefficients"
    mismatch_dir = output_path / "mismatches"
    coefficient_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        row for row in _read_table(local_fit_table)
        if row.get("value") not in (None, "") and row.get("target") not in (None, "")
    ]
    plt.rcParams.update({
        "figure.figsize": (7.0, 4.8),
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    coefficient_plots: list[Path] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("mode", "")), str(row.get("target", ""))), []).append(row)

    for (mode_label, target), target_rows in sorted(grouped.items()):
        values = np.array([_as_float(row.get("value"), math.nan) for row in target_rows])
        finite_mask = np.isfinite(values)
        if not finite_mask.any():
            continue
        finite_rows = [row for row, keep in zip(target_rows, finite_mask) if keep]
        values = values[finite_mask]
        chi_eff = np.array([_as_float(row.get("chi_eff"), 0.0) for row in finite_rows])
        nu = np.array([_as_float(row.get("nu"), math.nan) for row in finite_rows])

        fig, ax = plt.subplots()
        if np.all(np.abs(chi_eff) <= 1.0e-8):
            ax.scatter(nu, values, s=42, edgecolor="black", linewidth=0.25)
            ax.set_xlabel(r"$\nu$")
        else:
            scatter = ax.scatter(nu, chi_eff, c=values, cmap="viridis", s=52, edgecolor="black", linewidth=0.25)
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label(target)
            ax.set_xlabel(r"$\nu$")
            ax.set_ylabel(r"$\chi_\mathrm{eff}$")
        ax.set_title(f"Local fit {target}, mode {mode_label}")
        if np.all(np.abs(chi_eff) <= 1.0e-8):
            ax.set_ylabel(target)
        path = coefficient_dir / f"local_fit_coefficients_mode_{_safe_filename_token(mode_label)}_{_safe_filename_token(target)}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
        coefficient_plots.append(path)

    mismatch_summary: dict[str, Any] = {}
    if mismatch_tables:
        mismatch_summary = plot_teobpm_mismatch_comparison(mismatch_tables, mismatch_dir, labels=mismatch_labels, family=family)

    payload = {
        "local_fit_table": str(local_fit_table),
        "coefficient_plots": [str(path) for path in coefficient_plots],
        "mismatch_summary": mismatch_summary,
        "plots": [str(path) for path in coefficient_plots] + list(mismatch_summary.get("plots", [])),
        "notes": (
            "Local-fit coefficient plots use the collected construction table. "
            "Mismatch plots are written only from explicit evaluation mismatch tables, "
            "which must use the 22-peak comparison start."
        ),
    }
    (output_path / "local_fit_plot_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def plot_validation_diagnostics(predictions: list[dict[str, Any]], output_dir: str | Path) -> list[Path]:
    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_paths = []
    if not predictions:
        return plot_paths

    plt.rcParams.update({
        "figure.figsize": (7.0, 4.8),
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    q = np.array([_as_float(row.get("q")) for row in predictions])
    chi_eff = np.array([_as_float(row.get("chi_eff")) for row in predictions])
    abs_residual = np.array([_as_float(row.get("abs_residual")) for row in predictions])

    fig, ax = plt.subplots()
    scatter = ax.scatter(q, chi_eff, c=abs_residual, s=42, cmap="viridis", edgecolor="black", linewidth=0.25)
    ax.set_xlabel("Mass ratio q")
    ax.set_ylabel("Effective spin")
    ax.set_title("TEOBPM validation residuals across parameter space")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Absolute residual")
    fig.tight_layout()
    scatter_path = output_path / "validation_parameter_space_residuals.png"
    fig.savefig(scatter_path, dpi=220)
    plt.close(fig)
    plot_paths.append(scatter_path)

    fig, ax = plt.subplots()
    ax.hist(abs_residual, bins=min(30, max(5, int(math.sqrt(len(abs_residual))))), color="#2f6f88", edgecolor="white")
    ax.set_xlabel("Absolute residual")
    ax.set_ylabel("Count")
    ax.set_title("TEOBPM validation residual histogram")
    fig.tight_layout()
    hist_path = output_path / "validation_residual_histogram.png"
    fig.savefig(hist_path, dpi=220)
    plt.close(fig)
    plot_paths.append(hist_path)

    return plot_paths


def validate_global_fit(
    global_fit_file: str | Path,
    validation_table: str | Path,
    output_dir: str | Path,
    mismatch_tables: list[str | Path] | None = None,
    mismatch_labels: list[str] | None = None,
    family: str = "auto",
) -> dict[str, Any]:
    fit_payload = _load_global_fit(global_fit_file)
    predictions = _prediction_rows(validation_table, fit_payload)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_prediction_csv(predictions, output_path / "validation_predictions.csv")
    plot_paths = plot_validation_diagnostics(predictions, output_path)
    mismatch_summary: dict[str, Any] = {}
    if mismatch_tables:
        mismatch_summary = plot_teobpm_mismatch_comparison(
            mismatch_tables,
            output_path / "mismatches",
            labels=mismatch_labels,
            family=family,
        )
        plot_paths.extend(Path(path) for path in mismatch_summary.get("plots", []))
    if predictions:
        abs_residuals = sorted(float(row["abs_residual"]) for row in predictions)
        middle = len(abs_residuals) // 2
        if len(abs_residuals) % 2:
            median_abs_residual = abs_residuals[middle]
        else:
            median_abs_residual = 0.5 * (abs_residuals[middle - 1] + abs_residuals[middle])
        summary = {
            "n": len(predictions),
            "median_abs_residual": float(median_abs_residual),
            "max_abs_residual": float(max(abs_residuals)),
            "plots": [str(path) for path in plot_paths],
            "mismatch_summary": mismatch_summary,
        }
    else:
        summary = {"n": 0, "plots": [str(path) for path in plot_paths], "mismatch_summary": mismatch_summary}
    (output_path / "validation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _read_json_if_exists(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _latex_graphics(validation_summary: dict[str, Any], output_path: Path) -> str:
    lines = []
    for raw_path in validation_summary.get("plots", []):
        plot_path = Path(raw_path)
        if not plot_path.exists():
            continue
        relative = os.path.relpath(plot_path, output_path.parent)
        caption = plot_path.stem.replace("_", " ")
        lines.append(
            "\\begin{figure}[htbp]\n"
            "\\centering\n"
            f"\\includegraphics[width=0.92\\linewidth]{{\\detokenize{{{relative}}}}}\n"
            f"\\caption{{{caption}}}\n"
            "\\end{figure}"
        )
    if not lines:
        return "No validation figures were available when this report was rendered."
    return "\n\n".join(lines)


def render_report(campaign_dir: str | Path, output_tex: str | Path, compile_pdf: bool = True) -> Path | None:
    campaign_path = Path(campaign_dir)
    output_path = Path(output_tex)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = _read_json_if_exists(campaign_path / "campaign_config.json", {})
    validation_summary = _read_json_if_exists(campaign_path / "validation" / "validation_summary.json", {})
    run_summary = _read_json_if_exists(campaign_path / "local_fit_run_summary.json", {})
    collection_summary = _read_json_if_exists(campaign_path / "local_fit_collection_summary.json", {})
    global_fit = _read_json_if_exists(campaign_path / "teobpm_global_fit.json", {"fits": {}})
    training_manifest = campaign_path / "manifests" / "training_manifest.json"
    validation_manifest = campaign_path / "manifests" / "validation_manifest.json"
    n_train = len(json.loads(training_manifest.read_text(encoding="utf-8"))) if training_manifest.exists() else 0
    n_validation = len(json.loads(validation_manifest.read_text(encoding="utf-8"))) if validation_manifest.exists() else 0
    n_fit_modes = len(global_fit.get("fits", {}))
    n_fit_targets = sum(len(targets) for targets in global_fit.get("fits", {}).values())
    figures = _latex_graphics(validation_summary, output_path)

    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\usepackage{{siunitx}}
\title{{TEOBPM Calibration Against SXS Ringdown Injections}}
\author{{bayRing automatic calibration workflow}}
\date{{\today}}
\begin{{document}}
\maketitle

\section{{Scope}}
This report records a TEOBPM calibration campaign for non-eccentric,
non-precessing SXS simulations. The campaign family is
\texttt{{{config.get("family", "unspecified")}}}; aligned-spin campaigns are
constructed hierarchically from a nonspinning baseline.

\section{{Dataset}}
\begin{{tabular}}{{lr}}
\toprule
Training simulations & {n_train}\\
Validation simulations & {n_validation}\\
Validation fraction & {config.get("validation_fraction", "n/a")}\\
Maximum eccentricity & {config.get("max_eccentricity", "n/a")}\\
\bottomrule
\end{{tabular}}

\section{{Local Fits}}
Local TEOBPM fits are generated under \texttt{{local\_fit\_configs/}} and are
intended to estimate peak quantities, post-peak TEOBPM coefficients, and
relative merger phases mode by mode.

\begin{{tabular}}{{lr}}
\toprule
Indexed jobs & {run_summary.get("jobs", collection_summary.get("jobs", "n/a"))}\\
Completed jobs & {run_summary.get("completed", "n/a")}\\
Failed jobs & {run_summary.get("failed", "n/a")}\\
Collected coefficient rows & {collection_summary.get("rows", "n/a")}\\
Collection failures & {collection_summary.get("failures", "n/a")}\\
\bottomrule
\end{{tabular}}

\section{{Global Fits}}
The global-fit file is versioned with schema \texttt{{{FIT_SCHEMA}}}. The
selection rule is the lowest Bayesian information criterion among hierarchical
candidate bases, with the aligned-spin basis restricted to spin corrections on
top of the nonspinning layer.

\begin{{tabular}}{{lr}}
\toprule
Modes with fitted targets & {n_fit_modes}\\
Fitted mode-target pairs & {n_fit_targets}\\
\bottomrule
\end{{tabular}}

\section{{Validation}}
\begin{{tabular}}{{lr}}
\toprule
Validated coefficients & {validation_summary.get("n", 0)}\\
Median absolute residual & {validation_summary.get("median_abs_residual", "n/a")}\\
Maximum absolute residual & {validation_summary.get("max_abs_residual", "n/a")}\\
\bottomrule
\end{{tabular}}

{figures}

\section{{Open Production Steps}}
The workflow is ready to run the expensive SXS downloads and local fits. A final
science report should add the generated mismatch figures after the validation
table has been produced from the completed local-fit campaign.

\end{{document}}
"""
    output_path.write_text(tex, encoding="utf-8")

    if not compile_pdf:
        return None
    try:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", output_path.name],
            cwd=str(output_path.parent),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return output_path.with_suffix(".pdf")


def _appendix_a_properties_file(nc_ringdown_dir: str | Path, group: str) -> Path:
    data_dir = Path(nc_ringdown_dir) / "src" / "data"
    if group in {"equal-mass", "nonspinning"}:
        return data_dir / "SXS_Parameters.csv"
    raise ValueError(f"Unknown Appendix A group `{group}`.")


def _appendix_a_rows(nc_ringdown_dir: str | Path, group: str) -> dict[str, dict[str, str]]:
    path = _appendix_a_properties_file(nc_ringdown_dir, group)
    rows = {}
    for row in _rows_from_csv(path):
        rows[str(row.get("ID", "")).zfill(4)] = row
    return rows


def _appendix_a_ids(group: str) -> tuple[str, ...]:
    if group == "equal-mass":
        return APPENDIX_A_EQUAL_MASS_IDS
    if group == "nonspinning":
        return APPENDIX_A_NONSPINNING_IDS
    raise ValueError(f"Unknown Appendix A group `{group}`.")


def _appendix_a_campaign_name(group: str, template: str) -> str:
    prefix = "hyptan" if template == "HypTan" else "ratexp"
    return f"{prefix}_{group.replace('-', '_')}"


def _appendix_a_targets(template: str) -> tuple[str, ...]:
    return APPENDIX_A_TARGETS_HYPTAN if template == "HypTan" else APPENDIX_A_TARGETS_RATEXP


def _appendix_a_degree(group: str, template: str, target: str) -> int:
    if group == "nonspinning":
        return 1
    if group != "equal-mass":
        raise ValueError(f"Unknown Appendix A group `{group}`.")
    if template == "HypTan" and target == "c3p":
        return 4
    return 3


def _polyval_high_to_low(coefficients: Iterable[float], x_value: float | np.ndarray) -> float | np.ndarray:
    result: float | np.ndarray = 0.0
    for coefficient in coefficients:
        result = result * x_value + float(coefficient)
    return result


def _appendix_a_past_hyptan_value(group: str, target: str, x_value: float | np.ndarray) -> float | np.ndarray | None:
    if target not in APPENDIX_A_TARGETS_HYPTAN:
        return None
    if group == "equal-mass":
        coefficients = APPENDIX_A_PAST_HYPTAN_EQUAL_MASS.get(target)
        if coefficients is None:
            return None
        return _polyval_high_to_low(coefficients, x_value)
    if group == "nonspinning":
        fit = APPENDIX_A_PAST_HYPTAN_NONSPINNING.get(target)
        if fit is None:
            return None
        numerator = _polyval_high_to_low(fit["num"], x_value)
        denominator = _polyval_high_to_low(fit["den"], x_value)
        return float(fit["offset"]) + numerator / denominator
    return None


def _appendix_a_rao_fit_path(nc_ringdown_dir: str | Path, catalog: str, order: int) -> Path:
    if catalog not in {"sxs", "rit"}:
        raise ValueError("`catalog` must be either `sxs` or `rit`.")
    if order < 0:
        raise ValueError("`order` must be non-negative.")
    catalog_name = "nc_fits_sxs_non-spinning" if catalog == "sxs" else "nc_fits_rit_non-spinning"
    path = Path(nc_ringdown_dir) / "src" / "data" / "fits" / catalog_name / f"order_fits_nu_{order}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Rao/Carullo nonspinning fit file not found: `{path}`.")
    return path


def _appendix_a_load_rao_nu_fits(
    nc_ringdown_dir: str | Path,
    catalog: str = "sxs",
    order: int = 1,
) -> dict[str, dict[str, Any]]:
    path = _appendix_a_rao_fit_path(nc_ringdown_dir, catalog, order)
    rows = _read_table(path)
    if not rows:
        raise ValueError(f"Rao/Carullo nonspinning fit file is empty: `{path}`.")
    row = rows[0]
    fits: dict[str, dict[str, Any]] = {}
    for target, column_base in APPENDIX_A_RAO_TARGET_COLUMNS.items():
        coefficients = []
        for index in range(order + 1):
            key = f"{column_base}_p{index}"
            if key not in row:
                raise ValueError(f"Missing Rao/Carullo coefficient `{key}` in `{path}`.")
            coefficients.append(_as_float(row[key], default=math.nan))
        if any(not math.isfinite(value) for value in coefficients):
            raise ValueError(f"Non-finite Rao/Carullo coefficients for `{target}` in `{path}`.")
        fits[target] = {
            "source_file": str(path),
            "catalog": catalog,
            "order": order,
            "fit_type": row.get("fit_type", "nu"),
            "coefficients_high_to_low": coefficients,
            "reference_type": "appendix-nu",
        }
    return fits


def _appendix_a_rao_value(rao_fits: dict[str, dict[str, Any]], target: str, x_value: float | np.ndarray) -> float | np.ndarray | None:
    fit = rao_fits.get(target)
    if fit is None:
        return None
    return _polyval_high_to_low(fit["coefficients_high_to_low"], x_value)


def _appendix_a_resolve_current_rao_fit_file(nc_ringdown_dir: str | Path, fit_file: str | Path | None = None) -> Path:
    if fit_file is None:
        return Path(nc_ringdown_dir) / APPENDIX_A_CURRENT_RAO_FIT
    path = Path(fit_file)
    if path.is_absolute():
        return path
    return Path(nc_ringdown_dir) / path


def _appendix_a_fit_type_variables(fit_type: str, source: str | Path) -> list[str]:
    variables = [variable.strip().lower() for variable in str(fit_type).split("_") if variable.strip()]
    noncircular_variables = [variable for variable in variables if variable in APPENDIX_A_NONCIRCULAR_VARIABLES]
    if noncircular_variables:
        raise ValueError(
            "Appendix A workflow is restricted to non-eccentric/noncircular data; "
            f"`{source}` uses excluded fit variable(s): {', '.join(noncircular_variables)}."
        )
    unknown_variables = [variable for variable in variables if variable not in APPENDIX_A_RAO_VARIABLE_COLUMNS]
    if unknown_variables:
        raise ValueError(
            "Unsupported Rao/Carullo fit variable(s) in "
            f"`{source}`: {', '.join(unknown_variables)}."
        )
    return variables


def _appendix_a_load_current_rao_fits(
    nc_ringdown_dir: str | Path,
    fit_file: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    path = _appendix_a_resolve_current_rao_fit_file(nc_ringdown_dir, fit_file=fit_file)
    if not path.exists():
        raise FileNotFoundError(f"Current Rao/Carullo fit file not found: `{path}`.")
    rows = _read_table(path)
    if not rows:
        raise ValueError(f"Current Rao/Carullo fit file is empty: `{path}`.")
    row = rows[0]
    fit_order = _as_int(row.get("fit_order"), default=0)
    fit_type = str(row.get("fit_type", "")).strip()
    if not fit_type:
        raise ValueError(f"Current Rao/Carullo fit file has no `fit_type`: `{path}`.")
    fit_variables = _appendix_a_fit_type_variables(fit_type, path)

    norms = {}
    for variable in fit_variables:
        scale_key = f"norm_{variable}_scale"
        shift_key = f"norm_{variable}_shift"
        if scale_key in row or shift_key in row:
            norms[variable] = {
                "scale": _as_float(row.get(scale_key), default=1.0),
                "shift": _as_float(row.get(shift_key), default=0.0),
            }

    fits: dict[str, dict[str, Any]] = {}
    for target, column_base in APPENDIX_A_RAO_TARGET_COLUMNS.items():
        coefficients = {}
        index = 0
        while f"{column_base}_p{index}" in row:
            coefficients[f"p{index}"] = _as_float(row[f"{column_base}_p{index}"], default=math.nan)
            index += 1
        if not coefficients:
            raise ValueError(f"Missing Rao/Carullo coefficients for `{target}` in `{path}`.")
        if any(not math.isfinite(value) for value in coefficients.values()):
            raise ValueError(f"Non-finite Rao/Carullo coefficients for `{target}` in `{path}`.")
        fits[target] = {
            "source_file": str(path),
            "order": fit_order,
            "fit_type": fit_type,
            "coefficients_by_index": coefficients,
            "norms": norms,
            "reference_type": "current-nc",
        }
    return fits


def _appendix_a_load_rao_reference(
    nc_ringdown_dir: str | Path,
    reference: str,
    catalog: str = "sxs",
    order: int = 1,
    fit_file: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    if reference == "appendix-nu":
        return _appendix_a_load_rao_nu_fits(nc_ringdown_dir, catalog=catalog, order=order)
    if reference == "current-nc":
        return _appendix_a_load_current_rao_fits(nc_ringdown_dir, fit_file=fit_file)
    raise ValueError("`reference` must be either `appendix-nu` or `current-nc`.")


def _appendix_a_prior_text(template: str) -> str:
    if template == "HypTan":
        return """c3A_22-min        = -0.5
c3A_22-max        = -0.1
c3p_22-min        = 2
c3p_22-max        = 7
c4p_22-min        = 0
c4p_22-max        = 10
"""
    if template == "RatExp":
        return """c2A_22-min        = 0
c2A_22-max        = 5
c3A_22-min        = -5
c3A_22-max        = 0.9
c2p_22-min        = 0
c2p_22-max        = 10
c3p_22-min        = 0
c3p_22-max        = 10
c4p_22-min        = 0
c4p_22-max        = 10
"""
    raise ValueError(f"Unknown Appendix A template `{template}`.")


def _appendix_a_coordinate(row: dict[str, Any], group: str) -> tuple[str, float]:
    if group == "nonspinning":
        return "nu", _as_float(row.get("nu"))
    q_paper = 1.0 / _as_float(row.get("q"), default=1.0)
    chi1z = _as_float(row.get("chi1z"))
    chi2z = _as_float(row.get("chi2z"))
    a0 = (q_paper * chi1z + chi2z) / (q_paper + 1.0)
    a12 = (q_paper * chi1z - chi2z) / (q_paper + 1.0)
    x12 = (q_paper - 1.0) / (q_paper + 1.0)
    return "S_hat", a0 + x12 * a12


def _appendix_a_config_text(
    sxs_id: str,
    group: str,
    template: str,
    outdir: Path,
    properties_file: Path,
    method: str,
    n_random_seeds: int,
    nlive: int,
    maxmcmc: int,
    t_start: float,
    t_end: float,
) -> str:
    merger_data = 1 if _teobpm_requires_merger_peak_data(template) else 0
    inference_lines = [
        f"method            = {method}",
        "min-method        = trf",
        f"n-random-seeds    = {n_random_seeds}",
        f"nlive             = {nlive}",
        f"maxmcmc           = {maxmcmc}",
        f"t-start           = {t_start}",
        f"t-end             = {t_end}",
    ]
    return f"""[I/O]
outdir            = {outdir}
screen-output     = 0
run-type          = full

[NR-data]
catalog           = SXS
ID                = {sxs_id}
l-NR              = 2
m                 = 2
error             = late-time-const-error
properties-file   = {properties_file}

[Model]
template          = TEOBPM
TEOB-template     = {template}
TEOB-calibration  = qc
TEOB-global-fit   = 0
TEOB-merger-data  = {merger_data}

[Inference]
{chr(10).join(inference_lines)}

[Priors]
{_appendix_a_prior_text(template)}"""


def prepare_appendix_a_campaign(
    output_dir: str | Path,
    nc_ringdown_dir: str | Path = "/private/tmp/nc_ringdown",
    group: str = "all",
    template: str = "all",
    method: str = "Minimization",
    n_random_seeds: int = 16,
    nlive: int = 128,
    maxmcmc: int = 128,
    t_start: float = 0.0,
    t_end: float = 80.0,
    bayring_executable: str = "bayRing",
) -> dict[str, Any]:
    groups = ["equal-mass", "nonspinning"] if group == "all" else [group]
    templates = ["HypTan", "RatExp"] if template == "all" else [template]
    output_dir = Path(output_dir)
    config_dir = output_dir / "local_fit_configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[LocalFitJob] = []
    manifest_rows: list[dict[str, Any]] = []
    config_paths: list[Path] = []
    for group_name in groups:
        metadata_rows = _appendix_a_rows(nc_ringdown_dir, group_name)
        properties_file = _appendix_a_properties_file(nc_ringdown_dir, group_name)
        for template_name in templates:
            campaign_name = _appendix_a_campaign_name(group_name, template_name)
            for sxs_id in _appendix_a_ids(group_name):
                metadata = metadata_rows.get(sxs_id)
                if metadata is None:
                    raise ValueError(f"SXS:{sxs_id} was not found in `{properties_file}`.")
                outdir = output_dir / "local_fits" / campaign_name / f"SXS_{sxs_id}"
                config_path = config_dir / campaign_name / f"SXS_{sxs_id}.ini"
                config_path.parent.mkdir(parents=True, exist_ok=True)
                config_path.write_text(
                    _appendix_a_config_text(
                        sxs_id,
                        group_name,
                        template_name,
                        outdir,
                        properties_file,
                        method,
                        n_random_seeds,
                        nlive,
                        maxmcmc,
                        t_start,
                        t_end,
                    ),
                    encoding="utf-8",
                )
                config_paths.append(config_path)
                x_name, x_value = _appendix_a_coordinate(metadata, group_name)
                job = LocalFitJob(
                    sxs_id=f"SXS:BBH:{sxs_id}",
                    mode="22",
                    split="training",
                    config_file=str(config_path),
                    outdir=str(outdir),
                    q=_as_float(metadata.get("q")),
                    nu=_as_float(metadata.get("nu")),
                    chi1z=_as_float(metadata.get("chi1z")),
                    chi2z=_as_float(metadata.get("chi2z")),
                    chi_eff=_as_float(metadata.get("chieff")),
                    chi_a=0.5 * (_as_float(metadata.get("chi1z")) - _as_float(metadata.get("chi2z"))),
                    eccentricity=_as_float(metadata.get("ecc")),
                    quality_score=0.0,
                )
                jobs.append(job)
                manifest_rows.append({
                    "schema": APPENDIX_A_SCHEMA,
                    "campaign": campaign_name,
                    "group": group_name,
                    "template": template_name,
                    "sxs_id": job.sxs_id,
                    "sxs_number": sxs_id,
                    "outdir": job.outdir,
                    "config_file": job.config_file,
                    "x_name": x_name,
                    "x_value": x_value,
                    "q": job.q,
                    "nu": job.nu,
                    "chi1z": job.chi1z,
                    "chi2z": job.chi2z,
                    "chi_eff": job.chi_eff,
                    "chi_a": job.chi_a,
                    "eccentricity": job.eccentricity,
                    "properties_file": str(properties_file),
                })

    write_local_fit_index(jobs, output_dir / "local_fit_index.csv")
    _write_rows_csv(manifest_rows, output_dir / "appendix_a_manifest.csv")
    script_config = CalibrationConfig(family="aligned-spin", output_dir=str(output_dir), bayring_executable=bayring_executable)
    run_script = write_run_script(config_paths, script_config)
    config_payload = {
        "schema": APPENDIX_A_SCHEMA,
        "nc_ringdown_dir": str(nc_ringdown_dir),
        "group": group,
        "template": template,
        "teob_calibration": "qc",
        "method": method,
        "n_random_seeds": n_random_seeds,
        "nlive": nlive,
        "maxmcmc": maxmcmc,
        "t_start": t_start,
        "t_end": t_end,
        "bayring_executable": bayring_executable,
    }
    (output_dir / "campaign_config.json").write_text(json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "campaign_dir": str(output_dir),
        "jobs": len(jobs),
        "configs": len(config_paths),
        "manifest": str(output_dir / "appendix_a_manifest.csv"),
        "run_script": str(run_script),
    }


def _appendix_a_global_mismatch_source_rows(campaign_dir: str | Path) -> list[dict[str, str]]:
    campaign_dir = Path(campaign_dir)
    table = campaign_dir / "appendix_a_local_fit_summary.csv"
    if not table.exists():
        collect_appendix_a_outputs(campaign_dir)
    rows = _read_table(table)
    grouped: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        group_name = row.get("group", "")
        template_name = row.get("template", "")
        if group_name not in {"equal-mass", "nonspinning"} or template_name not in {"HypTan", "RatExp"}:
            continue
        if row.get("target") not in _appendix_a_targets(template_name):
            continue
        if row.get("value") in (None, ""):
            continue
        sxs_number = str(row.get("sxs_number") or row.get("sxs_id", "").split(":")[-1]).zfill(4)
        grouped.setdefault((group_name, template_name, sxs_number), row)
    return [grouped[key] for key in sorted(grouped)]


def _appendix_a_global_mismatch_fixed_coefficients(
    fits: dict[str, Any],
    group_name: str,
    template_name: str,
    x_value: float,
) -> dict[str, float]:
    template_fits = fits.get("fits", {}).get(group_name, {}).get(template_name, {})
    coefficients = {}
    missing_targets = []
    for target in _appendix_a_targets(template_name):
        fit = template_fits.get(target)
        if fit is None:
            missing_targets.append(target)
            continue
        coefficients[target] = float(_evaluate_appendix_fit(fit, x_value))
    if missing_targets:
        raise ValueError(
            "Appendix A global mismatch preparation needs fitted coefficients "
            f"for {group_name}/{template_name}; missing: {', '.join(missing_targets)}."
        )
    return coefficients


def _appendix_a_global_mismatch_config_text(
    sxs_number: str,
    template_name: str,
    outdir: Path,
    properties_file: Path,
    fixed_coefficients: dict[str, float] | None,
    teob_global_fit: int,
    merger_data: int,
    download: bool,
    n_random_seeds: int,
    nlive: int,
    maxmcmc: int,
    t_start: float,
    t_end: float,
    fits_file: Path | None = None,
) -> str:
    fits_file_line = f"fits-file         = {fits_file}\n" if fits_file is not None else ""
    prior_lines = []
    for target, value in sorted((fixed_coefficients or {}).items()):
        prior_lines.append(f"fix-{target}_22       = {value:.17g}")
    priors = "\n".join(prior_lines)
    if priors:
        priors += "\n"
    return f"""[I/O]
outdir            = {outdir}
screen-output     = 0
run-type          = full

[NR-data]
catalog           = SXS
ID                = {sxs_number}
l-NR              = 2
m                 = 2
error             = late-time-const-error
download          = {1 if download else 0}
properties-file   = {properties_file}
{fits_file_line}
[Model]
template          = TEOBPM
TEOB-template     = {template_name}
TEOB-calibration  = qc
TEOB-global-fit   = {teob_global_fit}
TEOB-merger-data  = {merger_data}

[Inference]
method            = Minimization
min-method        = trf
n-random-seeds    = {n_random_seeds}
nlive             = {nlive}
maxmcmc           = {maxmcmc}
t-start           = {t_start}
t-end             = {t_end}

[Priors]
{priors}"""


def prepare_appendix_a_global_mismatch_campaign(
    campaign_dir: str | Path,
    output_dir: str | Path | None = None,
    nc_ringdown_dir: str | Path = "/private/tmp/nc_ringdown",
    n_random_seeds: int = 4,
    nlive: int = 128,
    maxmcmc: int = 128,
    t_start: float = 0.0,
    t_end: float = 80.0,
    download: bool = True,
    bayring_executable: str = "bayRing",
    current_rao_fit_file: str | Path | None = None,
) -> dict[str, Any]:
    campaign_dir = Path(campaign_dir)
    output_dir = Path(output_dir) if output_dir is not None else campaign_dir / "appendix_a_global_mismatch"
    output_dir.mkdir(parents=True, exist_ok=True)
    fits_path = campaign_dir / "appendix_a_global_fits.json"
    if not fits_path.exists():
        construct_appendix_a_global_fits(campaign_dir)
    fits = json.loads(fits_path.read_text(encoding="utf-8"))
    source_rows = _appendix_a_global_mismatch_source_rows(campaign_dir)
    current_rao_path = _appendix_a_resolve_current_rao_fit_file(nc_ringdown_dir, current_rao_fit_file)
    _appendix_a_load_current_rao_fits(nc_ringdown_dir, fit_file=current_rao_path)

    jobs: list[LocalFitJob] = []
    config_paths: list[Path] = []
    manifest_rows: list[dict[str, Any]] = []
    scenarios = ("new_global", "existing_teobpm")
    for row in source_rows:
        group_name = row["group"]
        template_name = row["template"]
        sxs_number = str(row.get("sxs_number") or row.get("sxs_id", "").split(":")[-1]).zfill(4)
        x_value = _as_float(row.get("x_value"), default=math.nan)
        if not math.isfinite(x_value):
            raise ValueError(f"Missing Appendix A coordinate for SXS:{sxs_number} {group_name}/{template_name}.")
        fixed_coefficients = _appendix_a_global_mismatch_fixed_coefficients(fits, group_name, template_name, x_value)
        properties_text = row.get("properties_file") or str(_appendix_a_properties_file(nc_ringdown_dir, group_name))
        properties_file = Path(properties_text)
        q = _as_float(row.get("q"), default=math.nan)
        nu = _as_float(row.get("nu"), default=math.nan)
        chi1z = _as_float(row.get("chi1z"), default=math.nan)
        chi2z = _as_float(row.get("chi2z"), default=math.nan)
        chi_eff = _as_float(row.get("chi_eff") or row.get("chieff"), default=math.nan)
        chi_a = _as_float(row.get("chi_a"), default=math.nan)
        if not math.isfinite(chi_a) and math.isfinite(chi1z) and math.isfinite(chi2z):
            chi_a = 0.5 * (chi1z - chi2z)
        for scenario in scenarios:
            teob_global_fit = 0 if scenario == "new_global" else 1
            merger_data = 1 if _teobpm_requires_merger_peak_data(template_name) else 0
            scenario_fixed = fixed_coefficients if scenario == "new_global" else None
            fits_file = current_rao_path if scenario == "existing_teobpm" and template_name == "RatExp" else None
            campaign_name = _appendix_a_campaign_name(group_name, template_name)
            outdir = output_dir / "local_fits" / scenario / campaign_name / f"SXS_{sxs_number}"
            config_path = output_dir / "local_fit_configs" / scenario / campaign_name / f"SXS_{sxs_number}.ini"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                _appendix_a_global_mismatch_config_text(
                    sxs_number=sxs_number,
                    template_name=template_name,
                    outdir=outdir,
                    properties_file=properties_file,
                    fixed_coefficients=scenario_fixed,
                    teob_global_fit=teob_global_fit,
                    merger_data=merger_data,
                    download=download,
                    n_random_seeds=n_random_seeds,
                    nlive=nlive,
                    maxmcmc=maxmcmc,
                    t_start=t_start,
                    t_end=t_end,
                    fits_file=fits_file,
                ),
                encoding="utf-8",
            )
            config_paths.append(config_path)
            job = LocalFitJob(
                sxs_id=f"SXS:BBH:{sxs_number}",
                mode="22",
                split=scenario,
                config_file=str(config_path),
                outdir=str(outdir),
                q=q,
                nu=nu,
                chi1z=chi1z,
                chi2z=chi2z,
                chi_eff=chi_eff,
                chi_a=chi_a,
                eccentricity=_as_float(row.get("eccentricity"), default=0.0),
                quality_score=0.0,
            )
            jobs.append(job)
            manifest_row: dict[str, Any] = {
                "schema": APPENDIX_A_SCHEMA,
                "kind": "appendix-a-global-mismatch",
                "source_campaign_dir": str(campaign_dir),
                "scenario": scenario,
                "comparison_label": "New global fit" if scenario == "new_global" else "Existing TEOBPM global fit",
                "campaign": campaign_name,
                "group": group_name,
                "family": "spinning" if group_name == "equal-mass" else "nonspinning",
                "template": template_name,
                "sxs_id": job.sxs_id,
                "sxs_number": sxs_number,
                "outdir": job.outdir,
                "config_file": job.config_file,
                "x_name": row.get("x_name", "S_hat" if group_name == "equal-mass" else "nu"),
                "x_value": x_value,
                "q": q,
                "nu": nu,
                "chi1z": chi1z,
                "chi2z": chi2z,
                "chi_eff": chi_eff,
                "chi_a": chi_a,
                "properties_file": str(properties_file),
                "method": "Minimization",
            }
            for target, value in fixed_coefficients.items():
                manifest_row[f"new_global_{target}"] = value
            manifest_rows.append(manifest_row)

    write_local_fit_index(jobs, output_dir / "local_fit_index.csv")
    _write_rows_csv(manifest_rows, output_dir / "appendix_a_global_mismatch_manifest.csv")
    script_config = CalibrationConfig(family="aligned-spin", output_dir=str(output_dir), bayring_executable=bayring_executable)
    run_script = write_run_script(config_paths, script_config)
    config_payload = {
        "schema": APPENDIX_A_SCHEMA,
        "kind": "appendix-a-global-mismatch",
        "source_campaign_dir": str(campaign_dir),
        "global_fit_file": str(fits_path),
        "nc_ringdown_dir": str(nc_ringdown_dir),
        "current_rao_fit_file": str(current_rao_path),
        "teob_calibration": "qc",
        "method": "Minimization",
        "download": bool(download),
        "n_random_seeds": n_random_seeds,
        "nlive": nlive,
        "maxmcmc": maxmcmc,
        "t_start": t_start,
        "t_end": t_end,
        "bayring_executable": bayring_executable,
    }
    (output_dir / "campaign_config.json").write_text(json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "campaign_dir": str(output_dir),
        "source_campaign_dir": str(campaign_dir),
        "jobs": len(jobs),
        "configs": len(config_paths),
        "manifest": str(output_dir / "appendix_a_global_mismatch_manifest.csv"),
        "run_script": str(run_script),
        "method": "Minimization",
    }


def _appendix_a_global_mismatch_manifest_by_outdir(campaign_dir: str | Path) -> dict[str, dict[str, str]]:
    rows = _read_table(Path(campaign_dir) / "appendix_a_global_mismatch_manifest.csv")
    return {row["outdir"]: row for row in rows}


def collect_appendix_a_global_mismatch_outputs(
    campaign_dir: str | Path,
    output_table: str | Path | None = None,
    plot_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    campaign_dir = Path(campaign_dir)
    manifest_by_outdir = _appendix_a_global_mismatch_manifest_by_outdir(campaign_dir)
    long_rows: list[dict[str, Any]] = []
    wide_rows_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for job in read_local_fit_index(campaign_dir / "local_fit_index.csv"):
        metadata = manifest_by_outdir.get(job.outdir, {})
        mismatch_rows = collect_mismatch_rows(job)
        mismatch = representative_mismatch(mismatch_rows)
        scenario = metadata.get("scenario") or job.split
        row = {
            **metadata,
            "scenario": scenario,
            "mode": job.mode,
            "mismatch": mismatch if mismatch is not None else math.nan,
            "mismatch_rows": len(mismatch_rows),
        }
        long_rows.append(row)
        key = (str(metadata.get("sxs_id") or job.sxs_id), str(metadata.get("group") or ""), str(metadata.get("template") or ""))
        wide_row = wide_rows_by_key.setdefault(key, {
            "sxs_id": metadata.get("sxs_id") or job.sxs_id,
            "family": metadata.get("family") or _mismatch_family_from_row(metadata),
            "group": metadata.get("group", ""),
            "template": metadata.get("template", ""),
            "mode": job.mode,
            "q": metadata.get("q", job.q),
            "nu": metadata.get("nu", job.nu),
            "chi_eff": metadata.get("chi_eff", job.chi_eff),
            "chi_a": metadata.get("chi_a", job.chi_a),
            "x_name": metadata.get("x_name", ""),
            "x_value": metadata.get("x_value", ""),
        })
        if mismatch is not None:
            if scenario == "new_global":
                wide_row["new_global_mismatch"] = float(mismatch)
            elif scenario == "existing_teobpm":
                wide_row["existing_teobpm_mismatch"] = float(mismatch)

    long_path = campaign_dir / "appendix_a_global_mismatch_run_points.csv"
    wide_path = Path(output_table) if output_table else campaign_dir / "appendix_a_new_vs_existing_mismatch_points.csv"
    _write_rows_csv(long_rows, long_path)
    wide_rows = [row for row in wide_rows_by_key.values() if "new_global_mismatch" in row or "existing_teobpm_mismatch" in row]
    _write_rows_csv(wide_rows, wide_path)
    plot_dir = Path(plot_output_dir) if plot_output_dir else campaign_dir / "new_vs_existing_plots"
    plot_summary = plot_teobpm_mismatch_comparison([wide_path], plot_dir) if wide_rows else {}
    payload = {
        "campaign_dir": str(campaign_dir),
        "long_table": str(long_path),
        "mismatch_table": str(wide_path),
        "plot_summary": plot_summary,
        "jobs": len(long_rows),
        "completed_mismatches": sum(1 for row in long_rows if math.isfinite(_as_float(row.get("mismatch"), default=math.nan))),
        "plots": plot_summary.get("plots", []) if isinstance(plot_summary, dict) else [],
    }
    (campaign_dir / "appendix_a_global_mismatch_collection_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _appendix_manifest_by_outdir(campaign_dir: str | Path) -> dict[str, dict[str, str]]:
    rows = _read_table(Path(campaign_dir) / "appendix_a_manifest.csv")
    return {row["outdir"]: row for row in rows}


def collect_appendix_a_outputs(campaign_dir: str | Path) -> dict[str, Any]:
    campaign_dir = Path(campaign_dir)
    summary = collect_local_fit_outputs(campaign_dir)
    manifest_by_outdir = _appendix_manifest_by_outdir(campaign_dir)
    rows = []
    for row in _read_table(campaign_dir / "local_fit_summary.csv"):
        metadata = manifest_by_outdir.get(row.get("outdir", ""), {})
        rows.append({**metadata, **row})
    output = campaign_dir / "appendix_a_local_fit_summary.csv"
    _write_rows_csv(rows, output)
    summary["appendix_a_table"] = str(output)
    (campaign_dir / "appendix_a_collection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _fit_appendix_polynomial(rows: list[dict[str, Any]], degree: int) -> dict[str, Any]:
    x = np.array([_as_float(row["x_value"]) for row in rows])
    y = np.array([_as_float(row["value"]) for row in rows])
    sigma = np.array([_as_float(row.get("sigma"), default=math.nan) for row in rows])
    weights = np.array([1.0 / value if math.isfinite(float(value)) and float(value) > 0.0 else 1.0 for value in sigma])
    design = [[float(weight) * float(x_value) ** power for power in range(degree + 1)] for x_value, weight in zip(x, weights)]
    weighted_values = [float(weight) * float(value) for value, weight in zip(y, weights)]
    coeffs = _least_squares(design, weighted_values)
    predictions = [sum(coeffs[power] * (_as_float(row["x_value"]) ** power) for power in range(len(coeffs))) for row in rows]
    residuals = [_as_float(row["value"]) - pred for row, pred in zip(rows, predictions)]
    relative_residuals = [
        100.0 * residual / _as_float(row["value"])
        for row, residual in zip(rows, residuals)
        if abs(_as_float(row["value"])) > 1.0e-14
    ]
    return {
        "degree": degree,
        "coefficients": [{"power": power, "coefficient": coeff} for power, coeff in enumerate(coeffs)],
        "rss": float(sum(value * value for value in residuals)),
        "median_abs_relative_residual_pct": _median_abs(relative_residuals),
        "max_abs_relative_residual_pct": _max_abs(relative_residuals),
        "n": len(rows),
    }


def _evaluate_appendix_fit(fit: dict[str, Any], x_value: float) -> float:
    return sum(float(term["coefficient"]) * x_value ** int(term["power"]) for term in fit["coefficients"])


def construct_appendix_a_global_fits(campaign_dir: str | Path, output_file: str | Path | None = None) -> dict[str, Any]:
    campaign_dir = Path(campaign_dir)
    table = campaign_dir / "appendix_a_local_fit_summary.csv"
    if not table.exists():
        collect_appendix_a_outputs(campaign_dir)
    rows = _read_table(table)
    payload: dict[str, Any] = {"schema": APPENDIX_A_SCHEMA, "calibration": "qc", "fits": {}}
    for group_name in ["equal-mass", "nonspinning"]:
        x_name = "S_hat" if group_name == "equal-mass" else "nu"
        for template_name in ["HypTan", "RatExp"]:
            for target in _appendix_a_targets(template_name):
                degree = _appendix_a_degree(group_name, template_name, target)
                target_rows = [
                    row for row in rows
                    if row.get("group") == group_name
                    and row.get("template") == template_name
                    and row.get("target") == target
                    and row.get("value") not in (None, "")
                ]
                if len(target_rows) < degree + 1:
                    continue
                fit = _fit_appendix_polynomial(target_rows, degree)
                fit["x_name"] = x_name
                payload["fits"].setdefault(group_name, {}).setdefault(template_name, {})[target] = fit
    output_path = Path(output_file) if output_file else campaign_dir / "appendix_a_global_fits.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def plot_appendix_a_diagnostics(campaign_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    campaign_dir = Path(campaign_dir)
    output_dir = Path(output_dir) if output_dir else campaign_dir / "appendix_a_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    table = campaign_dir / "appendix_a_local_fit_summary.csv"
    if not table.exists():
        collect_appendix_a_outputs(campaign_dir)
    rows = _read_table(table)
    fits = construct_appendix_a_global_fits(campaign_dir)
    plot_paths = []

    plt.rcParams.update({
        "figure.figsize": (7.2, 5.6),
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    })

    for group_name in ["equal-mass", "nonspinning"]:
        x_label = r"$\hat{S}$" if group_name == "equal-mass" else r"$\nu$"
        for template_name in ["HypTan", "RatExp"]:
            for target in _appendix_a_targets(template_name):
                target_rows = [
                    row for row in rows
                    if row.get("group") == group_name
                    and row.get("template") == template_name
                    and row.get("target") == target
                    and row.get("value") not in (None, "")
                ]
                if not target_rows:
                    continue
                target_rows.sort(key=lambda item: _as_float(item["x_value"]))
                x = np.array([_as_float(row["x_value"]) for row in target_rows], dtype=float)
                y = np.array([_as_float(row["value"]) for row in target_rows], dtype=float)
                sigma = np.array([_as_float(row.get("sigma"), default=math.nan) for row in target_rows], dtype=float)
                mismatch = np.array([_as_float(row.get("mismatch"), default=math.nan) for row in target_rows], dtype=float)
                finite_mismatch = np.array([math.isfinite(float(value)) and float(value) > 0.0 for value in mismatch])

                fig = plt.figure(figsize=(7.2, 5.6))
                grid_spec = fig.add_gridspec(nrows=4, ncols=2, width_ratios=[1.0, 0.05], hspace=0.0, wspace=0.04)
                ax_main = fig.add_subplot(grid_spec[:3, 0])
                ax_residual = fig.add_subplot(grid_spec[3, 0], sharex=ax_main)
                cax = fig.add_subplot(grid_spec[:, 1]) if finite_mismatch.any() else None

                finite_sigma = np.array([math.isfinite(float(value)) and float(value) > 0.0 for value in sigma])
                if finite_sigma.any():
                    ax_main.errorbar(
                        x[finite_sigma],
                        y[finite_sigma],
                        yerr=sigma[finite_sigma],
                        fmt="none",
                        ecolor="#697386",
                        elinewidth=0.8,
                        capsize=2.5,
                        zorder=0,
                    )

                if finite_mismatch.any():
                    scatter = ax_main.scatter(
                        x,
                        y,
                        c=np.where(finite_mismatch, np.log10(mismatch), np.nan),
                        cmap="inferno_r",
                        edgecolor="black",
                        linewidth=0.25,
                        s=52,
                        label="Local fits",
                    )
                    fig.colorbar(scatter, cax=cax, label=r"$\log_{10}\mathcal{M}$")
                else:
                    ax_main.scatter(x, y, color="#293241", edgecolor="black", linewidth=0.25, s=52, label="Local fits")

                fit = fits.get("fits", {}).get(group_name, {}).get(template_name, {}).get(target)
                if fit:
                    xmin, xmax = min(x), max(x)
                    grid = np.linspace(xmin, xmax, 200)
                    fit_values = np.array([_evaluate_appendix_fit(fit, float(value)) for value in grid])
                    point_predictions = np.array([_evaluate_appendix_fit(fit, float(value)) for value in x])
                    ax_main.plot(grid, fit_values, color="#2364aa", ls="-.", lw=1.9, label="New global fit")
                    residual = np.full_like(y, np.nan, dtype=float)
                    nonzero = np.abs(y) > 1.0e-14
                    residual[nonzero] = 100.0 * (y[nonzero] - point_predictions[nonzero]) / y[nonzero]
                    ax_residual.scatter(x, residual, color="#2364aa", edgecolor="white", linewidth=0.3, s=34)

                past_values = None
                if template_name == "HypTan":
                    past_values = _appendix_a_past_hyptan_value(group_name, target, x)
                    past_grid = _appendix_a_past_hyptan_value(group_name, target, np.linspace(min(x), max(x), 200))
                    if past_grid is not None:
                        grid = np.linspace(min(x), max(x), 200)
                        ax_main.plot(grid, past_grid, color="#0b8f83", ls="--", lw=1.8, label="Past TEOB fit")
                    if past_values is not None:
                        residual = np.full_like(y, np.nan, dtype=float)
                        nonzero = np.abs(y) > 1.0e-14
                        residual[nonzero] = 100.0 * (y[nonzero] - np.array(past_values)[nonzero]) / y[nonzero]
                        ax_residual.scatter(x, residual, marker="x", color="#0b8f83", s=42)

                ax_residual.axhline(0.0, color="#3b4252", lw=0.8, alpha=0.5)
                ax_main.set_ylabel(target)
                ax_residual.set_ylabel("Res. [%]")
                ax_residual.set_xlabel(x_label)
                ax_main.tick_params(axis="x", labelbottom=False)
                ax_main.legend(loc="best", fontsize=8, frameon=False)
                fig.suptitle(f"{template_name} {target}: {group_name}", y=0.985, fontsize=11)
                fig.tight_layout()

                suffix = "chi" if group_name == "equal-mass" else "nu"
                prefix = f"{target}_qc_fit_sxs" if template_name == "HypTan" else f"{target}_fit_sxs"
                path = output_dir / f"{prefix}_{suffix}.png"
                fig.savefig(path, dpi=240)
                plt.close(fig)
                plot_paths.append(path)

    for group_name in ["equal-mass", "nonspinning"]:
        fig, ax = plt.subplots()
        histogram_values: list[tuple[str, list[float]]] = []
        for template_name, marker in [("HypTan", "x"), ("RatExp", "o")]:
            target_rows = [
                row for row in rows
                if row.get("group") == group_name
                and row.get("template") == template_name
                and row.get("target") == _appendix_a_targets(template_name)[0]
                and row.get("mismatch") not in (None, "")
            ]
            if not target_rows:
                continue
            values = [_as_float(row["mismatch"]) for row in target_rows]
            values = [value for value in values if math.isfinite(value) and value > 0.0]
            if not values:
                continue
            histogram_values.append((template_name, values))
            ax.scatter(
                [_as_float(row["x_value"]) for row in target_rows],
                [_as_float(row["mismatch"]) for row in target_rows],
                marker=marker,
                label=f"{template_name} local",
                s=42,
            )
            ax.axhline(float(np.median(values)), lw=1.2, ls="--", alpha=0.45)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\hat{S}$" if group_name == "equal-mass" else r"$\nu$")
        ax.set_ylabel("Representative mismatch")
        ax.legend(loc="best", frameon=False)
        fig.tight_layout()
        path = output_dir / f"appendix_a_mismatch_{group_name.replace('-', '_')}.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        plot_paths.append(path)

        if histogram_values:
            fig, ax = plt.subplots()
            bins = min(18, max(5, int(math.sqrt(sum(len(values) for _, values in histogram_values))) + 2))
            for template_name, values in histogram_values:
                ax.hist(
                    np.log10(values),
                    bins=bins,
                    alpha=0.58,
                    label=f"{template_name} local",
                    edgecolor="white",
                )
            ax.set_xlabel(r"$\log_{10}\mathcal{M}$")
            ax.set_ylabel("Count")
            ax.legend(loc="best", frameon=False)
            fig.tight_layout()
            path = output_dir / f"appendix_a_mismatch_histogram_{group_name.replace('-', '_')}.png"
            fig.savefig(path, dpi=220)
            plt.close(fig)
            plot_paths.append(path)

    summary = {"plots": [str(path) for path in plot_paths], "global_fit_file": str(campaign_dir / "appendix_a_global_fits.json")}
    (output_dir / "appendix_a_plot_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _appendix_a_nonspinning_target_rows(rows: list[dict[str, Any]], template: str, target: str) -> list[dict[str, Any]]:
    target_rows = [
        row for row in rows
        if row.get("group") == "nonspinning"
        and row.get("template") == template
        and row.get("target") == target
        and row.get("value") not in (None, "")
        and row.get("x_value") not in (None, "")
    ]
    target_rows.sort(key=lambda item: (_as_float(item.get("x_value")), item.get("sxs_id", "")))
    return target_rows


def _appendix_a_enrich_rows_with_properties(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cache: dict[Path, dict[str, dict[str, str]]] = {}
    enriched = []
    for row in rows:
        properties_file = row.get("properties_file")
        sxs_number = str(row.get("sxs_number") or row.get("sxs_id", "").split(":")[-1]).zfill(4)
        metadata = {}
        if properties_file:
            path = Path(properties_file)
            if path.exists():
                if path not in cache:
                    cache[path] = {str(item.get("ID", "")).zfill(4): item for item in _rows_from_csv(path)}
                metadata = cache[path].get(sxs_number, {})
        enriched.append({**metadata, **row})
    return enriched


def _appendix_a_rao_variable_value(row: dict[str, Any], variable: str) -> float:
    for column in APPENDIX_A_RAO_VARIABLE_COLUMNS.get(variable, (variable,)):
        value = row.get(column)
        if value not in (None, ""):
            return _as_float(value, default=math.nan)
    return math.nan


def _appendix_a_current_rao_value(fit: dict[str, Any], row: dict[str, Any]) -> float:
    variables = _appendix_a_fit_type_variables(str(fit.get("fit_type", "")), fit.get("source_file", "current Rao/Carullo fit"))
    values = []
    for variable in variables:
        raw_value = _appendix_a_rao_variable_value(row, variable)
        if not math.isfinite(raw_value):
            return math.nan
        norm = fit.get("norms", {}).get(variable, {})
        values.append(raw_value * _as_float(norm.get("scale"), default=1.0) + _as_float(norm.get("shift"), default=0.0))

    coefficients = fit.get("coefficients_by_index", {})
    result = float(coefficients.get("p0", 0.0))
    coefficient_index = 1
    for degree in range(1, int(fit.get("order", 0)) + 1):
        for indices in itertools.combinations_with_replacement(range(len(values)), degree):
            term = 1.0
            for index in indices:
                term *= values[index]
            result += float(coefficients.get(f"p{coefficient_index}", 0.0)) * term
            coefficient_index += 1
    return result


def _finite_percent_residual(observed: Iterable[float], prediction: Iterable[float]) -> list[float]:
    residual = []
    for observed_value, prediction_value in zip(observed, prediction):
        observed_float = float(observed_value)
        prediction_float = float(prediction_value)
        if math.isfinite(observed_float) and math.isfinite(prediction_float) and abs(observed_float) > 1.0e-14:
            residual.append(100.0 * (observed_float - prediction_float) / observed_float)
        else:
            residual.append(math.nan)
    return residual


def _finite_percent_delta(reference: Iterable[float], prediction: Iterable[float]) -> list[float]:
    delta = []
    for reference_value, prediction_value in zip(reference, prediction):
        reference_float = float(reference_value)
        prediction_float = float(prediction_value)
        if math.isfinite(reference_float) and math.isfinite(prediction_float) and abs(reference_float) > 1.0e-14:
            delta.append(100.0 * (prediction_float - reference_float) / reference_float)
        else:
            delta.append(math.nan)
    return delta


def _median_abs(values: Iterable[float]) -> float:
    finite = sorted(abs(float(value)) for value in values if math.isfinite(float(value)))
    if not finite:
        return math.nan
    middle = len(finite) // 2
    if len(finite) % 2:
        return float(finite[middle])
    return float(0.5 * (finite[middle - 1] + finite[middle]))


def _max_abs(values: Iterable[float]) -> float:
    finite = [abs(float(value)) for value in values if math.isfinite(float(value))]
    return float(max(finite)) if finite else math.nan


def _float_sequence(values: Any) -> list[float]:
    if isinstance(values, (int, float)):
        return [float(values)]
    try:
        return [float(value) for value in values]
    except TypeError:
        return [float(values)]


def _appendix_a_reference_values(
    template: str,
    target: str,
    x: np.ndarray,
    rao_fits: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]] | None = None,
) -> tuple[list[float] | None, str, str]:
    x_eval = np.array(x) if not isinstance(x, (int, float)) else x
    if template == "HypTan":
        values = _appendix_a_past_hyptan_value("nonspinning", target, x_eval)
        if values is None:
            return None, "", ""
        return _float_sequence(values), "Current TEOB HypTan", "embedded Appendix A TEOB HypTan fit"
    if template == "RatExp":
        fit = rao_fits.get(target)
        if fit and fit.get("reference_type") == "current-nc":
            if rows is None:
                return None, "", ""
            source_file = str(fit["source_file"])
            fit_type = str(fit["fit_type"])
            order = int(fit["order"])
            values = [_appendix_a_current_rao_value(fit, row) for row in rows]
            return values, f"Rao/Carullo current {fit_type} order {order}", source_file
        values = _appendix_a_rao_value(rao_fits, target, x_eval)
        if values is None:
            return None, "", ""
        source_file = str(rao_fits[target]["source_file"])
        order = int(rao_fits[target]["order"])
        catalog = str(rao_fits[target]["catalog"]).upper()
        return _float_sequence(values), f"Rao/Carullo {catalog} order {order}", source_file
    return None, "", ""


def _maybe_symlog_axis(axis: Any, values: Iterable[float], ratio_threshold: float = 80.0) -> None:
    finite_abs = [abs(float(value)) for value in values if math.isfinite(float(value)) and abs(float(value)) > 0.0]
    if finite_abs and max(finite_abs) / min(finite_abs) > ratio_threshold:
        from matplotlib.ticker import MaxNLocator, NullLocator

        axis.set_yscale("symlog", linthresh=max(min(finite_abs) * 0.75, 1.0e-6))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
        axis.yaxis.set_minor_locator(NullLocator())


def _padded_limits(values: Iterable[float], pad_fraction: float = 0.12) -> tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return -1.0, 1.0
    lower = min(finite)
    upper = max(finite)
    if abs(upper - lower) < 1.0e-14:
        pad = max(abs(upper) * pad_fraction, 1.0e-3)
    else:
        pad = (upper - lower) * pad_fraction
    return lower - pad, upper + pad


def _symmetric_limits(values: Iterable[float], pad_fraction: float = 0.15) -> tuple[float, float]:
    finite_abs = [abs(float(value)) for value in values if math.isfinite(float(value))]
    upper = max(finite_abs) if finite_abs else 1.0
    upper = max(upper * (1.0 + pad_fraction), 1.0)
    return -upper, upper


def _clipped_marker(value: float, lower: float, upper: float) -> tuple[float, str]:
    if value > upper:
        return upper, "^"
    if value < lower:
        return lower, "v"
    return value, "x"


def _appendix_a_compare_template_plot(
    rows: list[dict[str, Any]],
    fits: dict[str, Any],
    rao_fits: dict[str, dict[str, Any]],
    template: str,
    output_dir: Path,
) -> Path | None:
    import matplotlib.pyplot as plt

    targets = _appendix_a_targets(template)
    available_targets = [
        target for target in targets
        if _appendix_a_nonspinning_target_rows(rows, template, target)
    ]
    if not available_targets:
        return None

    ncols = len(available_targets)
    fig = plt.figure(figsize=(max(7.2, 3.15 * ncols), 5.7))
    grid_spec = fig.add_gridspec(nrows=2, ncols=ncols, height_ratios=[3.0, 1.0], hspace=0.05, wspace=0.28)
    legend_handles = None

    for column, target in enumerate(available_targets):
        ax_main = fig.add_subplot(grid_spec[0, column])
        ax_residual = fig.add_subplot(grid_spec[1, column], sharex=ax_main)
        target_rows = _appendix_a_nonspinning_target_rows(rows, template, target)
        x = np.array([_as_float(row["x_value"]) for row in target_rows], dtype=float)
        y = np.array([_as_float(row["value"]) for row in target_rows], dtype=float)
        sigma = np.array([_as_float(row.get("sigma"), default=math.nan) for row in target_rows], dtype=float)
        finite_sigma = np.isfinite(sigma) & (sigma > 0.0)
        main_axis_values = [float(value) for value in y]
        residual_axis_values: list[float] = []
        new_residual: list[float] = []

        if finite_sigma.any():
            ax_main.errorbar(
                x[finite_sigma],
                y[finite_sigma],
                yerr=sigma[finite_sigma],
                fmt="none",
                ecolor="#7f8794",
                elinewidth=0.75,
                capsize=2.0,
                zorder=0,
            )
        local = ax_main.scatter(x, y, color="#202a36", edgecolor="white", linewidth=0.35, s=42, label="Local fits", zorder=3)

        grid = np.linspace(float(np.min(x)), float(np.max(x)), 240)
        fit = fits.get("fits", {}).get("nonspinning", {}).get(template, {}).get(target)
        new_line = None
        if fit:
            new_grid = np.array([_evaluate_appendix_fit(fit, float(value)) for value in grid])
            new_points = np.array([_evaluate_appendix_fit(fit, float(value)) for value in x])
            main_axis_values.extend(float(value) for value in new_grid)
            new_line = ax_main.plot(grid, new_grid, color="#2364aa", lw=1.9, label="New global fit")[0]
            new_residual = _finite_percent_residual(y, new_points)
            residual_axis_values.extend(new_residual)
            ax_residual.scatter(
                x,
                new_residual,
                color="#2364aa",
                edgecolor="white",
                linewidth=0.25,
                s=28,
                label="New global fit",
            )

        reference_is_current = template == "RatExp" and rao_fits.get(target, {}).get("reference_type") == "current-nc"
        reference_x = x if reference_is_current else grid
        reference_rows = target_rows if reference_is_current else None
        reference_grid, reference_label, _ = _appendix_a_reference_values(template, target, reference_x, rao_fits, rows=reference_rows)
        reference_points, _, _ = _appendix_a_reference_values(template, target, x, rao_fits, rows=target_rows)
        reference_line = None
        if reference_grid is not None and reference_points is not None:
            reference_residual = _finite_percent_residual(y, reference_points)
            reference_scale = _max_abs(reference_grid) / max(_max_abs(main_axis_values), 1.0e-12)
            clip_reference = reference_is_current and reference_scale > 20.0
            if clip_reference:
                lower, upper = _padded_limits(main_axis_values)
                ax_main.set_ylim(lower, upper)
                first_marker = True
                for x_value, reference_value in zip(x, reference_points):
                    clipped_value, marker = _clipped_marker(float(reference_value), lower, upper)
                    label = f"{reference_label} (clipped)" if first_marker else None
                    reference_handle = ax_main.scatter(
                        [x_value],
                        [clipped_value],
                        color="#0b8f83",
                        marker=marker,
                        s=44,
                        label=label,
                        zorder=4,
                    )
                    if first_marker:
                        reference_line = reference_handle
                    first_marker = False

                residual_lower, residual_upper = _symmetric_limits(new_residual)
                ax_residual.set_ylim(residual_lower, residual_upper)
                first_marker = True
                for x_value, residual_value in zip(x, reference_residual):
                    clipped_value, marker = _clipped_marker(float(residual_value), residual_lower, residual_upper)
                    ax_residual.scatter(
                        [x_value],
                        [clipped_value],
                        color="#0b8f83",
                        marker=marker,
                        s=34,
                        label=f"{reference_label} (clipped)" if first_marker else None,
                    )
                    first_marker = False
            else:
                main_axis_values.extend(reference_grid)
                reference_line = ax_main.plot(reference_x, reference_grid, color="#0b8f83", lw=1.7, ls="--", label=reference_label)[0]
                residual_axis_values.extend(reference_residual)
                ax_residual.scatter(
                    x,
                    reference_residual,
                    color="#0b8f83",
                    marker="x",
                    s=34,
                    label=reference_label,
                )

        if ax_main.get_yscale() == "linear":
            _maybe_symlog_axis(ax_main, main_axis_values)
        if ax_residual.get_yscale() == "linear":
            _maybe_symlog_axis(ax_residual, residual_axis_values)
        ax_main.set_title(target, fontsize=10)
        ax_main.tick_params(axis="x", labelbottom=False)
        ax_main.grid(True, alpha=0.24)
        ax_residual.grid(True, alpha=0.24)
        ax_residual.axhline(0.0, color="#3b4252", lw=0.8, alpha=0.55)
        ax_residual.set_xlabel(r"$\nu$")
        if column == 0:
            ax_main.set_ylabel("Coefficient")
            ax_residual.set_ylabel("Res. [%]")
        else:
            ax_main.set_ylabel("")
            ax_residual.set_ylabel("")
        if legend_handles is None:
            legend_handles = [handle for handle in [local, new_line, reference_line] if handle is not None]

    if legend_handles:
        labels = [handle.get_label() for handle in legend_handles]
        fig.legend(legend_handles, labels, loc="upper center", ncol=min(3, len(legend_handles)), frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Appendix A nonspinning {template} global-fit comparison", y=1.05, fontsize=12)
    fig.tight_layout()
    path = output_dir / f"appendix_a_nonspinning_{template.lower()}_comparison.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def _appendix_a_compare_summary_plot(summary_rows: list[dict[str, Any]], output_dir: Path) -> Path | None:
    import matplotlib.pyplot as plt

    if not summary_rows:
        return None
    labels = [f"{row['template']}\n{row['target']}" for row in summary_rows]
    new_values = [float(row["new_median_abs_relative_residual_pct"]) for row in summary_rows]
    reference_values = [float(row["reference_median_abs_relative_residual_pct"]) for row in summary_rows]
    x = np.arange(len(summary_rows), dtype=float)
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(8.0, 0.82 * len(summary_rows)), 4.8))
    ax.bar(x - width / 2.0, new_values, width=width, color="#2364aa", label="New global fit")
    ax.bar(x + width / 2.0, reference_values, width=width, color="#0b8f83", label="Reference fit")
    finite_positive = [value for value in new_values + reference_values if math.isfinite(value) and value > 0.0]
    if finite_positive and max(finite_positive) / min(finite_positive) > 80.0:
        ax.set_yscale("log")
        ax.set_ylim(max(min(finite_positive) * 0.55, 1.0e-5), max(finite_positive) * 1.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Median absolute residual [%]")
    ax.set_title("Appendix A nonspinning fit residual comparison")
    ax.grid(axis="y", alpha=0.24)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    path = output_dir / "appendix_a_nonspinning_comparison_residual_summary.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def compare_appendix_a_nonspinning_fits(
    campaign_dir: str | Path,
    nc_ringdown_dir: str | Path = "/private/tmp/nc_ringdown",
    output_dir: str | Path | None = None,
    rao_reference: str = "current-nc",
    rao_catalog: str = "sxs",
    rao_order: int = 1,
    rao_fit_file: str | Path | None = None,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    campaign_dir = Path(campaign_dir)
    output_dir = Path(output_dir) if output_dir else campaign_dir / "appendix_a_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    table = campaign_dir / "appendix_a_local_fit_summary.csv"
    if not table.exists():
        collect_appendix_a_outputs(campaign_dir)
    rows = _appendix_a_enrich_rows_with_properties(_read_table(table))
    fits = construct_appendix_a_global_fits(campaign_dir)
    rao_fits = _appendix_a_load_rao_reference(
        nc_ringdown_dir,
        reference=rao_reference,
        catalog=rao_catalog,
        order=rao_order,
        fit_file=rao_fit_file,
    )

    plt.rcParams.update({
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    })

    point_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    plot_paths: list[Path] = []

    for template in ["HypTan", "RatExp"]:
        for target in _appendix_a_targets(template):
            target_rows = _appendix_a_nonspinning_target_rows(rows, template, target)
            fit = fits.get("fits", {}).get("nonspinning", {}).get(template, {}).get(target)
            if not target_rows or not fit:
                continue

            x = [_as_float(row["x_value"]) for row in target_rows]
            y = [_as_float(row["value"]) for row in target_rows]
            new_points = [float(_evaluate_appendix_fit(fit, float(value))) for value in x]
            reference_points, reference_label, reference_source = _appendix_a_reference_values(template, target, x, rao_fits, rows=target_rows)
            if reference_points is None:
                continue

            new_residual_pct = _finite_percent_residual(y, new_points)
            reference_residual_pct = _finite_percent_residual(y, reference_points)
            delta = [new_value - reference_value for new_value, reference_value in zip(new_points, reference_points)]
            delta_pct = _finite_percent_delta(reference_points, new_points)

            for row, new_value, reference_value, new_residual, reference_residual, delta_value, delta_pct_value in zip(
                target_rows,
                new_points,
                reference_points,
                new_residual_pct,
                reference_residual_pct,
                delta,
                delta_pct,
            ):
                point_rows.append({
                    "template": template,
                    "target": target,
                    "reference": reference_label,
                    "sxs_id": row.get("sxs_id", ""),
                    "nu": _as_float(row.get("x_value")),
                    "local_value": _as_float(row.get("value")),
                    "new_global_value": float(new_value),
                    "reference_value": float(reference_value),
                    "new_relative_residual_pct": float(new_residual),
                    "reference_relative_residual_pct": float(reference_residual),
                    "new_minus_reference": float(delta_value),
                    "new_minus_reference_pct": float(delta_pct_value),
                    "mismatch": _as_float(row.get("mismatch"), default=math.nan),
                })

            reference_is_current = template == "RatExp" and rao_fits.get(target, {}).get("reference_type") == "current-nc"
            grid = list(x) if reference_is_current else list(np.linspace(float(min(x)), float(max(x)), 240))
            grid_reference_rows = target_rows if reference_is_current else None
            new_grid = [float(_evaluate_appendix_fit(fit, float(value))) for value in grid]
            reference_grid, _, _ = _appendix_a_reference_values(template, target, grid, rao_fits, rows=grid_reference_rows)
            if reference_grid is not None:
                for x_value, new_value, reference_value in zip(grid, new_grid, reference_grid):
                    grid_rows.append({
                        "template": template,
                        "target": target,
                        "reference": reference_label,
                        "nu": float(x_value),
                        "new_global_value": float(new_value),
                        "reference_value": float(reference_value),
                        "new_minus_reference": float(new_value - reference_value),
                    })

            summary_rows.append({
                "template": template,
                "target": target,
                "reference": reference_label,
                "n": len(target_rows),
                "nu_min": float(min(x)),
                "nu_max": float(max(x)),
                "new_median_abs_relative_residual_pct": _median_abs(new_residual_pct),
                "new_max_abs_relative_residual_pct": _max_abs(new_residual_pct),
                "reference_median_abs_relative_residual_pct": _median_abs(reference_residual_pct),
                "reference_max_abs_relative_residual_pct": _max_abs(reference_residual_pct),
                "new_reference_median_abs_delta": _median_abs(delta),
                "new_reference_max_abs_delta": _max_abs(delta),
                "new_reference_median_abs_delta_pct": _median_abs(delta_pct),
                "new_reference_max_abs_delta_pct": _max_abs(delta_pct),
                "reference_source": reference_source,
            })

    for template in ["HypTan", "RatExp"]:
        path = _appendix_a_compare_template_plot(rows, fits, rao_fits, template, output_dir)
        if path is not None:
            plot_paths.append(path)
    summary_plot = _appendix_a_compare_summary_plot(summary_rows, output_dir)
    if summary_plot is not None:
        plot_paths.append(summary_plot)

    point_path = output_dir / "appendix_a_nonspinning_fit_comparison_points.csv"
    grid_path = output_dir / "appendix_a_nonspinning_fit_comparison_grid.csv"
    summary_path = output_dir / "appendix_a_nonspinning_fit_comparison_summary.csv"
    _write_rows_csv(point_rows, point_path)
    _write_rows_csv(grid_rows, grid_path)
    _write_rows_csv(summary_rows, summary_path)
    representative_rao_fit = next(iter(rao_fits.values()), {})
    payload = {
        "campaign_dir": str(campaign_dir),
        "global_fit_file": str(campaign_dir / "appendix_a_global_fits.json"),
        "rao_reference": rao_reference,
        "rao_catalog": rao_catalog,
        "rao_fit_type": str(representative_rao_fit.get("fit_type", "")),
        "rao_order": int(representative_rao_fit.get("order", rao_order)),
        "rao_fit_file": str(_appendix_a_rao_fit_path(nc_ringdown_dir, rao_catalog, rao_order))
        if rao_reference == "appendix-nu"
        else str(_appendix_a_resolve_current_rao_fit_file(nc_ringdown_dir, fit_file=rao_fit_file)),
        "point_table": str(point_path),
        "grid_table": str(grid_path),
        "summary_table": str(summary_path),
        "plots": [str(path) for path in plot_paths],
        "notes": (
            "HypTan coefficients are compared against the current TEOB HypTan nonspinning formulas. "
            "RatExp coefficients are compared against the selected Rao/Carullo nc_ringdown reference fit."
        ),
    }
    (output_dir / "appendix_a_nonspinning_fit_comparison_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _config_from_args(args: argparse.Namespace) -> CalibrationConfig:
    return CalibrationConfig(
        family=args.family,
        output_dir=args.output_dir,
        modes=parse_modes(args.modes),
        mode_mixing_modes=parse_modes(args.mode_mixing_modes) if getattr(args, "mode_mixing_modes", "") else [],
        counter_rotating_modes=parse_modes(args.counter_rotating_modes) if getattr(args, "counter_rotating_modes", "") else [],
        template=args.template,
        teob_calibration=args.teob_calibration,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        sxs_ids=sorted(set(parse_sxs_ids(getattr(args, "sxs_ids", "")) + read_sxs_id_file(getattr(args, "sxs_id_file", "")))),
        random_fraction=args.random_fraction,
        max_eccentricity=args.max_eccentricity,
        max_transverse_spin=args.max_transverse_spin,
        nonspinning_spin_tolerance=args.nonspinning_spin_tolerance,
        equal_mass_tolerance=args.equal_mass_tolerance,
        min_quality_score=args.min_quality_score,
        bayring_executable=args.bayring_executable,
        nr_data_dir=args.nr_data_dir,
        properties_file=args.properties_file,
        extrap_order=args.extrap_order,
        resolution_level=args.resolution_level,
        t_start=args.t_start,
        t_end=args.t_end,
        method=args.method,
        min_method=args.min_method,
        n_random_seeds=args.n_random_seeds,
        n_mode_workers=args.n_mode_workers,
        base_nonspinning_fit=args.base_nonspinning_fit or "",
        notes=args.notes,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and analyse TEOBPM SXS calibration campaigns.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Discover/filter SXS metadata and generate local-fit configs.")
    prepare.add_argument("--family", choices=["nonspinning", "equal-mass-spinning", "spinning", "aligned-spin"], required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--catalog-file", default=None)
    prepare.add_argument("--modes", default=",".join(_mode_label(mode) for mode in DEFAULT_MODES))
    prepare.add_argument("--mode-mixing-modes", default=",".join(_mode_label(mode) for mode in DEFAULT_MODE_MIXING_MODES), help="Comma-separated TEOBPM modes that should enable TEOB mode-mixing, e.g. 31,32,41,42,43.")
    prepare.add_argument("--counter-rotating-modes", default=",".join(_mode_label(mode) for mode in DEFAULT_COUNTER_ROTATING_MODES), help="Comma-separated TEOBPM modes that should add an independent counter-rotating contribution, e.g. 21.")
    prepare.add_argument("--template", choices=["HypTan", "RatExp", "SEOBNRv5"], default="RatExp")
    prepare.add_argument("--teob-calibration", choices=["qc", "noncirc"], default="qc")
    prepare.add_argument("--validation-fraction", type=float, default=0.33)
    prepare.add_argument("--seed", type=int, default=1234)
    prepare.add_argument("--sxs-ids", default="", help="Optional comma-separated SXS IDs to keep after loading the catalog.")
    prepare.add_argument("--sxs-id-file", default="", help="Optional text file with one SXS ID per line, or comma-separated IDs.")
    prepare.add_argument("--random-fraction", type=float, default=1.0, help="Randomly keep this fraction of the filtered catalog, using --seed.")
    prepare.add_argument("--max-eccentricity", type=float, default=1.0e-3)
    prepare.add_argument("--max-transverse-spin", type=float, default=1.0e-6)
    prepare.add_argument("--nonspinning-spin-tolerance", type=float, default=1.0e-6)
    prepare.add_argument("--equal-mass-tolerance", type=float, default=1.0e-6)
    prepare.add_argument("--min-quality-score", type=float, default=None)
    prepare.add_argument("--bayring-executable", default="bayRing")
    prepare.add_argument("--nr-data-dir", default="")
    prepare.add_argument("--properties-file", default="", help="Optional CSV with supplemental SXS metadata; TEOBPM merger peak quantities are computed from the loaded NR waveform.")
    prepare.add_argument("--extrap-order", type=int, default=2)
    prepare.add_argument("--resolution-level", type=int, default=-1)
    prepare.add_argument("--t-start", type=float, default=0.0)
    prepare.add_argument("--t-end", type=float, default=140.0)
    prepare.add_argument("--method", choices=["Minimization", "Nested-sampler"], default="Minimization")
    prepare.add_argument("--min-method", default="trf")
    prepare.add_argument("--n-random-seeds", type=int, default=16)
    prepare.add_argument("--n-mode-workers", type=int, default=1)
    prepare.add_argument("--base-nonspinning-fit", default="")
    prepare.add_argument("--notes", default="")

    fit = subparsers.add_parser("global-fit", help="Construct the versioned pyRing-ingestable global-fit file.")
    fit.add_argument("--local-fit-table", required=True)
    fit.add_argument("--output-file", required=True)
    fit.add_argument("--family", choices=["nonspinning", "equal-mass-spinning", "spinning", "aligned-spin"], required=True)
    fit.add_argument("--template", choices=["HypTan", "RatExp", "SEOBNRv5"], default="RatExp")
    fit.add_argument("--teob-calibration", choices=["qc", "noncirc"], default="qc")
    fit.add_argument("--base-nonspinning-file", default=None)
    fit.add_argument("--max-polynomial-degree", type=int, default=3)

    run = subparsers.add_parser("run-local-fits", help="Run the local-fit jobs listed in a prepared campaign.")
    run.add_argument("--campaign-dir", required=True)
    run.add_argument("--workers", type=int, default=_default_worker_count())
    run.add_argument("--bayring-executable", default=None)
    run.add_argument("--force", action="store_true")
    run.add_argument("--timeout", type=float, default=None)
    run.add_argument("--split", choices=["all", "training", "validation"], default="all")
    run.add_argument("--modes", default="", help="Optional comma-separated mode subset to run, e.g. 22 or 21,33.")

    collect = subparsers.add_parser("collect-local-fits", help="Collect point estimates and construction mismatch diagnostics from local-fit outputs.")
    collect.add_argument("--campaign-dir", required=True)
    collect.add_argument("--output-table", default=None)
    collect.add_argument("--split", choices=["all", "training", "validation"], default="all")

    fill_hm = subparsers.add_parser(
        "fill-hm-inputs",
        help="Fill higher-mode local-fit configs with t-peak-22 and mixed-mode parent local-fit values.",
    )
    fill_hm.add_argument("--campaign-dir", required=True)
    fill_hm.add_argument("--mode-mixing-modes", default=",".join(_mode_label(mode) for mode in DEFAULT_MODE_MIXING_MODES))

    appendix_prepare = subparsers.add_parser("appendix-a-prepare", help="Prepare the Appendix A quasi-circular SXS reproduction campaigns.")
    appendix_prepare.add_argument("--output-dir", required=True)
    appendix_prepare.add_argument("--nc-ringdown-dir", default="/private/tmp/nc_ringdown")
    appendix_prepare.add_argument("--group", choices=["all", "equal-mass", "nonspinning"], default="all")
    appendix_prepare.add_argument("--template", choices=["all", "HypTan", "RatExp"], default="all")
    appendix_prepare.add_argument("--method", choices=["Minimization", "Nested-sampler"], default="Minimization")
    appendix_prepare.add_argument("--n-random-seeds", type=int, default=16)
    appendix_prepare.add_argument("--nlive", type=int, default=128)
    appendix_prepare.add_argument("--maxmcmc", type=int, default=128)
    appendix_prepare.add_argument("--t-start", type=float, default=0.0)
    appendix_prepare.add_argument("--t-end", type=float, default=80.0)
    appendix_prepare.add_argument("--bayring-executable", default="bayRing")

    appendix_collect = subparsers.add_parser("appendix-a-collect", help="Collect Appendix A local fits and merge manifest metadata.")
    appendix_collect.add_argument("--campaign-dir", required=True)

    appendix_fit = subparsers.add_parser("appendix-a-global-fit", help="Construct Appendix A polynomial global fits.")
    appendix_fit.add_argument("--campaign-dir", required=True)
    appendix_fit.add_argument("--output-file", default=None)

    appendix_global_mismatch = subparsers.add_parser(
        "appendix-a-prepare-global-mismatch",
        help="Prepare fixed-coefficient Appendix A new-vs-existing TEOBPM mismatch runs.",
    )
    appendix_global_mismatch.add_argument("--campaign-dir", required=True)
    appendix_global_mismatch.add_argument("--output-dir", default=None)
    appendix_global_mismatch.add_argument("--nc-ringdown-dir", default="/private/tmp/nc_ringdown")
    appendix_global_mismatch.add_argument("--n-random-seeds", type=int, default=4)
    appendix_global_mismatch.add_argument("--nlive", type=int, default=128)
    appendix_global_mismatch.add_argument("--maxmcmc", type=int, default=128)
    appendix_global_mismatch.add_argument("--t-start", type=float, default=0.0)
    appendix_global_mismatch.add_argument("--t-end", type=float, default=80.0)
    appendix_global_mismatch.add_argument("--no-download", action="store_true")
    appendix_global_mismatch.add_argument("--bayring-executable", default="bayRing")
    appendix_global_mismatch.add_argument("--rao-fit-file", default=None)

    appendix_global_mismatch_collect = subparsers.add_parser(
        "appendix-a-collect-global-mismatch",
        help="Collect fixed-coefficient Appendix A mismatch runs and generate standard mismatch plots.",
    )
    appendix_global_mismatch_collect.add_argument("--campaign-dir", required=True)
    appendix_global_mismatch_collect.add_argument("--output-table", default=None)
    appendix_global_mismatch_collect.add_argument("--plot-output-dir", default=None)

    appendix_plot = subparsers.add_parser("appendix-a-plot", help="Plot Appendix A coefficient and mismatch diagnostics.")
    appendix_plot.add_argument("--campaign-dir", required=True)
    appendix_plot.add_argument("--output-dir", default=None)

    appendix_compare = subparsers.add_parser(
        "appendix-a-compare-nonspinning",
        help="Compare Appendix A nonspinning HypTan/RatExp global fits against current TEOB and Rao/Carullo references.",
    )
    appendix_compare.add_argument("--campaign-dir", required=True)
    appendix_compare.add_argument("--nc-ringdown-dir", default="/private/tmp/nc_ringdown")
    appendix_compare.add_argument("--output-dir", default=None)
    appendix_compare.add_argument("--rao-reference", choices=["current-nc", "appendix-nu"], default="current-nc")
    appendix_compare.add_argument("--rao-catalog", choices=["sxs", "rit"], default="sxs")
    appendix_compare.add_argument("--rao-order", type=int, default=1)
    appendix_compare.add_argument("--rao-fit-file", default=None)

    validate = subparsers.add_parser("validate", help="Validate a global fit against a validation table.")
    validate.add_argument("--global-fit-file", required=True)
    validate.add_argument("--validation-table", required=True)
    validate.add_argument("--output-dir", required=True)
    validate.add_argument("--mismatch-table", action="append", default=None, help="Evaluation mismatch CSV table. May be repeated; rows should declare t_start=0 or tref=peak22 when metadata are present.")
    validate.add_argument("--label", action="append", default=None, help="Series label for the corresponding --mismatch-table. May be repeated.")
    validate.add_argument("--family", choices=["auto", "all", "nonspinning", "spinning"], default="auto")

    mismatch_compare = subparsers.add_parser(
        "plot-mismatch-comparison",
        help="Plot new/global/reference TEOBPM mismatch tables over the calibration parameter space.",
    )
    mismatch_compare.add_argument("--mismatch-table", action="append", required=True, help="CSV table containing mismatch and parameter-space columns. May be repeated.")
    mismatch_compare.add_argument("--label", action="append", default=None, help="Series label for the corresponding --mismatch-table. May be repeated.")
    mismatch_compare.add_argument("--output-dir", required=True)
    mismatch_compare.add_argument("--family", choices=["auto", "all", "nonspinning", "spinning"], default="auto")

    mismatch_histograms = subparsers.add_parser(
        "plot-mismatch-histograms",
        help="Plot mismatch histograms per case and combined by mode.",
    )
    mismatch_histograms.add_argument("--mismatch-table", action="append", default=None, help="CSV table containing mismatch rows. May be repeated.")
    mismatch_histograms.add_argument("--campaign-root", action="append", default=None, help="Campaign root containing template/family/mismatch_summary.csv tables. May be repeated.")
    mismatch_histograms.add_argument("--label", action="append", default=None, help="Series label for the corresponding --mismatch-table. Must match the final table count when provided.")
    mismatch_histograms.add_argument("--output-dir", required=True)
    mismatch_histograms.add_argument("--bins", type=int, default=24)

    local_plot = subparsers.add_parser("plot-local-fits", help="Plot collected local-fit coefficients and optional 22-peak evaluation mismatch diagnostics.")
    local_plot.add_argument("--local-fit-table", required=True)
    local_plot.add_argument("--output-dir", required=True)
    local_plot.add_argument("--family", choices=["auto", "all", "nonspinning", "spinning"], default="auto")
    local_plot.add_argument("--mismatch-table", action="append", default=None, help="Evaluation mismatch CSV table. May be repeated; rows should declare t_start=0 or tref=peak22 when metadata are present.")
    local_plot.add_argument("--label", action="append", default=None, help="Series label for the corresponding --mismatch-table. May be repeated.")

    report = subparsers.add_parser("report", help="Render and optionally compile a campaign report.")
    report.add_argument("--campaign-dir", required=True)
    report.add_argument("--output-tex", required=True)
    report.add_argument("--no-compile", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare":
        summary = prepare_campaign(args.catalog_file, _config_from_args(args))
    elif args.command == "global-fit":
        payload = construct_global_fit(
            args.local_fit_table,
            args.output_file,
            args.family,
            template=args.template,
            teob_calibration=args.teob_calibration,
            base_nonspinning_file=args.base_nonspinning_file,
            max_polynomial_degree=args.max_polynomial_degree,
        )
        summary = {"output_file": args.output_file, "n_modes": len(payload["fits"])}
    elif args.command == "run-local-fits":
        summary = run_local_fits(
            args.campaign_dir,
            workers=args.workers,
            bayring_executable=args.bayring_executable,
            force=args.force,
            timeout=args.timeout,
            split=args.split,
            modes=parse_modes(args.modes) if args.modes else None,
        )
    elif args.command == "collect-local-fits":
        summary = collect_local_fit_outputs(args.campaign_dir, output_table=args.output_table, split=args.split)
    elif args.command == "fill-hm-inputs":
        summary = fill_higher_mode_inputs(args.campaign_dir, mode_mixing_modes=parse_modes(args.mode_mixing_modes))
    elif args.command == "appendix-a-prepare":
        summary = prepare_appendix_a_campaign(
            args.output_dir,
            nc_ringdown_dir=args.nc_ringdown_dir,
            group=args.group,
            template=args.template,
            method=args.method,
            n_random_seeds=args.n_random_seeds,
            nlive=args.nlive,
            maxmcmc=args.maxmcmc,
            t_start=args.t_start,
            t_end=args.t_end,
            bayring_executable=args.bayring_executable,
        )
    elif args.command == "appendix-a-collect":
        summary = collect_appendix_a_outputs(args.campaign_dir)
    elif args.command == "appendix-a-global-fit":
        payload = construct_appendix_a_global_fits(args.campaign_dir, output_file=args.output_file)
        summary = {"campaign_dir": args.campaign_dir, "groups": list(payload["fits"].keys())}
    elif args.command == "appendix-a-prepare-global-mismatch":
        summary = prepare_appendix_a_global_mismatch_campaign(
            args.campaign_dir,
            output_dir=args.output_dir,
            nc_ringdown_dir=args.nc_ringdown_dir,
            n_random_seeds=args.n_random_seeds,
            nlive=args.nlive,
            maxmcmc=args.maxmcmc,
            t_start=args.t_start,
            t_end=args.t_end,
            download=not args.no_download,
            bayring_executable=args.bayring_executable,
            current_rao_fit_file=args.rao_fit_file,
        )
    elif args.command == "appendix-a-collect-global-mismatch":
        summary = collect_appendix_a_global_mismatch_outputs(
            args.campaign_dir,
            output_table=args.output_table,
            plot_output_dir=args.plot_output_dir,
        )
    elif args.command == "appendix-a-plot":
        summary = plot_appendix_a_diagnostics(args.campaign_dir, output_dir=args.output_dir)
    elif args.command == "appendix-a-compare-nonspinning":
        summary = compare_appendix_a_nonspinning_fits(
            args.campaign_dir,
            nc_ringdown_dir=args.nc_ringdown_dir,
            output_dir=args.output_dir,
            rao_reference=args.rao_reference,
            rao_catalog=args.rao_catalog,
            rao_order=args.rao_order,
            rao_fit_file=args.rao_fit_file,
        )
    elif args.command == "validate":
        summary = validate_global_fit(
            args.global_fit_file,
            args.validation_table,
            args.output_dir,
            mismatch_tables=args.mismatch_table,
            mismatch_labels=args.label,
            family=args.family,
        )
    elif args.command == "plot-mismatch-comparison":
        summary = plot_teobpm_mismatch_comparison(
            args.mismatch_table,
            args.output_dir,
            labels=args.label,
            family=args.family,
        )
    elif args.command == "plot-mismatch-histograms":
        summary = plot_mismatch_histograms(
            args.mismatch_table,
            args.output_dir,
            labels=args.label,
            campaign_roots=args.campaign_root,
            bins=args.bins,
        )
    elif args.command == "plot-local-fits":
        summary = plot_local_fit_diagnostics(
            args.local_fit_table,
            args.output_dir,
            family=args.family,
            mismatch_tables=args.mismatch_table,
            mismatch_labels=args.label,
        )
    elif args.command == "report":
        pdf = render_report(args.campaign_dir, args.output_tex, compile_pdf=not args.no_compile)
        summary = {"output_tex": args.output_tex, "output_pdf": str(pdf) if pdf else ""}
    else:
        parser.error(f"Unknown command `{args.command}`.")
        return
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
