from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any

import numpy as np


@dataclass
class WaveformData:
    sxs_id: str
    lev: int
    extrapolation_order: str
    time: np.ndarray
    modes: dict[tuple[int, int], np.ndarray]
    ell_min: int
    ell_max: int
    metadata: dict[str, Any]


def extrapolation_order_to_int(order: str | int) -> int | str:
    if isinstance(order, int):
        return order
    text = str(order).strip()
    if text.upper().startswith("N"):
        return int(text[1:])
    return int(text)


def extrapolation_order_label(order: str | int) -> str:
    if isinstance(order, str) and order.strip().upper().startswith("N"):
        return order.strip().upper()
    value = extrapolation_order_to_int(order)
    return f"N{value}" if isinstance(value, int) else str(value)


def _call_sxs_write_config(*, download: bool, cache: bool) -> None:
    import sxs

    if not hasattr(sxs, "write_config"):
        return
    try:
        sxs.write_config(download=download, cache=cache, auto_supersede=False)
    except TypeError:
        try:
            sxs.write_config(download=download, cache=cache)
        except TypeError:
            return


def configure_sxs(config: Any) -> None:
    _call_sxs_write_config(download=bool(config.catalog.auto_download), cache=bool(config.catalog.cache))


@lru_cache(maxsize=8)
def _load_simulations_metadata(tag: str | None) -> Any:
    import sxs

    if tag:
        return sxs.load("simulations", tag=tag)
    return sxs.load("simulations")


@lru_cache(maxsize=8)
def _ensure_dataframe_loaded(tag: str | None) -> None:
    import sxs

    if tag:
        sxs.load("dataframe", tag=tag)
    else:
        sxs.load("dataframe")


def discover_available_levs(sxs_id: str, config: Any | None = None) -> list[int]:
    import sxs

    levs: set[int] = set()
    tag = getattr(getattr(config, "catalog", None), "tag", None)
    for metadata_tag in (str(tag) if tag else None, None):
        try:
            simulations = _load_simulations_metadata(metadata_tag)
            metadata = simulations.get(sxs_id)
            if metadata is not None:
                levs.update(_parse_lev_values(metadata.get("lev_numbers", [])))
                if levs:
                    return sorted(levs)
        except Exception:
            pass

    try:
        sim = sxs.load(sxs_id, auto_supersede=False, download=False)
        for attr in ("levs", "Lev", "resolutions", "available_levs", "available_resolutions"):
            value = getattr(sim, attr, None)
            if value is None and isinstance(sim, dict):
                value = sim.get(attr)
            if value is not None:
                levs.update(_parse_lev_values(value))
    except Exception:
        pass

    if levs:
        return sorted(levs)

    download = bool(getattr(getattr(config, "catalog", None), "auto_download", False))
    for lev in range(1, 11):
        try:
            sxs.load(f"{sxs_id}/Lev{lev}", auto_supersede=False, download=download)
        except Exception:
            continue
        levs.add(lev)
    return sorted(levs)


def _parse_lev_values(value: Any) -> list[int]:
    values = value.values() if isinstance(value, dict) else value
    parsed: list[int] = []
    if isinstance(values, (str, bytes)):
        values = [values]
    for item in values:
        text = str(item)
        if text.lower().startswith("lev"):
            text = text[3:]
        try:
            parsed.append(int(text))
        except ValueError:
            continue
    return parsed


def _parse_extrapolation_orders_from_files(value: Any, lev: int) -> list[str]:
    files = value.values() if isinstance(value, dict) else value
    if files is None:
        return []
    if isinstance(files, (str, bytes)):
        files = [files]
    lev_marker = f"Lev{int(lev)}"
    order_pattern = re.compile(r"Strain_([Nn]?\d+)")
    orders: list[str] = []
    for item in files:
        text = str(item)
        if lev_marker not in text:
            continue
        match = order_pattern.search(text)
        if not match:
            continue
        orders.append(match.group(1).upper() if match.group(1).upper().startswith("N") else f"N{match.group(1)}")
    return list(dict.fromkeys(orders))


def discover_available_extrapolation_orders(sxs_id: str, lev: int, config: Any | None = None) -> list[str] | None:
    tag = getattr(getattr(config, "catalog", None), "tag", None)
    for metadata_tag in (str(tag) if tag else None, None):
        try:
            metadata = _load_simulations_metadata(metadata_tag).get(sxs_id)
        except Exception:
            metadata = None
        if metadata is None:
            continue
        files = metadata.get("files")
        orders = _parse_extrapolation_orders_from_files(files, lev)
        if orders:
            return orders
    return None


def extrapolation_order_available(sxs_id: str, lev: int, extrapolation_order: str | int, config: Any) -> bool | None:
    tag = getattr(getattr(config, "catalog", None), "tag", None)
    metadata = None
    for metadata_tag in (str(tag) if tag else None, None):
        try:
            metadata = _load_simulations_metadata(metadata_tag).get(sxs_id)
        except Exception:
            metadata = None
        if metadata is not None:
            break
    if metadata is None:
        return None

    files = metadata.get("files", None)
    if not files:
        return None
    names = list(files.keys()) if isinstance(files, dict) else list(files)
    order_label = extrapolation_order_label(extrapolation_order)
    lev_prefix = f"Lev{int(lev)}:"
    strain_token = f"Strain_{order_label}"
    return any(str(name).startswith(lev_prefix) and strain_token in str(name) for name in names)


def _load_simulation(location: str, extrapolation_order: str, config: Any) -> Any:
    import sxs

    download = bool(config.catalog.auto_download)
    order_int = extrapolation_order_to_int(extrapolation_order)
    attempts = [
        {"download": download, "extrapolation_order": order_int, "auto_supersede": False},
        {"download": download, "extrapolation_order": str(extrapolation_order), "auto_supersede": False},
        {"download": download, "auto_supersede": False},
    ]
    errors = []
    for kwargs in attempts:
        try:
            return sxs.load(location, **kwargs)
        except TypeError as exc:
            errors.append(exc)
        except Exception as exc:
            errors.append(exc)
    raise RuntimeError(f"Could not load {location} at extrapolation order {extrapolation_order}: {errors[-1]}")


def _waveform_from_simulation(sim: Any, location: str, extrapolation_order: str, config: Any) -> Any:
    import sxs

    h = getattr(sim, config.waveform.data_type, None)
    if callable(h):
        h = h()
    if h is not None:
        return h

    order_int = extrapolation_order_to_int(extrapolation_order)
    path_candidates = [
        f"{location}/{config.waveform.data_type}",
        f"{location}/rhOverM",
        f"{location}/Strain_N{order_int}",
    ]
    errors = []
    for path in path_candidates:
        try:
            return sxs.load(
                path,
                download=bool(config.catalog.auto_download),
                extrapolation_order=order_int,
                auto_supersede=False,
            )
        except Exception as exc:
            errors.append(exc)
    raise RuntimeError(f"Could not find strain waveform for {location}: {errors[-1]}")


def _maybe_remove_memory(h: Any) -> Any:
    for method_name in ("remove_memory", "remove_strain_memory", "remove_avg"):
        method = getattr(h, method_name, None)
        if callable(method):
            result = method()
            return h if result is None else result
    return h


def _time_array(h: Any) -> np.ndarray:
    for attr in ("t", "time", "times"):
        value = getattr(h, attr, None)
        if value is not None:
            arr = value() if callable(value) else value
            return np.asarray(arr, dtype=float)
    arr = np.asarray(h)
    if arr.ndim >= 2:
        return np.asarray(arr[:, 0], dtype=float)
    raise ValueError("Could not extract waveform time array.")


def _mode_labels(h: Any) -> list[tuple[int, int]]:
    for attr in ("LM", "lm", "modes"):
        value = getattr(h, attr, None)
        if value is None:
            continue
        value = value() if callable(value) else value
        labels = []
        for item in value:
            try:
                ell, m = item
            except Exception:
                continue
            labels.append((int(ell), int(m)))
        if labels:
            return labels

    ell_min = int(getattr(h, "ell_min", getattr(h, "l_min", 2)))
    ell_max = int(getattr(h, "ell_max", getattr(h, "l_max", ell_min)))
    return [(ell, m) for ell in range(ell_min, ell_max + 1) for m in range(-ell, ell + 1)]


def _extract_mode(h: Any, ell: int, m: int, labels: list[tuple[int, int]]) -> np.ndarray | None:
    idx = None
    index_method = getattr(h, "index", None)
    if callable(index_method):
        try:
            idx = int(index_method(ell, m))
        except Exception:
            idx = None
    if idx is None:
        try:
            idx = labels.index((ell, m))
        except ValueError:
            return None

    data = getattr(h, "data", None)
    if data is None:
        data = np.asarray(h)
    else:
        data = np.asarray(data)

    candidates = []
    if data.ndim == 2:
        candidates.extend([lambda: data[:, idx], lambda: data[idx, :]])
    if hasattr(h, "__getitem__"):
        candidates.extend([lambda: h[:, idx], lambda: h[idx]])
    for candidate in candidates:
        try:
            mode = np.asarray(candidate(), dtype=complex)
            if mode.ndim == 1:
                return mode
        except Exception:
            continue
    return None


def _metadata_dict(sim: Any) -> dict[str, Any]:
    metadata = getattr(sim, "metadata", {})
    if callable(metadata):
        metadata = metadata()
    if isinstance(metadata, dict):
        return dict(metadata)
    return {}


def load_waveform(
    sxs_id: str,
    lev: int,
    extrapolation_order: str,
    config: Any,
) -> WaveformData:
    import sxs

    try:
        _ensure_dataframe_loaded(str(config.catalog.tag) if config.catalog.tag else None)
    except Exception:
        pass

    location = f"{sxs_id}/Lev{int(lev)}"
    available = extrapolation_order_available(sxs_id, lev, extrapolation_order, config)
    if available is False:
        raise FileNotFoundError(f"{sxs_id} Lev{lev} has no strain file for {extrapolation_order}.")
    sim = _load_simulation(location, extrapolation_order, config)
    h = _waveform_from_simulation(sim, location, extrapolation_order, config)
    if config.waveform.remove_memory:
        h = _maybe_remove_memory(h)

    time = _time_array(h)
    labels = _mode_labels(h)
    modes: dict[tuple[int, int], np.ndarray] = {}
    for ell, m in labels:
        mode = _extract_mode(h, ell, m, labels)
        if mode is not None and len(mode) == len(time):
            modes[(ell, m)] = np.asarray(mode, dtype=complex)

    if not modes:
        raise ValueError(f"No modes could be extracted from {location}.")
    ells = [ell for ell, _ in modes]
    return WaveformData(
        sxs_id=sxs_id,
        lev=int(lev),
        extrapolation_order=str(extrapolation_order),
        time=np.asarray(time, dtype=float),
        modes=modes,
        ell_min=min(ells),
        ell_max=max(ells),
        metadata=_metadata_dict(sim),
    )
