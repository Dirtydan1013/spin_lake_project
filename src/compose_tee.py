"""
Post-processing helpers for Path C ratio-estimator outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


KP_COEFFICIENTS = {
    "A": 1.0,
    "B": 1.0,
    "C": 1.0,
    "AB": -1.0,
    "BC": -1.0,
    "CA": -1.0,
    "ABC": 1.0,
}


@dataclass
class KPResult:
    region_summaries: dict[str, "RegionEntropySummary"]
    gamma: float
    gamma_err: float


@dataclass
class RegionEntropySummary:
    region_name: str
    sites_ordered: np.ndarray
    ratios: np.ndarray
    ratio_errs: np.ndarray
    log_ratios: np.ndarray
    log_ratio_errs: np.ndarray
    S_2: float
    S_2_err: float


def _as_1d_float(arr) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def _as_1d_int(arr) -> np.ndarray:
    return np.asarray(arr, dtype=np.int32).reshape(-1)


def summarize_region(region_name: str, sites_ordered, ratios, ratio_errs=None) -> RegionEntropySummary:
    ratios_arr = _as_1d_float(ratios)
    sites_arr = _as_1d_int(sites_ordered)
    if ratios_arr.size != sites_arr.size:
        raise ValueError("sites_ordered and ratios must have the same length")
    if np.any(ratios_arr <= 0.0):
        raise ValueError("all ratios must be strictly positive to form log-ratios")

    if ratio_errs is None:
        ratio_errs_arr = np.zeros_like(ratios_arr)
    else:
        ratio_errs_arr = _as_1d_float(ratio_errs)
        if ratio_errs_arr.size != ratios_arr.size:
            raise ValueError("ratio_errs must match ratios length")
        if np.any(ratio_errs_arr < 0.0):
            raise ValueError("ratio_errs must be non-negative")

    log_ratios = np.log(ratios_arr)
    log_ratio_errs = np.divide(
        ratio_errs_arr,
        ratios_arr,
        out=np.zeros_like(ratio_errs_arr),
        where=ratios_arr > 0.0,
    )
    s2 = float(-np.sum(log_ratios))
    s2_err = float(np.sqrt(np.sum(log_ratio_errs ** 2)))
    return RegionEntropySummary(
        region_name=str(region_name),
        sites_ordered=sites_arr,
        ratios=ratios_arr,
        ratio_errs=ratio_errs_arr,
        log_ratios=log_ratios,
        log_ratio_errs=log_ratio_errs,
        S_2=s2,
        S_2_err=s2_err,
    )


def compose_kp(region_summaries: dict[str, RegionEntropySummary | float],
               region_errors: dict[str, float] | None = None) -> tuple[float, float]:
    gamma = 0.0
    gamma_var = 0.0
    for region, coeff in KP_COEFFICIENTS.items():
        if region not in region_summaries:
            raise KeyError(f"missing KP region: {region}")
        item = region_summaries[region]
        if isinstance(item, RegionEntropySummary):
            value = item.S_2
            err = item.S_2_err
        else:
            value = float(item)
            err = 0.0 if region_errors is None else float(region_errors.get(region, 0.0))
        gamma += coeff * value
        gamma_var += err * err
    return float(gamma), float(np.sqrt(gamma_var))


def aggregate_ratios(hdf5_paths) -> dict[str, RegionEntropySummary]:
    """Aggregate per-ratio HDF5 outputs into per-region entropy summaries.

    Expected minimal schema per file:
    - /params attrs: region_name (str), step_index (int)
    - /ratio/next_site, /ratio/ratio, /ratio/ratio_err
    """
    grouped: dict[str, list[tuple[int, int, float, float]]] = {}
    for path in hdf5_paths:
        file_path = Path(path)
        with h5py.File(file_path, "r") as h5:
            params = h5["params"].attrs
            ratio_group = h5["ratio"]
            region_name = params["region_name"]
            if isinstance(region_name, bytes):
                region_name = region_name.decode()
            step_index = int(params["step_index"])
            next_site = int(np.asarray(ratio_group["next_site"]))
            ratio = float(np.asarray(ratio_group["ratio"]))
            ratio_err = float(np.asarray(ratio_group["ratio_err"]))
        grouped.setdefault(str(region_name), []).append((step_index, next_site, ratio, ratio_err))

    summaries: dict[str, RegionEntropySummary] = {}
    for region_name, rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda item: item[0])
        sites = [row[1] for row in rows_sorted]
        ratios = [row[2] for row in rows_sorted]
        ratio_errs = [row[3] for row in rows_sorted]
        summaries[region_name] = summarize_region(region_name, sites, ratios, ratio_errs)
    return summaries


def save_region_summary_hdf5(filepath, summary: RegionEntropySummary):
    file_path = Path(filepath)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(file_path, "w") as h5:
        region_group = h5.create_group("region")
        region_group.attrs["region_name"] = summary.region_name
        region_group.create_dataset("sites_ordered", data=summary.sites_ordered, dtype="int32")
        region_group.create_dataset("ratios", data=summary.ratios, dtype="float64")
        region_group.create_dataset("ratio_errs", data=summary.ratio_errs, dtype="float64")
        region_group.create_dataset("log_ratios", data=summary.log_ratios, dtype="float64")
        region_group.create_dataset("log_ratio_errs", data=summary.log_ratio_errs, dtype="float64")
        region_group.create_dataset("S_2", data=float(summary.S_2))
        region_group.create_dataset("S_2_err", data=float(summary.S_2_err))


def load_region_summary_hdf5(filepath) -> RegionEntropySummary:
    with h5py.File(filepath, "r") as h5:
        region_group = h5["region"]
        region_name = region_group.attrs["region_name"]
        if isinstance(region_name, bytes):
            region_name = region_name.decode()
        return RegionEntropySummary(
            region_name=str(region_name),
            sites_ordered=np.asarray(region_group["sites_ordered"], dtype=np.int32),
            ratios=np.asarray(region_group["ratios"], dtype=np.float64),
            ratio_errs=np.asarray(region_group["ratio_errs"], dtype=np.float64),
            log_ratios=np.asarray(region_group["log_ratios"], dtype=np.float64),
            log_ratio_errs=np.asarray(region_group["log_ratio_errs"], dtype=np.float64),
            S_2=float(np.asarray(region_group["S_2"])),
            S_2_err=float(np.asarray(region_group["S_2_err"])),
        )


def save_kp_result_hdf5(filepath, kp_result: KPResult):
    file_path = Path(filepath)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(file_path, "w") as h5:
        regions_group = h5.create_group("regions")
        for region_name, summary in kp_result.region_summaries.items():
            region_group = regions_group.create_group(region_name)
            region_group.create_dataset("sites_ordered", data=summary.sites_ordered, dtype="int32")
            region_group.create_dataset("ratios", data=summary.ratios, dtype="float64")
            region_group.create_dataset("ratio_errs", data=summary.ratio_errs, dtype="float64")
            region_group.create_dataset("log_ratios", data=summary.log_ratios, dtype="float64")
            region_group.create_dataset("log_ratio_errs", data=summary.log_ratio_errs, dtype="float64")
            region_group.create_dataset("S_2", data=float(summary.S_2))
            region_group.create_dataset("S_2_err", data=float(summary.S_2_err))
        result_group = h5.create_group("result")
        result_group.create_dataset("gamma", data=float(kp_result.gamma))
        result_group.create_dataset("gamma_err", data=float(kp_result.gamma_err))


def load_kp_result_hdf5(filepath) -> KPResult:
    with h5py.File(filepath, "r") as h5:
        summaries: dict[str, RegionEntropySummary] = {}
        for region_name in h5["regions"]:
            region_group = h5[f"regions/{region_name}"]
            summaries[region_name] = RegionEntropySummary(
                region_name=region_name,
                sites_ordered=np.asarray(region_group["sites_ordered"], dtype=np.int32),
                ratios=np.asarray(region_group["ratios"], dtype=np.float64),
                ratio_errs=np.asarray(region_group["ratio_errs"], dtype=np.float64),
                log_ratios=np.asarray(region_group["log_ratios"], dtype=np.float64),
                log_ratio_errs=np.asarray(region_group["log_ratio_errs"], dtype=np.float64),
                S_2=float(np.asarray(region_group["S_2"])),
                S_2_err=float(np.asarray(region_group["S_2_err"])),
            )
        result_group = h5["result"]
        return KPResult(
            region_summaries=summaries,
            gamma=float(np.asarray(result_group["gamma"])),
            gamma_err=float(np.asarray(result_group["gamma_err"])),
        )
