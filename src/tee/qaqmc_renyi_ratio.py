"""
Single-ratio driver for Path C of the QAQMC Renyi engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from src.tee.compose_tee import KPResult, RegionEntropySummary, compose_kp, summarize_region
from src.tee.ensemble_ladder import build_boundary_first_site_order
from src.engines.qaqmc_renyi import QAQMCRenyiRydberg


@dataclass
class RatioResult:
    ratio: float
    ratio_err: float
    visit_count_low: int
    visit_count_high: int
    block_visit_count_low: np.ndarray | None = None
    block_visit_count_high: np.ndarray | None = None


@dataclass
class RatioJobSpec:
    region_name: str
    step_index: int
    A_mask: np.ndarray
    next_site: int
    output_path: str | None = None


@dataclass
class RegionRunResult:
    region_name: str
    site_order: np.ndarray
    ratio_results: list[RatioResult]
    summary: RegionEntropySummary


@dataclass
class KPRunResult:
    region_runs: dict[str, RegionRunResult]
    kp_result: KPResult


@dataclass
class BlockingScanEntry:
    coarse_factor: int
    n_blocks: int
    s2: float
    s2_err: float
    ratios: np.ndarray
    ratio_errs: np.ndarray


def _as_mask(mask) -> np.ndarray:
    return np.ascontiguousarray(mask, dtype=np.uint8).reshape(-1)


def estimate_pair_ratio_error(visit_count_low: int, visit_count_high: int) -> float:
    low = int(visit_count_low)
    high = int(visit_count_high)
    if low < 0 or high < 0:
        raise ValueError("visit counts must be non-negative")
    if low == 0 and high == 0:
        return 0.0
    if low == 0 or high == 0:
        return float("inf")
    ratio = float(high / low)
    return float(ratio * np.sqrt((1.0 / low) + (1.0 / high)))


def _as_int64_counts(values, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    if np.any(arr < 0):
        raise ValueError(f"{name} must be non-negative")
    return arr


def estimate_pair_ratio_error_from_blocks(visit_counts_low, visit_counts_high) -> float:
    low_arr = _as_int64_counts(visit_counts_low, name="visit_counts_low")
    high_arr = _as_int64_counts(visit_counts_high, name="visit_counts_high")
    if low_arr.size != high_arr.size:
        raise ValueError("visit_counts_low and visit_counts_high must have the same length")
    if low_arr.size <= 1:
        total_low = int(np.sum(low_arr, dtype=np.int64))
        total_high = int(np.sum(high_arr, dtype=np.int64))
        return estimate_pair_ratio_error(total_low, total_high)

    total_low = int(np.sum(low_arr, dtype=np.int64))
    total_high = int(np.sum(high_arr, dtype=np.int64))
    leave_one_out_low = total_low - low_arr
    leave_one_out_high = total_high - high_arr
    if np.any(leave_one_out_low <= 0) or np.any(leave_one_out_high <= 0):
        return float("inf")

    loo_ratios = leave_one_out_high.astype(np.float64) / leave_one_out_low.astype(np.float64)
    loo_mean = float(np.mean(loo_ratios))
    prefactor = (low_arr.size - 1) / float(low_arr.size)
    return float(np.sqrt(prefactor * np.sum((loo_ratios - loo_mean) ** 2)))


def build_ratio_result(visit_counts_low, visit_counts_high) -> RatioResult:
    low_arr = _as_int64_counts(visit_counts_low, name="visit_counts_low")
    high_arr = _as_int64_counts(visit_counts_high, name="visit_counts_high")
    if low_arr.size != high_arr.size:
        raise ValueError("visit_counts_low and visit_counts_high must have the same length")

    total_low = int(np.sum(low_arr, dtype=np.int64))
    total_high = int(np.sum(high_arr, dtype=np.int64))
    ratio = float(total_high / total_low) if total_low > 0 else float("inf")
    ratio_err = estimate_pair_ratio_error_from_blocks(low_arr, high_arr)
    return RatioResult(
        ratio=ratio,
        ratio_err=ratio_err,
        visit_count_low=total_low,
        visit_count_high=total_high,
        block_visit_count_low=low_arr.copy(),
        block_visit_count_high=high_arr.copy(),
    )


def combine_visit_counts(visit_counts_low, visit_counts_high) -> RatioResult:
    return build_ratio_result(visit_counts_low, visit_counts_high)


def combine_ratio_results(results) -> RatioResult:
    result_list = list(results)
    if not result_list:
        raise ValueError("results must be non-empty")
    if all(item.block_visit_count_low is not None and item.block_visit_count_high is not None for item in result_list):
        lows = np.concatenate(
            [np.asarray(item.block_visit_count_low, dtype=np.int64).reshape(-1) for item in result_list]
        )
        highs = np.concatenate(
            [np.asarray(item.block_visit_count_high, dtype=np.int64).reshape(-1) for item in result_list]
        )
        return build_ratio_result(lows, highs)
    lows = [int(item.visit_count_low) for item in result_list]
    highs = [int(item.visit_count_high) for item in result_list]
    return build_ratio_result(lows, highs)


def estimate_s2_error_from_ratio_blocks(ratio_results) -> float:
    result_list = list(ratio_results)
    if not result_list:
        return 0.0
    if not all(item.block_visit_count_low is not None and item.block_visit_count_high is not None for item in result_list):
        return float(np.sqrt(np.sum([(item.ratio_err / item.ratio) ** 2 for item in result_list if item.ratio > 0.0])))

    n_blocks = int(np.asarray(result_list[0].block_visit_count_low).size)
    if n_blocks <= 1:
        return float(np.sqrt(np.sum([(item.ratio_err / item.ratio) ** 2 for item in result_list if item.ratio > 0.0])))
    for item in result_list[1:]:
        if int(np.asarray(item.block_visit_count_low).size) != n_blocks:
            return float(np.sqrt(np.sum([(entry.ratio_err / entry.ratio) ** 2 for entry in result_list if entry.ratio > 0.0])))

    leave_one_out_s2 = []
    for block_index in range(n_blocks):
        ratios = []
        for item in result_list:
            low_blocks = np.asarray(item.block_visit_count_low, dtype=np.int64).reshape(-1)
            high_blocks = np.asarray(item.block_visit_count_high, dtype=np.int64).reshape(-1)
            low = int(np.sum(low_blocks, dtype=np.int64) - low_blocks[block_index])
            high = int(np.sum(high_blocks, dtype=np.int64) - high_blocks[block_index])
            if low <= 0 or high <= 0:
                return float("inf")
            ratios.append(float(high / low))
        leave_one_out_s2.append(float(-np.sum(np.log(np.asarray(ratios, dtype=np.float64)))))

    loo = np.asarray(leave_one_out_s2, dtype=np.float64)
    loo_mean = float(np.mean(loo))
    prefactor = (n_blocks - 1) / float(n_blocks)
    return float(np.sqrt(prefactor * np.sum((loo - loo_mean) ** 2)))


def summarize_region_from_ratio_results(region_name: str, site_order, ratio_results) -> RegionEntropySummary:
    ratios = np.array([item.ratio for item in ratio_results], dtype=np.float64)
    ratio_errs = np.array([item.ratio_err for item in ratio_results], dtype=np.float64)
    summary = summarize_region(region_name, site_order, ratios, ratio_errs)
    summary.S_2_err = estimate_s2_error_from_ratio_blocks(ratio_results)
    return summary


def coarsen_ratio_result_blocks(result: RatioResult, coarse_factor: int) -> RatioResult:
    factor = int(coarse_factor)
    if factor <= 0:
        raise ValueError("coarse_factor must be positive")
    if result.block_visit_count_low is None or result.block_visit_count_high is None:
        raise ValueError("result must carry block visit counts")

    low = np.asarray(result.block_visit_count_low, dtype=np.int64).reshape(-1)
    high = np.asarray(result.block_visit_count_high, dtype=np.int64).reshape(-1)
    n_groups = low.size // factor
    if n_groups <= 0:
        raise ValueError("coarse_factor is larger than the number of blocks")
    low_trim = low[:n_groups * factor].reshape(n_groups, factor).sum(axis=1)
    high_trim = high[:n_groups * factor].reshape(n_groups, factor).sum(axis=1)
    return build_ratio_result(low_trim, high_trim)


def scan_region_blocking(region_run: RegionRunResult, coarse_factors=None) -> list[BlockingScanEntry]:
    results = list(region_run.ratio_results)
    if not results:
        return []
    if not all(item.block_visit_count_low is not None and item.block_visit_count_high is not None for item in results):
        raise ValueError("all ratio_results must carry block visit counts")

    n_base_blocks = int(np.asarray(results[0].block_visit_count_low, dtype=np.int64).size)
    for item in results[1:]:
        if int(np.asarray(item.block_visit_count_low, dtype=np.int64).size) != n_base_blocks:
            raise ValueError("all ratio_results must have the same number of blocks")

    if coarse_factors is None:
        factors = [1]
        candidate = 2
        while candidate <= n_base_blocks:
            factors.append(candidate)
            candidate *= 2
    else:
        factors = sorted({int(value) for value in coarse_factors if int(value) > 0})
        if not factors:
            raise ValueError("coarse_factors must contain at least one positive integer")

    entries: list[BlockingScanEntry] = []
    for factor in factors:
        n_groups = n_base_blocks // factor
        if n_groups < 2:
            continue
        coarsened_results = [coarsen_ratio_result_blocks(item, factor) for item in results]
        summary = summarize_region_from_ratio_results(region_run.region_name, region_run.site_order, coarsened_results)
        entries.append(
            BlockingScanEntry(
                coarse_factor=factor,
                n_blocks=n_groups,
                s2=float(summary.S_2),
                s2_err=float(summary.S_2_err),
                ratios=np.asarray([item.ratio for item in coarsened_results], dtype=np.float64),
                ratio_errs=np.asarray([item.ratio_err for item in coarsened_results], dtype=np.float64),
            )
        )
    return entries


def combine_region_runs(region_runs) -> RegionRunResult:
    run_list = list(region_runs)
    if not run_list:
        raise ValueError("region_runs must be non-empty")

    first = run_list[0]
    n_steps = len(first.ratio_results)
    site_order = np.asarray(first.site_order, dtype=np.int32)

    for item in run_list[1:]:
        if item.region_name != first.region_name:
            raise ValueError("all region runs must have the same region_name")
        if len(item.ratio_results) != n_steps:
            raise ValueError("all region runs must have the same number of steps")
        if not np.array_equal(np.asarray(item.site_order, dtype=np.int32), site_order):
            raise ValueError("all region runs must have the same site_order")

    combined_ratio_results = [
        combine_ratio_results(run.ratio_results[step_index] for run in run_list)
        for step_index in range(n_steps)
    ]
    ratios = np.array([item.ratio for item in combined_ratio_results], dtype=np.float64)
    ratio_errs = np.array([item.ratio_err for item in combined_ratio_results], dtype=np.float64)
    summary = summarize_region_from_ratio_results(first.region_name, site_order, combined_ratio_results)
    return RegionRunResult(
        region_name=str(first.region_name),
        site_order=site_order.copy(),
        ratio_results=combined_ratio_results,
        summary=summary,
    )


# Backward-compatible aliases for older callers/tests while the terminology
# shifts from midpoint indicators to topology visit counts.
estimate_binomial_error = estimate_pair_ratio_error
combine_indicator_counts = combine_visit_counts


def build_ratio_job_specs(region_name: str, region_mask, *, bond_sites=None, site_order=None,
                          output_dir=None, filename_template: str = "{region}_step{step:03d}.h5"
                          ) -> list[RatioJobSpec]:
    region_mask_arr = _as_mask(region_mask)
    if site_order is None:
        if bond_sites is None:
            raise ValueError("bond_sites are required when site_order is not provided")
        site_order_arr = np.asarray(
            build_boundary_first_site_order(region_mask_arr, bond_sites),
            dtype=np.int32,
        )
    else:
        site_order_arr = np.asarray(site_order, dtype=np.int32).reshape(-1)

    jobs: list[RatioJobSpec] = []
    current_mask = np.zeros_like(region_mask_arr, dtype=np.uint8)
    output_root = None if output_dir is None else Path(output_dir)
    for step_index, next_site in enumerate(site_order_arr):
        output_path = None
        if output_root is not None:
            filename = filename_template.format(region=region_name, step=int(step_index), site=int(next_site))
            output_path = str(output_root / filename)
        jobs.append(
            RatioJobSpec(
                region_name=str(region_name),
                step_index=int(step_index),
                A_mask=current_mask.copy(),
                next_site=int(next_site),
                output_path=output_path,
            )
        )
        current_mask[int(next_site)] = 1
    return jobs


class RatioRunner:
    def __init__(self, engine: QAQMCRenyiRydberg | None = None, **engine_kwargs):
        if engine is None:
            engine = QAQMCRenyiRydberg(**engine_kwargs)
        self.engine = engine

    def run_single_ratio(self, A_mask, next_site: int, n_therm: int, n_measure: int,
                         measure_stride: int = 1, block_size: int | None = None) -> RatioResult:
        mask = _as_mask(A_mask)
        if int(next_site) < 0 or int(next_site) >= mask.size:
            raise ValueError("next_site out of range")
        if int(mask[int(next_site)]) != 0:
            raise ValueError("next_site must be outside the current A_mask")
        stride = int(measure_stride)
        if stride <= 0:
            raise ValueError("measure_stride must be positive")
        block_len = None if block_size is None else int(block_size)
        if block_len is not None and block_len <= 0:
            raise ValueError("block_size must be positive")
        next_mask = mask.copy()
        next_mask[int(next_site)] = 1
        self.engine.set_topology_pair(mask, next_mask, int(next_site))
        if n_therm > 0:
            self.engine.run_steps(int(n_therm))
        if n_measure <= 0:
            return build_ratio_result([0], [0])

        if block_len is None:
            self.engine.reset_visit_counts()
            self.engine.run_steps(int(n_measure) * stride)
            visit_count_low, visit_count_high = self.engine.get_visit_counts().tolist()
            return build_ratio_result([visit_count_low], [visit_count_high])

        block_lows: list[int] = []
        block_highs: list[int] = []
        remaining = int(n_measure)
        while remaining > 0:
            current_block = min(block_len, remaining)
            self.engine.reset_visit_counts()
            self.engine.run_steps(current_block * stride)
            visit_count_low, visit_count_high = self.engine.get_visit_counts().tolist()
            block_lows.append(int(visit_count_low))
            block_highs.append(int(visit_count_high))
            remaining -= current_block
        return build_ratio_result(block_lows, block_highs)


class RegionRatioRunner:
    def __init__(self, engine_factory=None, **engine_kwargs):
        self.engine_factory = engine_factory
        self.engine_kwargs = dict(engine_kwargs)

    def _make_engine(self, step_index: int):
        if self.engine_factory is not None:
            return self.engine_factory(step_index)
        kwargs = dict(self.engine_kwargs)
        if "seed" in kwargs:
            kwargs["seed"] = int(kwargs["seed"]) + 104729 * step_index
        return QAQMCRenyiRydberg(**kwargs)

    def run_region(self, region_name: str, region_mask, *, bond_sites=None, site_order=None,
                   n_therm: int, n_measure: int, measure_stride: int = 1,
                   block_size: int | None = None) -> RegionRunResult:
        region_mask_arr = _as_mask(region_mask)
        jobs = build_ratio_job_specs(
            region_name,
            region_mask_arr,
            bond_sites=bond_sites,
            site_order=site_order,
        )
        site_order_arr = np.asarray([job.next_site for job in jobs], dtype=np.int32)

        ratio_results: list[RatioResult] = []
        for job in jobs:
            engine = self._make_engine(job.step_index)
            runner = RatioRunner(engine=engine)
            result = runner.run_single_ratio(
                job.A_mask,
                next_site=int(job.next_site),
                n_therm=int(n_therm),
                n_measure=int(n_measure),
                measure_stride=int(measure_stride),
                block_size=block_size,
            )
            ratio_results.append(result)
        summary = summarize_region_from_ratio_results(region_name, site_order_arr, ratio_results)
        return RegionRunResult(
            region_name=str(region_name),
            site_order=site_order_arr,
            ratio_results=ratio_results,
            summary=summary,
        )


class KPRatioRunner:
    def __init__(self, region_runner: RegionRatioRunner | None = None, **engine_kwargs):
        if region_runner is None:
            region_runner = RegionRatioRunner(**engine_kwargs)
        self.region_runner = region_runner

    def run_kp(self, region_masks: dict[str, np.ndarray], *, bond_sites=None, site_orders=None,
               n_therm: int, n_measure: int, measure_stride: int = 1,
               block_size: int | None = None) -> KPRunResult:
        region_runs: dict[str, RegionRunResult] = {}
        summaries: dict[str, RegionEntropySummary] = {}
        for region_name, region_mask in region_masks.items():
            site_order = None if site_orders is None else site_orders.get(region_name)
            region_result = self.region_runner.run_region(
                region_name,
                region_mask,
                bond_sites=bond_sites,
                site_order=site_order,
                n_therm=n_therm,
                n_measure=n_measure,
                measure_stride=measure_stride,
                block_size=block_size,
            )
            region_runs[region_name] = region_result
            summaries[region_name] = region_result.summary

        gamma, gamma_err = compose_kp(summaries)
        return KPRunResult(
            region_runs=region_runs,
            kp_result=KPResult(region_summaries=summaries, gamma=gamma, gamma_err=gamma_err),
        )


def save_ratio_result_hdf5(filepath, *, region_name: str, step_index: int,
                           A_mask, next_site: int, result: RatioResult,
                           params: dict | None = None,
                           per_rank_visit_count_low=None,
                           per_rank_visit_count_high=None,
                           pos=None,
                           delta_schedule=None):
    file_path = Path(filepath)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    mask_arr = _as_mask(A_mask)

    with h5py.File(file_path, "w") as h5:
        params_group = h5.create_group("params")
        params_group.attrs["region_name"] = str(region_name)
        params_group.attrs["step_index"] = int(step_index)
        if params:
            for key, value in params.items():
                params_group.attrs[str(key)] = value

        ratio_group = h5.create_group("ratio")
        next_mask = mask_arr.copy()
        next_mask[int(next_site)] = 1
        ratio_group.create_dataset("A_k_mask", data=mask_arr, dtype="uint8")
        ratio_group.create_dataset("A_kp1_mask", data=next_mask, dtype="uint8")
        ratio_group.create_dataset("diff_site", data=int(next_site))
        ratio_group.create_dataset("next_site", data=int(next_site))
        ratio_group.create_dataset("visit_count_low", data=int(result.visit_count_low))
        ratio_group.create_dataset("visit_count_high", data=int(result.visit_count_high))
        ratio_group.create_dataset("ratio", data=float(result.ratio))
        ratio_group.create_dataset("ratio_err", data=float(result.ratio_err))
        if result.block_visit_count_low is not None:
            ratio_group.create_dataset(
                "block_visit_count_low",
                data=np.asarray(result.block_visit_count_low, dtype=np.int64).reshape(-1),
                dtype="int64",
            )
        if result.block_visit_count_high is not None:
            ratio_group.create_dataset(
                "block_visit_count_high",
                data=np.asarray(result.block_visit_count_high, dtype=np.int64).reshape(-1),
                dtype="int64",
            )
        if per_rank_visit_count_low is not None:
            ratio_group.create_dataset(
                "visit_count_low_per_rank",
                data=np.asarray(per_rank_visit_count_low, dtype=np.int64).reshape(-1),
                dtype="int64",
            )
        if per_rank_visit_count_high is not None:
            ratio_group.create_dataset(
                "visit_count_high_per_rank",
                data=np.asarray(per_rank_visit_count_high, dtype=np.int64).reshape(-1),
                dtype="int64",
            )

        if pos is not None:
            geometry_group = h5.create_group("geometry")
            geometry_group.create_dataset("pos", data=np.asarray(pos, dtype=np.float64))
        if delta_schedule is not None:
            schedule_group = h5.create_group("schedule")
            schedule_group.create_dataset("delta_schedule", data=np.asarray(delta_schedule, dtype=np.float64))


def load_ratio_result_hdf5(filepath) -> dict:
    with h5py.File(filepath, "r") as h5:
        params = dict(h5["params"].attrs)
        ratio_group = h5["ratio"]
        loaded = {
            "params": params,
            "A_mask": np.asarray(ratio_group["A_k_mask"], dtype=np.uint8)
            if "A_k_mask" in ratio_group else np.asarray(ratio_group["A_mask"], dtype=np.uint8),
            "A_kp1_mask": np.asarray(ratio_group["A_kp1_mask"], dtype=np.uint8)
            if "A_kp1_mask" in ratio_group else None,
            "next_site": int(np.asarray(ratio_group["next_site"])),
            "result": RatioResult(
                ratio=float(np.asarray(ratio_group["ratio"])),
                ratio_err=float(np.asarray(ratio_group["ratio_err"])),
                visit_count_low=int(np.asarray(ratio_group["visit_count_low"]))
                if "visit_count_low" in ratio_group else int(np.asarray(ratio_group["indicator_count"])) - int(np.asarray(ratio_group["indicator_sum"])),
                visit_count_high=int(np.asarray(ratio_group["visit_count_high"]))
                if "visit_count_high" in ratio_group else int(np.asarray(ratio_group["indicator_sum"])),
                block_visit_count_low=np.asarray(ratio_group["block_visit_count_low"], dtype=np.int64)
                if "block_visit_count_low" in ratio_group else None,
                block_visit_count_high=np.asarray(ratio_group["block_visit_count_high"], dtype=np.int64)
                if "block_visit_count_high" in ratio_group else None,
            ),
        }
        if "diff_site" in ratio_group:
            loaded["diff_site"] = int(np.asarray(ratio_group["diff_site"]))
        if "visit_count_low_per_rank" in ratio_group:
            loaded["visit_count_low_per_rank"] = np.asarray(ratio_group["visit_count_low_per_rank"], dtype=np.int64)
        if "visit_count_high_per_rank" in ratio_group:
            loaded["visit_count_high_per_rank"] = np.asarray(ratio_group["visit_count_high_per_rank"], dtype=np.int64)
        if "indicator_sum_per_rank" in ratio_group:
            loaded["visit_count_high_per_rank"] = np.asarray(ratio_group["indicator_sum_per_rank"], dtype=np.int64)
        if "indicator_count_per_rank" in ratio_group and "visit_count_high_per_rank" in loaded:
            loaded["visit_count_low_per_rank"] = (
                np.asarray(ratio_group["indicator_count_per_rank"], dtype=np.int64)
                - loaded["visit_count_high_per_rank"]
            )
        if "geometry" in h5 and "pos" in h5["geometry"]:
            loaded["pos"] = np.asarray(h5["geometry"]["pos"], dtype=np.float64)
        if "schedule" in h5 and "delta_schedule" in h5["schedule"]:
            loaded["delta_schedule"] = np.asarray(h5["schedule"]["delta_schedule"], dtype=np.float64)
        return loaded


def save_ratio_manifest_hdf5(filepath, jobs: list[RatioJobSpec], *, params: dict | None = None):
    file_path = Path(filepath)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    n_jobs = len(jobs)
    mask_len = 0 if n_jobs == 0 else int(jobs[0].A_mask.size)
    with h5py.File(file_path, "w") as h5:
        manifest_group = h5.create_group("manifest")
        if params:
            for key, value in params.items():
                manifest_group.attrs[str(key)] = value
        utf8 = h5py.string_dtype(encoding="utf-8")
        manifest_group.create_dataset(
            "region_name",
            data=np.asarray([job.region_name for job in jobs], dtype=object),
            dtype=utf8,
        )
        manifest_group.create_dataset(
            "step_index",
            data=np.asarray([job.step_index for job in jobs], dtype=np.int32),
            dtype="int32",
        )
        manifest_group.create_dataset(
            "next_site",
            data=np.asarray([job.next_site for job in jobs], dtype=np.int32),
            dtype="int32",
        )
        manifest_group.create_dataset(
            "A_mask",
            data=np.asarray(
                [job.A_mask for job in jobs],
                dtype=np.uint8,
            ).reshape(n_jobs, mask_len),
            dtype="uint8",
        )
        manifest_group.create_dataset(
            "output_path",
            data=np.asarray(
                ["" if job.output_path is None else job.output_path for job in jobs],
                dtype=object,
            ),
            dtype=utf8,
        )


def load_ratio_manifest_hdf5(filepath) -> dict:
    with h5py.File(filepath, "r") as h5:
        manifest_group = h5["manifest"]
        region_names = [item.decode() if isinstance(item, bytes) else str(item)
                        for item in np.asarray(manifest_group["region_name"])]
        output_paths = [item.decode() if isinstance(item, bytes) else str(item)
                        for item in np.asarray(manifest_group["output_path"])]
        jobs = [
            RatioJobSpec(
                region_name=region_names[idx],
                step_index=int(np.asarray(manifest_group["step_index"])[idx]),
                A_mask=np.asarray(manifest_group["A_mask"][idx], dtype=np.uint8),
                next_site=int(np.asarray(manifest_group["next_site"])[idx]),
                output_path=None if output_paths[idx] == "" else output_paths[idx],
            )
            for idx in range(len(region_names))
        ]
        return {
            "params": dict(manifest_group.attrs),
            "jobs": jobs,
        }
