"""
High-level KP workflows that connect kagome KP geometry to the existing
ratio-estimator and expanded-ensemble engines.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.kp_geometry import (
    KPRegionSpec,
    RegionLadderSpec,
    attach_kp_site_orders,
    build_kp_ladders,
    build_kp_region_masks,
)
from src.qaqmc_renyi_ratio import KPRatioRunner
from src.reweighting import ExpandedProductionResult, ReweightingDriver


@dataclass(frozen=True)
class KagomeKPRatioResult:
    spec: KPRegionSpec
    result: object


@dataclass(frozen=True)
class KagomeExpandedRegionResult:
    spec: KPRegionSpec
    ladder: RegionLadderSpec
    autotune: object | None
    production: ExpandedProductionResult


class KagomeKPRatioWorkflow:
    def __init__(self, kp_runner: KPRatioRunner | None = None, **engine_kwargs):
        if kp_runner is None:
            kp_runner = KPRatioRunner(**engine_kwargs)
        self.kp_runner = kp_runner

    def run(
        self,
        *,
        nx: int,
        ny: int,
        m: int,
        bond_sites,
        n_therm: int,
        n_measure: int,
        a: float = 1.0,
        preferred_center_label: str = "C35",
        measure_stride: int = 1,
        block_size: int | None = None,
    ) -> KagomeKPRatioResult:
        spec = build_kp_region_masks(
            nx, ny, m=m, a=a, preferred_center_label=preferred_center_label
        )
        spec = attach_kp_site_orders(spec, bond_sites)
        result = self.kp_runner.run_kp(
            spec.region_masks,
            bond_sites=bond_sites,
            site_orders=spec.site_orders,
            n_therm=n_therm,
            n_measure=n_measure,
            measure_stride=measure_stride,
            block_size=block_size,
        )
        return KagomeKPRatioResult(spec=spec, result=result)


class KagomeKPExpandedWorkflow:
    def __init__(self, driver_factory=None, **engine_kwargs):
        self.driver_factory = driver_factory
        self.engine_kwargs = dict(engine_kwargs)

    def _make_driver(self, region_name: str) -> ReweightingDriver:
        if self.driver_factory is not None:
            return self.driver_factory(region_name)
        return ReweightingDriver(**self.engine_kwargs)

    def run_region(
        self,
        region_name: str,
        *,
        nx: int,
        ny: int,
        m: int,
        bond_sites,
        a: float = 1.0,
        preferred_center_label: str = "C35",
        autotune_steps_per_iter: int | None = None,
        autotune_max_iters: int = 10,
        autotune_tol: float = 0.3,
        autotune_method: str = "transition_matrix",
        autotune_damping: float = 1.0,
        n_steps: int | None = None,
        block_size: int | None = None,
        target_s2_err: float | None = None,
        batch_steps: int | None = None,
        max_steps: int | None = None,
        min_steps: int = 0,
        estimator: str = "collection",
    ) -> KagomeExpandedRegionResult:
        spec = build_kp_region_masks(
            nx, ny, m=m, a=a, preferred_center_label=preferred_center_label
        )
        ladders = build_kp_ladders(spec, bond_sites=bond_sites)
        ladder = ladders[str(region_name)]

        driver = self._make_driver(str(region_name))
        driver.set_ensemble_ladder(ladder.masks, ladder.neighbors, initial_ensemble=0)

        autotune = None
        if autotune_steps_per_iter is not None and int(autotune_steps_per_iter) > 0:
            autotune = driver.auto_tune(
                n_steps_per_iter=int(autotune_steps_per_iter),
                max_iters=int(autotune_max_iters),
                tol=float(autotune_tol),
                method=str(autotune_method),
                damping=float(autotune_damping),
            )

        if target_s2_err is not None:
            if batch_steps is None or max_steps is None or block_size is None:
                raise ValueError(
                    "target_s2_err mode requires batch_steps, max_steps, and block_size"
                )
            production = driver.run_until_target_error(
                target_ensemble=ladder.target_ensemble,
                target_s2_err=float(target_s2_err),
                batch_steps=int(batch_steps),
                block_size=int(block_size),
                max_steps=int(max_steps),
                min_steps=int(min_steps),
                estimator=str(estimator),
            )
        else:
            if n_steps is None:
                raise ValueError("either n_steps or target_s2_err must be provided")
            production = driver.run_production(
                n_steps=int(n_steps),
                block_size=block_size,
            )

        return KagomeExpandedRegionResult(
            spec=spec,
            ladder=ladder,
            autotune=autotune,
            production=production,
        )
