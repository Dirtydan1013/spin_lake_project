import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tee.compose_tee import aggregate_ratios
from src.tee.ensemble_ladder import build_boundary_first_site_order
from src.rydberg.hamiltonian import build_rydberg_vij
from src.rydberg.lattices import generate_1d_chain
from src.tee.qaqmc_renyi_ratio import RegionRatioRunner, save_ratio_result_hdf5


def _test_region_runner_and_aggregation_with_real_engine():
    pos = generate_1d_chain(6)
    _, _, _, _, bond_sites, _ = build_rydberg_vij(
        N=6, Omega=1.0, Rb=1.2, pos=pos, verbose=False, neighbor_cutoff=1
    )
    region_mask = np.zeros(6, dtype=np.uint8)
    region_mask[[1, 2, 3]] = 1
    site_order = build_boundary_first_site_order(region_mask, bond_sites)

    runner = RegionRatioRunner(
        N=6,
        M=10,
        Omega=1.0,
        Rb=1.2,
        delta_min=0.0,
        delta_max=1.0,
        pos=pos,
        seed=23,
        neighbor_cutoff=1,
    )
    region_result = runner.run_region(
        "A",
        region_mask,
        site_order=site_order,
        n_therm=3,
        n_measure=10,
    )

    assert len(region_result.ratio_results) == len(site_order)
    assert region_result.site_order.tolist() == site_order
    assert np.isfinite(region_result.summary.S_2)
    assert region_result.summary.S_2_err >= 0.0

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        current_mask = np.zeros_like(region_mask)
        for step_index, (site, ratio_result) in enumerate(zip(region_result.site_order, region_result.ratio_results)):
            path = os.path.join(tmpdir, f"ratio_{step_index}.h5")
            save_ratio_result_hdf5(
                path,
                region_name="A",
                step_index=step_index,
                A_mask=current_mask,
                next_site=int(site),
                result=ratio_result,
            )
            current_mask[int(site)] = 1
            paths.append(path)

        summaries = aggregate_ratios(paths)
        assert np.isclose(summaries["A"].S_2, region_result.summary.S_2)


def main():
    _test_region_runner_and_aggregation_with_real_engine()
    print("QAQMC Renyi region runner integration checks passed")


if __name__ == "__main__":
    main()
