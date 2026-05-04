import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.engines.qaqmc_renyi import QAQMCRenyiRydberg


def _make_manual_strings(engine):
    op_types_0 = np.ones(engine.M_total, dtype=np.int32)
    op_sites_0 = np.zeros(engine.M_total, dtype=np.int32)
    op_types_1 = np.ones(engine.M_total, dtype=np.int32)
    op_sites_1 = np.zeros(engine.M_total, dtype=np.int32)

    # Site flips keep the tested site paths closed at the right boundary.
    op_types_0[[1, 8]] = -1
    op_sites_0[[1, 8]] = 2
    op_types_1[[2, 9]] = -1
    op_sites_1[[2, 9]] = 2

    # Bond operators touching the promoted site exercise real delta-slice weights.
    op_types_0[[3, 7]] = 2
    op_sites_0[[3, 7]] = [1, 5]
    op_types_1[[4, 6]] = 2
    op_sites_1[[4, 6]] = [1, 5]
    return op_types_0, op_sites_0, op_types_1, op_sites_1


def _configured_engine(delta_groups):
    engine = QAQMCRenyiRydberg(
        N=6,
        M=6,
        Omega=1.0,
        Rb=1.2,
        delta_min=0.0,
        delta_max=2.0,
        seed=17,
        delta_groups=delta_groups,
    )
    lower = np.zeros(engine.N, dtype=np.uint8)
    upper = lower.copy()
    upper[2] = 1

    op_types_0, op_sites_0, op_types_1, op_sites_1 = _make_manual_strings(engine)
    engine.set_replica_op_string(0, op_types_0, op_sites_0)
    engine.set_replica_op_string(1, op_types_1, op_sites_1)
    engine.set_topology_pair(lower, upper, diff_site=2)
    return engine


def _test_delta_groups_property_is_exposed_and_clamped():
    engine = QAQMCRenyiRydberg(N=4, M=3, seed=5, delta_groups=999)
    assert engine.delta_groups == engine.M_total
    assert engine._cpp_engine.delta_groups == engine.M_total


def _test_grouped_log_weight_ratio_matches_full_precompute():
    full = _configured_engine(delta_groups=0)
    grouped = _configured_engine(delta_groups=3)

    full_forward = full.log_weight_ratio_for_site(2, 0, 1)
    grouped_forward = grouped.log_weight_ratio_for_site(2, 0, 1)
    full_backward = full.log_weight_ratio_for_site(2, 1, 0)
    grouped_backward = grouped.log_weight_ratio_for_site(2, 1, 0)

    assert np.isfinite(grouped_forward)
    assert np.isclose(grouped_forward, full_forward)
    assert np.isclose(grouped_backward, full_backward)
    assert np.isclose(grouped_forward, -grouped_backward)


def _test_grouped_engine_runs_pair_toggle_steps():
    engine = _configured_engine(delta_groups=4)
    engine.run_steps(12)
    counts = engine.get_visit_counts()
    assert counts.sum() == 12
    assert counts[0] >= 0
    assert counts[1] >= 0


def main():
    _test_delta_groups_property_is_exposed_and_clamped()
    _test_grouped_log_weight_ratio_matches_full_precompute()
    _test_grouped_engine_runs_pair_toggle_steps()
    print("QAQMC Renyi delta-groups unit checks passed")


if __name__ == "__main__":
    main()
