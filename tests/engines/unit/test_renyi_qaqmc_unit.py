import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.rydberg.hamiltonian import build_rydberg_vij
from src.engines.qaqmc_renyi import QAQMCRenyiRydberg


def _make_engine():
    return QAQMCRenyiRydberg(N=4, M=3, seed=7)


def _bond_weight(delta, coord_number, vij, si, sj, n_i, n_j, epsilon=0.01):
    delta_i = delta / coord_number[si] if coord_number[si] > 0 else 0.0
    delta_j = delta / coord_number[sj] if coord_number[sj] > 0 else 0.0
    raw = np.array([0.0, delta_j, delta_i, -vij + delta_i + delta_j], dtype=np.float64)
    m_min = float(np.min(raw))
    m_abs = float(np.min(np.abs(raw)))
    c_ij = ((-m_min) if m_min < 0.0 else 0.0) + epsilon * m_abs
    return float(raw[n_i * 2 + n_j] + c_ij)


def _build_channel_occupancies(types0, sites0, types1, sites1, mask, M):
    M_total = 2 * M
    N = mask.size
    occ = np.zeros((2, M_total + 1, N), dtype=np.int32)
    for channel in range(2):
        for site in range(N):
            value = 0
            for p in range(M_total):
                replica = channel if p < M else (1 - channel if mask[site] else channel)
                if replica == 0 and types0[p] == -1 and sites0[p] == site:
                    value ^= 1
                if replica == 1 and types1[p] == -1 and sites1[p] == site:
                    value ^= 1
                occ[channel, p + 1, site] = value
    return occ


def _global_log_weight(types0, sites0, types1, sites1, mask, delta_schedule, bond_sites, vij_list, coord_number):
    M_total = delta_schedule.size
    M = M_total // 2
    occ = _build_channel_occupancies(types0, sites0, types1, sites1, mask, M)

    if np.any(occ[:, M_total, :] != 0):
        return -1e30

    log_weight = 0.0
    for replica, (op_types, op_sites) in enumerate(((types0, sites0), (types1, sites1))):
        for p in range(M_total):
            if op_types[p] != 2:
                continue
            bond = op_sites[p]
            si, sj = bond_sites[bond]
            c_i = replica if p < M else (1 - replica if mask[si] else replica)
            c_j = replica if p < M else (1 - replica if mask[sj] else replica)
            n_i = int(occ[c_i, p, si])
            n_j = int(occ[c_j, p, sj])
            w = _bond_weight(
                float(delta_schedule[p]),
                coord_number,
                float(vij_list[bond]),
                int(si),
                int(sj),
                n_i,
                n_j,
            )
            if w <= 1e-300:
                return -1e30
            log_weight += np.log(w)
    return float(log_weight)


def _test_stitched_paths_single_joined_site():
    engine = _make_engine()
    mask = np.zeros(engine.N, dtype=np.uint8)
    mask[0] = 1
    engine.set_A_mask(mask)

    op_types_0 = np.ones(engine.M_total, dtype=np.int32)
    op_sites_0 = np.zeros(engine.M_total, dtype=np.int32)
    op_types_0[0] = -1
    op_types_0[3] = -1

    op_types_1 = np.ones(engine.M_total, dtype=np.int32)
    op_sites_1 = np.zeros(engine.M_total, dtype=np.int32)
    op_types_1[1] = -1
    op_types_1[4] = -1

    engine.set_replica_op_string(0, op_types_0, op_sites_0)
    engine.set_replica_op_string(1, op_types_1, op_sites_1)
    engine.recompute_midpoint_states()

    paths = engine.get_site_paths(0)
    assert paths["replica_0"].tolist() == [0, 1, 1, 1, 0, 0, 0]
    assert paths["replica_1"].tolist() == [0, 0, 1, 1, 1, 0, 0]
    assert paths["channel_0"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert paths["channel_1"].tolist() == [0, 0, 1, 1, 0, 0, 0]


def _test_indicator_matches_midpoint_state():
    engine = _make_engine()
    mask = np.zeros(engine.N, dtype=np.uint8)
    mask[0] = 1
    engine.set_A_mask(mask)

    op_types_0 = np.ones(engine.M_total, dtype=np.int32)
    op_sites_0 = np.zeros(engine.M_total, dtype=np.int32)
    op_types_0[0] = -1

    op_types_1 = np.ones(engine.M_total, dtype=np.int32)
    op_sites_1 = np.zeros(engine.M_total, dtype=np.int32)
    op_types_1[0] = -1

    engine.set_replica_op_string(0, op_types_0, op_sites_0)
    engine.set_replica_op_string(1, op_types_1, op_sites_1)
    engine.recompute_midpoint_states()
    engine.set_indicator_site(0)
    assert engine.current_indicator() == 1

    op_types_1[0] = 1
    engine.set_replica_op_string(1, op_types_1, op_sites_1)
    engine.recompute_midpoint_states()
    assert engine.current_indicator() == 0


def _test_topology_pair_state_and_visit_count_reset():
    engine = _make_engine()
    lower = np.zeros(engine.N, dtype=np.uint8)
    upper = lower.copy()
    upper[2] = 1

    engine.set_topology_pair(lower, upper, diff_site=2)
    assert engine.current_topology == 0
    assert engine.diff_site == 2
    assert engine.A_mask.tolist() == lower.tolist()
    assert engine.get_topology_mask(0).tolist() == lower.tolist()
    assert engine.get_topology_mask(1).tolist() == upper.tolist()
    assert engine.get_visit_counts().tolist() == [0, 0]

    engine.run_steps(4)
    counts = engine.get_visit_counts()
    assert counts.sum() == 4

    engine.reset_visit_counts()
    assert engine.get_visit_counts().tolist() == [0, 0]


def _test_topology_log_ratio_is_antisymmetric():
    engine = _make_engine()
    lower = np.zeros(engine.N, dtype=np.uint8)
    upper = lower.copy()
    upper[1] = 1

    op_types_0 = np.ones(engine.M_total, dtype=np.int32)
    op_sites_0 = np.zeros(engine.M_total, dtype=np.int32)
    op_types_0[0] = -1
    op_sites_0[0] = 1

    op_types_1 = np.ones(engine.M_total, dtype=np.int32)
    op_sites_1 = np.zeros(engine.M_total, dtype=np.int32)
    op_types_1[1] = -1
    op_sites_1[1] = 1

    engine.set_replica_op_string(0, op_types_0, op_sites_0)
    engine.set_replica_op_string(1, op_types_1, op_sites_1)
    engine.set_topology_pair(lower, upper, diff_site=1)

    forward = engine.log_weight_ratio_for_site(1, 0, 1)
    backward = engine.log_weight_ratio_for_site(1, 1, 0)
    assert np.isfinite(forward)
    assert np.isfinite(backward)
    assert np.isclose(forward, -backward)


def _test_same_slice_replica_site_ops_propagate_independently():
    engine = _make_engine()
    mask = np.zeros(engine.N, dtype=np.uint8)
    engine.set_A_mask(mask)

    op_types_0 = np.ones(engine.M_total, dtype=np.int32)
    op_sites_0 = np.zeros(engine.M_total, dtype=np.int32)
    op_types_0[1] = -1
    op_sites_0[1] = 2

    op_types_1 = np.ones(engine.M_total, dtype=np.int32)
    op_sites_1 = np.zeros(engine.M_total, dtype=np.int32)
    op_types_1[1] = -1
    op_sites_1[1] = 2

    engine.set_replica_op_string(0, op_types_0, op_sites_0)
    engine.set_replica_op_string(1, op_types_1, op_sites_1)
    engine.recompute_midpoint_states()

    paths = engine.get_site_paths(2)
    expected = [0, 0, 1, 1, 1, 1, 1]
    assert paths["replica_0"].tolist() == expected
    assert paths["replica_1"].tolist() == expected
    assert paths["channel_0"].tolist() == expected
    assert paths["channel_1"].tolist() == expected


def _test_topology_log_ratio_matches_bruteforce_global_weight():
    engine = _make_engine()
    lower = np.zeros(engine.N, dtype=np.uint8)
    upper = lower.copy()
    upper[1] = 1

    _, bonds_i, bonds_j, vij_list, bond_sites, coord_number = build_rydberg_vij(
        N=engine.N,
        Omega=1.0,
        Rb=1.2,
        pos=engine.pos,
        verbose=False,
    )
    bond_map = {(int(si), int(sj)): idx for idx, (si, sj) in enumerate(zip(bonds_i, bonds_j))}
    b01 = bond_map[(0, 1)]
    b12 = bond_map[(1, 2)]

    op_types_0 = np.ones(engine.M_total, dtype=np.int32)
    op_sites_0 = np.zeros(engine.M_total, dtype=np.int32)
    op_types_1 = np.ones(engine.M_total, dtype=np.int32)
    op_sites_1 = np.zeros(engine.M_total, dtype=np.int32)

    op_types_0[1] = -1
    op_sites_0[1] = 1
    op_types_0[4] = -1
    op_sites_0[4] = 1
    op_types_1[2] = -1
    op_sites_1[2] = 1
    op_types_1[5] = -1
    op_sites_1[5] = 1

    op_types_0[0] = 2
    op_sites_0[0] = b01
    op_types_0[3] = 2
    op_sites_0[3] = b12
    op_types_1[1] = 2
    op_sites_1[1] = b12
    op_types_1[4] = 2
    op_sites_1[4] = b01

    engine.set_replica_op_string(0, op_types_0, op_sites_0)
    engine.set_replica_op_string(1, op_types_1, op_sites_1)
    engine.set_topology_pair(lower, upper, diff_site=1)

    brute_low = _global_log_weight(
        op_types_0,
        op_sites_0,
        op_types_1,
        op_sites_1,
        lower,
        engine.delta_schedule,
        bond_sites,
        vij_list,
        coord_number,
    )
    brute_high = _global_log_weight(
        op_types_0,
        op_sites_0,
        op_types_1,
        op_sites_1,
        upper,
        engine.delta_schedule,
        bond_sites,
        vij_list,
        coord_number,
    )
    brute_ratio = brute_high - brute_low
    cpp_ratio = engine.log_weight_ratio_for_site(1, 0, 1)
    assert np.isfinite(brute_ratio)
    assert np.isclose(cpp_ratio, brute_ratio)


def _test_expanded_ladder_state_and_resets():
    engine = _make_engine()
    masks = [
        np.zeros(engine.N, dtype=np.uint8),
        np.array([1, 0, 0, 0], dtype=np.uint8),
        np.array([1, 1, 0, 0], dtype=np.uint8),
    ]
    neighbors = [[1], [0, 2], [1]]

    engine.set_ensemble_ladder(masks, neighbors, initial_ensemble=1)
    assert engine.mode == 1
    assert engine.ensemble_count == 3
    assert engine.current_ensemble == 1
    assert engine.A_mask.tolist() == [1, 0, 0, 0]
    assert engine.get_visit_counts_ext().tolist() == [0, 0, 0]
    assert engine.get_transition_counts().shape == (3, 3)
    assert engine.get_collection_counts().shape == (3, 3)

    engine.run_steps(3)
    assert int(engine.get_visit_counts_ext().sum()) == 3
    assert float(np.sum(engine.get_collection_counts())) > 0.0

    engine.reset_visit_counts_ext()
    engine.reset_transition_counts()
    engine.reset_collection_counts()
    assert engine.get_visit_counts_ext().tolist() == [0, 0, 0]
    assert np.all(engine.get_transition_counts() == 0)
    assert np.all(engine.get_collection_counts() == 0.0)


def _test_expanded_switch_follows_log_g_bias():
    engine = _make_engine()
    masks = [
        np.zeros(engine.N, dtype=np.uint8),
        np.array([1, 0, 0, 0], dtype=np.uint8),
    ]
    neighbors = [[1], [0]]

    op_types = np.ones(engine.M_total, dtype=np.int32)
    op_sites = np.zeros(engine.M_total, dtype=np.int32)
    engine.set_replica_op_string(0, op_types, op_sites)
    engine.set_replica_op_string(1, op_types, op_sites)

    engine.set_ensemble_ladder(masks, neighbors, initial_ensemble=0)
    engine.set_log_g(np.array([0.0, 8.0], dtype=np.float64))
    engine.reset_transition_counts()
    engine.run_steps(10)

    assert engine.current_ensemble == 1
    assert engine.get_visit_counts_ext()[1] >= engine.get_visit_counts_ext()[0]
    assert int(engine.get_transition_counts()[0, 1]) >= 1


def main():
    _test_stitched_paths_single_joined_site()
    _test_indicator_matches_midpoint_state()
    _test_topology_pair_state_and_visit_count_reset()
    _test_topology_log_ratio_is_antisymmetric()
    _test_same_slice_replica_site_ops_propagate_independently()
    _test_topology_log_ratio_matches_bruteforce_global_weight()
    _test_expanded_ladder_state_and_resets()
    _test_expanded_switch_follows_log_g_bias()
    print("Renyi QAQMC unit checks passed")


if __name__ == "__main__":
    main()
