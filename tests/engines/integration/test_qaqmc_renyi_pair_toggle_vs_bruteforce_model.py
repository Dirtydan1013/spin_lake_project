import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.rydberg.hamiltonian import build_rydberg_vij
from src.rydberg.lattices import generate_1d_chain
from src.engines.qaqmc_renyi import QAQMCRenyiRydberg
from src.tee.qaqmc_renyi_ratio import RatioRunner


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


def _global_weight(types0, sites0, types1, sites1, mask, delta_schedule, bond_sites, vij_list, coord_number):
    M_total = delta_schedule.size
    M = M_total // 2
    occ = _build_channel_occupancies(types0, sites0, types1, sites1, mask, M)
    if np.any(occ[:, M_total, :] != 0):
        return 0.0

    weight = 1.0
    site_weight = 0.5
    for op_types, op_sites in ((types0, sites0), (types1, sites1)):
        for p in range(M_total):
            ot = int(op_types[p])
            if ot == 2:
                bond = int(op_sites[p])
                si, sj = bond_sites[bond]
                replica = 0 if op_types is types0 else 1
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
                    return 0.0
                weight *= w
            else:
                weight *= site_weight
    return float(weight)


def _enumerate_exact_ratio(delta_schedule, bond_sites, vij_list, coord_number, lower_mask, upper_mask):
    choices = [
        (1, 0),
        (1, 1),
        (-1, 0),
        (-1, 1),
        (2, 0),
    ]
    z_low = 0.0
    z_high = 0.0
    for config in itertools.product(choices, repeat=4):
        types0 = np.array([config[0][0], config[1][0]], dtype=np.int32)
        sites0 = np.array([config[0][1], config[1][1]], dtype=np.int32)
        types1 = np.array([config[2][0], config[3][0]], dtype=np.int32)
        sites1 = np.array([config[2][1], config[3][1]], dtype=np.int32)
        z_low += _global_weight(types0, sites0, types1, sites1, lower_mask, delta_schedule, bond_sites, vij_list, coord_number)
        z_high += _global_weight(types0, sites0, types1, sites1, upper_mask, delta_schedule, bond_sites, vij_list, coord_number)
    return float(z_high / z_low)


def _test_pair_toggle_matches_bruteforce_operator_string_model():
    N = 2
    M = 1
    Omega = 1.0
    Rb = 1.2
    delta_min = 0.3
    delta_max = 0.9
    pos = generate_1d_chain(N)

    lower_mask = np.zeros(N, dtype=np.uint8)
    upper_mask = lower_mask.copy()
    upper_mask[0] = 1

    _, _b_i, _b_j, vij_list, bond_sites, coord_number = build_rydberg_vij(
        N=N,
        Omega=Omega,
        Rb=Rb,
        pos=pos,
        verbose=False,
    )
    delta_schedule = np.array([delta_min, delta_max], dtype=np.float64)
    exact_ratio = _enumerate_exact_ratio(
        delta_schedule,
        bond_sites,
        vij_list,
        coord_number,
        lower_mask,
        upper_mask,
    )

    runner = RatioRunner(
        engine=QAQMCRenyiRydberg(
            N=N,
            M=M,
            Omega=Omega,
            Rb=Rb,
            delta_min=delta_min,
            delta_max=delta_max,
            pos=pos,
            seed=17,
        )
    )
    result = runner.run_single_ratio(
        lower_mask,
        next_site=0,
        n_therm=5000,
        n_measure=60000,
        measure_stride=2,
    )

    assert abs(result.ratio - exact_ratio) < 0.05
    assert abs(result.ratio - exact_ratio) < 4.0 * result.ratio_err


def main():
    _test_pair_toggle_matches_bruteforce_operator_string_model()
    print("QAQMC Renyi pair-toggle vs brute-force model checks passed")


if __name__ == "__main__":
    main()
