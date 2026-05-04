import os
import sys
import tempfile

import numpy as np
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.analysis.ed_core import build_qaqmc_midpoint_state
from src.rydberg.lattices import generate_1d_chain
from src.tee.reweighting import load_expanded_result_hdf5
from src.mpi.reweighting_mpi import run_expanded_mpi


def _reduced_density_matrix(psi, region_mask):
    n_sites = int(np.log2(psi.size))
    region = [i for i in range(n_sites) if region_mask[i]]
    complement = [i for i in range(n_sites) if not region_mask[i]]
    mat = np.zeros((1 << len(region), 1 << len(complement)), dtype=np.complex128)
    for state in range(1 << n_sites):
        row = 0
        col = 0
        for bit, site in enumerate(region):
            row |= (((state >> site) & 1) << bit)
        for bit, site in enumerate(complement):
            col |= (((state >> site) & 1) << bit)
        mat[row, col] = psi[state]
    return mat @ mat.conj().T


def _test_single_rank_real_expanded_mpi():
    N = 4
    Omega = 1.0
    Rb = 1.2
    delta_min = 0.0
    delta_max = 1.5
    M = 100
    pos = generate_1d_chain(N)
    region_mask = np.array([1, 1, 0, 0], dtype=np.uint8)

    psi = build_qaqmc_midpoint_state(
        N=N,
        Omega=Omega,
        delta_min=delta_min,
        delta_max=delta_max,
        Rb=Rb,
        M=M,
        pos=pos,
        neighbor_cutoff=1,
    )
    rho = _reduced_density_matrix(psi, region_mask)
    s2_ed = -np.log(float(np.real(np.trace(rho @ rho))))

    masks = [
        np.array([0, 0, 0, 0], dtype=np.uint8),
        np.array([1, 0, 0, 0], dtype=np.uint8),
        np.array([1, 1, 0, 0], dtype=np.uint8),
    ]
    neighbors = [[1], [0, 2], [1]]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "expanded_mpi.h5")
        result = run_expanded_mpi(
            N=N,
            M=M,
            masks=masks,
            neighbors=neighbors,
            target_ensemble=2,
            Omega=Omega,
            Rb=Rb,
            delta_min=delta_min,
            delta_max=delta_max,
            pos=pos,
            seed=1234,
            neighbor_cutoff=1,
            autotune_steps_per_iter=12000,
            autotune_max_iters=6,
            autotune_tol=1.18,
            autotune_method="transition_matrix",
            autotune_damping=0.7,
            n_steps=120000,
            block_size=2000,
            filepath=path,
            comm=MPI.COMM_SELF,
        )

        s2_coll, s2_coll_err = result.production.s2_collection(2)
        assert result.auto_tune.collection_counts is not None
        assert abs(s2_coll - s2_ed) < 0.05
        assert abs(s2_coll - s2_ed) < 3.5 * s2_coll_err

        loaded = load_expanded_result_hdf5(path)
        loaded_s2, loaded_err = loaded["result"].s2_collection(2)
        assert np.isclose(loaded_s2, s2_coll)
        assert np.isclose(loaded_err, s2_coll_err)
        assert loaded["masks"].tolist() == [mask.tolist() for mask in masks]
        assert loaded["neighbors"] == neighbors
        assert loaded["auto_tune"] is not None


def main():
    _test_single_rank_real_expanded_mpi()
    print("Expanded reweighting MPI integration checks passed")


if __name__ == "__main__":
    main()
