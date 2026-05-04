import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.analysis.ed_core import build_qaqmc_midpoint_state, build_rydberg_hamiltonian
from src.rydberg.lattices import generate_1d_chain


def _test_midpoint_state_is_normalized():
    psi = build_qaqmc_midpoint_state(
        N=4,
        Omega=1.0,
        delta_min=0.0,
        delta_max=1.5,
        Rb=1.2,
        M=20,
        pos=generate_1d_chain(4),
        neighbor_cutoff=1,
    )
    assert np.isclose(np.linalg.norm(psi), 1.0)


def _test_hamiltonian_neighbor_cutoff_changes_spectrum():
    pos = generate_1d_chain(4)
    h_full = build_rydberg_hamiltonian(4, 1.0, 0.8, 1.2, pos, neighbor_cutoff=None)
    h_nn = build_rydberg_hamiltonian(4, 1.0, 0.8, 1.2, pos, neighbor_cutoff=1)
    assert not np.allclose(h_full, h_nn)


def main():
    _test_midpoint_state_is_normalized()
    _test_hamiltonian_neighbor_cutoff_changes_spectrum()
    print("ED QAQMC midpoint unit checks passed")


if __name__ == "__main__":
    main()
