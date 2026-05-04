import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tee.ensemble_ladder import build_boundary_first_site_order
from src.rydberg.hamiltonian import build_rydberg_vij
from src.rydberg.lattices import generate_1d_chain


def _test_chain_order_with_real_bonds():
    pos = generate_1d_chain(6)
    _, _, _, _, bond_sites, _ = build_rydberg_vij(
        N=6, Omega=1.0, Rb=1.2, pos=pos, verbose=False, neighbor_cutoff=1
    )
    mask = np.zeros(6, dtype=np.uint8)
    mask[[1, 2, 3, 4]] = 1
    order = build_boundary_first_site_order(mask, bond_sites)
    assert order == [1, 2, 3, 4]


def main():
    _test_chain_order_with_real_bonds()
    print("Ensemble ladder integration checks passed")


if __name__ == "__main__":
    main()
