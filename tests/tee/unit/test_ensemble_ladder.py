import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tee.ensemble_ladder import (
    boundary_sites,
    build_boundary_first_site_order,
    build_region_adjacency,
)


def _test_region_adjacency_chain():
    mask = np.array([0, 1, 1, 1, 0], dtype=np.uint8)
    bond_sites = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int32)
    adjacency = build_region_adjacency(mask, bond_sites)
    assert adjacency == {1: [2], 2: [1, 3], 3: [2]}


def _test_boundary_first_order_grows_by_adjacency():
    mask = np.array([0, 1, 1, 1, 0], dtype=np.uint8)
    bond_sites = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int32)
    order = build_boundary_first_site_order(mask, bond_sites)
    assert order == [1, 2, 3]
    visited = set()
    adjacency = build_region_adjacency(mask, bond_sites)
    for idx, site in enumerate(order):
        if idx == 0:
            assert site in boundary_sites(mask, bond_sites)
        else:
            assert any(neigh in visited for neigh in adjacency[site])
        visited.add(site)


def main():
    _test_region_adjacency_chain()
    _test_boundary_first_order_grows_by_adjacency()
    print("Ensemble ladder unit checks passed")


if __name__ == "__main__":
    main()
