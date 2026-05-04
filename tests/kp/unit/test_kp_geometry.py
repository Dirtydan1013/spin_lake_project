from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.kp.kp_geometry import (
    KP_REGION_NAMES,
    build_cropped_kagome_bond_ordering_bonds,
    build_cropped_kp_region_masks,
    build_kagome_bond_ordering_bonds,
    build_kp_region_masks,
    build_region_ladder,
)
from src.tee.ensemble_ladder import build_region_adjacency
from src.rydberg.lattices import generate_kagome_bond_lattice


def _test_m2_region_a_indices_match_known_construction():
    # Reference indices were hand-computed with centre at C35; pin it explicitly
    # so the test is independent of the auto-centre default.
    spec = build_kp_region_masks(8, 8, m=2, preferred_center_label="C35")
    expected = np.array(
        [115, 156, 157, 161, 163, 164, 165, 198, 203, 204, 205, 206,
         207, 208, 209, 212, 213, 246, 250, 251, 254, 255, 256, 298],
        dtype=np.int32,
    )
    assert np.array_equal(spec.region_indices["A"], expected)


def _test_boundary_rule_rejects_too_large_region():
    try:
        build_kp_region_masks(8, 8, m=3)
    except ValueError as exc:
        assert "No valid center" in str(exc)
    else:
        raise AssertionError("expected m=3 on 8x8 to violate the safety margin rule")


def _test_region_ladder_is_atom_by_atom_chain():
    mask = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
    bond_sites = np.array([[1, 2], [2, 4]], dtype=np.int32)
    ladder = build_region_ladder("X", mask, bond_sites=bond_sites)
    assert ladder.target_ensemble == 3
    assert len(ladder.masks) == 4
    assert ladder.neighbors == [[1], [0, 2], [1, 3], [2]]
    assert np.all(ladder.masks[0] == 0)
    assert int(np.sum(ladder.masks[-1])) == 3


def _test_kagome_ordering_bonds_connect_kp_regions():
    spec = build_kp_region_masks(8, 8, m=2)
    ordering_bonds = build_kagome_bond_ordering_bonds(8, 8)
    assert ordering_bonds.ndim == 2 and ordering_bonds.shape[1] == 2
    assert np.issubdtype(ordering_bonds.dtype, np.integer)

    for name in KP_REGION_NAMES:
        adjacency = build_region_adjacency(spec.region_masks[name], ordering_bonds)
        sites = list(adjacency)
        seen = {sites[0]}
        queue = [sites[0]]
        for site in queue:
            for nxt in adjacency[site]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        assert len(seen) == len(sites), f"{name} should be connected by ordering bonds"


def _test_cropped_kp_geometry_matches_center_crop_lattice():
    spec = build_cropped_kp_region_masks(8, 8, m=2)
    ordering_bonds = build_cropped_kagome_bond_ordering_bonds(8, 8)

    assert spec.center_label == "C40"
    assert all(int(spec.region_masks[name].sum()) == 24 for name in ("A", "B", "C"))
    assert int(spec.region_masks["ABC"].sum()) == 72
    assert all(mask.shape == (384,) for mask in spec.region_masks.values())
    assert np.max(spec.region_masks["A"] + spec.region_masks["B"] + spec.region_masks["C"]) == 1

    for name in KP_REGION_NAMES:
        adjacency = build_region_adjacency(spec.region_masks[name], ordering_bonds)
        sites = list(adjacency)
        seen = {sites[0]}
        queue = [sites[0]]
        for site in queue:
            for nxt in adjacency[site]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        assert len(seen) == len(sites), f"cropped {name} should be connected"


def _test_ladder_rejects_coordinate_array_as_bonds():
    mask = np.ones(6, dtype=np.uint8)
    pos = generate_kagome_bond_lattice(1, 1)
    try:
        build_region_ladder("bad", mask, bond_sites=pos)
    except ValueError as exc:
        assert "not coordinates" in str(exc)
    else:
        raise AssertionError("coordinate arrays must not be accepted as bond_sites")


def main():
    _test_m2_region_a_indices_match_known_construction()
    _test_boundary_rule_rejects_too_large_region()
    _test_region_ladder_is_atom_by_atom_chain()
    _test_kagome_ordering_bonds_connect_kp_regions()
    _test_cropped_kp_geometry_matches_center_crop_lattice()
    _test_ladder_rejects_coordinate_array_as_bonds()
    print("KP geometry unit checks passed")


if __name__ == "__main__":
    main()
