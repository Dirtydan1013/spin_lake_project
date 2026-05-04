from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree

from src.rydberg.lattices import (
    generate_kagome_bond_lattice,
    generate_kagome_bond_triangle_lattice,
    generate_kagome_lattice,
    kagome_edge_patch_graph,
    kagome_hex_centers,
)


def test_generate_kagome_lattice_single_void_has_six_vertices():
    pos = generate_kagome_lattice(1, 1, a=1.0)
    assert pos.shape == (6, 2)

    radii = np.linalg.norm(pos, axis=1)
    assert np.allclose(radii, 0.5)


def test_generate_kagome_bond_sites_are_midpoints_of_kagome_edges_for_single_void():
    kagome = generate_kagome_lattice(1, 1, a=1.0)
    bond = generate_kagome_bond_lattice(1, 1, a=1.0)

    expected = []
    for i in range(6):
        j = (i + 1) % 6
        expected.append(0.5 * (kagome[i] + kagome[j]))
    expected = np.array(expected)

    tree = KDTree(expected)
    dists, _ = tree.query(bond)
    assert np.allclose(dists, 0.0, atol=1e-10)


def test_kagome_hex_centers_use_row_major_bottom_to_top_ordering():
    centers = kagome_hex_centers(3, 2, a=1.0)
    expected = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [0.5, np.sqrt(3.0) / 2.0],
        [1.5, np.sqrt(3.0) / 2.0],
        [2.5, np.sqrt(3.0) / 2.0],
    ])
    assert np.allclose(centers, expected)


def test_edge_terminated_kagome_bond_lattice_uses_retained_kagome_edge_midpoints():
    nx, ny = 3, 2
    vertices, edges = kagome_edge_patch_graph(nx, ny, a=1.0)
    pos = generate_kagome_bond_triangle_lattice(nx, ny, a=1.0)
    expected = 0.5 * (vertices[edges[:, 0]] + vertices[edges[:, 1]])

    assert pos.shape == expected.shape
    assert pos.shape == (6 * nx * ny, 2)
    assert edges.shape == (6 * nx * ny, 2)
    assert np.allclose(pos, expected)
    assert np.unique(np.round(pos, 10), axis=0).shape[0] == len(pos)
