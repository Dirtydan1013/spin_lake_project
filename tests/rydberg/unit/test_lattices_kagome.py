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


def test_generate_kagome_lattice_periodic_has_three_site_basis():
    nx, ny, a = 4, 3, 2.0
    pos = generate_kagome_lattice(nx, ny, a, boundary="periodic")
    assert pos.shape == (3 * nx * ny, 2)

    # unique modulo the torus
    from src.rydberg.lattices import lattice_box_vectors
    B = lattice_box_vectors("kagome", nx, ny, a)
    frac = np.linalg.solve(np.asarray(B).T, pos.T).T % 1.0
    assert np.unique(np.round(frac, 8) % 1.0, axis=0).shape[0] == len(pos)

    # min-image nn distance a/2 and kagome coordination 4 for every site
    shifts = [i * B[0] + j * B[1] for i in (-1, 0, 1) for j in (-1, 0, 1)]
    D = np.full((len(pos), len(pos)), np.inf)
    for s in shifts:
        d = np.linalg.norm(pos[None, :, :] + s[None, None, :] - pos[:, None, :],
                           axis=-1)
        D = np.minimum(D, d)
    np.fill_diagonal(D, np.inf)
    assert np.isclose(D.min(), a / 2.0)
    coord = (np.abs(D - a / 2.0) < 1e-9).sum(axis=1)
    assert set(np.unique(coord)) == {4}


def test_generate_kagome_lattice_periodic_sites_lie_on_open_vertex_family():
    nx, ny, a = 3, 3, 1.0
    pos_p = generate_kagome_lattice(nx, ny, a, boundary="periodic")
    # oversized open patch covers every periodic representative
    pos_o = generate_kagome_lattice(nx + 1, ny + 1, a, boundary="open")
    open_set = {tuple(q) for q in np.round(pos_o, 8).tolist()}
    assert all(tuple(q) in open_set for q in np.round(pos_p, 8).tolist())
