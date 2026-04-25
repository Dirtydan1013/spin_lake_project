"""
Helpers for constructing ratio-estimator site orderings and, later, ensemble ladders.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def _normalise_mask(region_mask) -> np.ndarray:
    mask = np.asarray(region_mask, dtype=np.uint8).reshape(-1)
    return (mask > 0).astype(np.uint8)


def _normalise_bond_sites(bond_sites, n_sites: int) -> np.ndarray:
    bonds_raw = np.asarray(bond_sites)
    if bonds_raw.ndim != 2 or bonds_raw.shape[1] != 2:
        raise ValueError("bond_sites must be an array of site-index pairs with shape (n_bonds, 2)")
    if not np.issubdtype(bonds_raw.dtype, np.integer):
        if not np.all(np.isfinite(bonds_raw)) or not np.allclose(bonds_raw, np.rint(bonds_raw)):
            raise ValueError(
                "bond_sites must contain integer site indices, not coordinates; "
                "use a geometry adjacency list for KP site ordering"
            )
    bonds = bonds_raw.astype(np.int32, copy=False)
    if bonds.size and (int(np.min(bonds)) < 0 or int(np.max(bonds)) >= int(n_sites)):
        raise ValueError("bond_sites contains site indices outside the region mask length")
    return bonds


def build_region_adjacency(region_mask, bond_sites) -> dict[int, list[int]]:
    """Build the induced graph on the selected region sites."""
    mask = _normalise_mask(region_mask)
    bonds = _normalise_bond_sites(bond_sites, mask.size)
    adjacency = {site: set() for site in np.flatnonzero(mask)}
    for si, sj in bonds:
        if si == sj:
            continue
        if mask[si] and mask[sj]:
            adjacency[int(si)].add(int(sj))
            adjacency[int(sj)].add(int(si))
    return {site: sorted(neigh) for site, neigh in adjacency.items()}


def boundary_sites(region_mask, bond_sites) -> list[int]:
    """Sites in the region that touch at least one site outside the region."""
    mask = _normalise_mask(region_mask)
    bonds = _normalise_bond_sites(bond_sites, mask.size)
    boundary = set()
    for si, sj in bonds:
        if mask[si] != mask[sj]:
            if mask[si]:
                boundary.add(int(si))
            if mask[sj]:
                boundary.add(int(sj))
    if boundary:
        return sorted(boundary)
    return sorted(int(s) for s in np.flatnonzero(mask))


def build_boundary_first_site_order(region_mask, bond_sites, seed_site: int | None = None) -> list[int]:
    """Order region sites by BFS starting from a boundary site.

    The result is a boundary-first heuristic suitable for Path C ratio-estimator
    site promotion: each added site belongs to the target region and is adjacent
    to at least one previously added site within its connected component.
    """
    mask = _normalise_mask(region_mask)
    region_sites = sorted(int(s) for s in np.flatnonzero(mask))
    if not region_sites:
        return []

    adjacency = build_region_adjacency(mask, bond_sites)
    bsites = boundary_sites(mask, bond_sites)
    if seed_site is not None:
        if seed_site not in adjacency:
            raise ValueError("seed_site must belong to the selected region")
        seeds = [int(seed_site)] + [s for s in bsites if s != seed_site]
    else:
        seeds = list(bsites)

    visited = set()
    order: list[int] = []

    def bfs(start: int):
        queue = deque([start])
        visited.add(start)
        while queue:
            cur = queue.popleft()
            order.append(cur)
            for nxt in adjacency[cur]:
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)

    for start in seeds:
        if start not in visited:
            bfs(start)

    for site in region_sites:
        if site not in visited:
            bfs(site)

    return order
