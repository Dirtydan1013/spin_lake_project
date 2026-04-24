"""
Programmatic Kitaev-Preskill region construction on the kagome-bond lattice.

This module lifts the geometry that was previously explored in plotting scripts
into reusable code for production runs:

    lattice size (nx, ny) + KP size m
        -> KP regions A/B/C/AB/BC/CA/ABC as site masks / site-index lists
        -> boundary-first site orders for ratio-estimator Path C
        -> intermediate expanded-ensemble ladders for Path A

Conventions:
    - ``m=1`` is the minimal KP construction.
    - The outer hexagon side length is ``2*m`` in unit-cell-edge units.
    - The internal Y-cut branch length is ``m``.
    - A stricter one-cell safety margin is enforced: the construction may not
      touch the outermost unit-cell ring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path
from scipy.spatial import KDTree

from src.tee.ensemble_ladder import build_boundary_first_site_order
from src.rydberg.lattices import (
    generate_kagome_bond_lattice,
    generate_kagome_lattice,
    kagome_hex_centers,
)


KP_REGION_NAMES = ("A", "B", "C", "AB", "BC", "CA", "ABC")


@dataclass(frozen=True)
class KPRegionSpec:
    nx: int
    ny: int
    m: int
    center_label: str
    region_masks: dict[str, np.ndarray]
    region_indices: dict[str, np.ndarray]
    site_orders: dict[str, np.ndarray] | None
    outer_paths: list[list[str]]
    branch_paths: list[list[str]]


@dataclass(frozen=True)
class RegionLadderSpec:
    region_name: str
    site_order: np.ndarray
    masks: list[np.ndarray]
    neighbors: list[list[int]]
    target_ensemble: int


def _resolve_label(label: str, kagome: np.ndarray, centers: np.ndarray) -> np.ndarray:
    prefix = label[0]
    idx = int(label[1:])
    if prefix == "K":
        return kagome[idx]
    if prefix == "C":
        return centers[idx]
    raise ValueError(f"Unsupported point label: {label}")


def _center_label_to_ij(center_label: str, nx: int) -> tuple[int, int]:
    idx = int(center_label[1:])
    return idx % nx, idx // nx


def _center_label(nx: int, ij: tuple[int, int]) -> str:
    i, j = ij
    return f"C{j * nx + i}"


def _choose_center_ij(nx: int, ny: int, m: int,
                      preferred: tuple[int, int] | None) -> tuple[int, int]:
    """Pick a valid (i, j) void-index satisfying the one-cell safety margin.

    If ``preferred`` is ``None`` the geometric centre ``((nx-1)/2, (ny-1)/2)`` is
    used as the target. Otherwise the nearest valid position to ``preferred`` is
    returned; the exact ``preferred`` is kept when it is itself valid.
    Ties are broken lexicographically by ``(i, j)`` for reproducibility.
    """
    valid = []
    for j in range(ny):
        for i in range(nx):
            if i - m >= 1 and i + m <= nx - 2 and j - m >= 1 and j + m <= ny - 2:
                valid.append((i, j))
    if not valid:
        raise ValueError(
            f"No valid center for m={m} inside {nx}x{ny} lattice under the one-cell safety margin rule"
        )
    if preferred is not None and preferred in valid:
        return preferred
    if preferred is None:
        target_i = (nx - 1) / 2.0
        target_j = (ny - 1) / 2.0
    else:
        target_i, target_j = preferred
    return min(
        valid,
        key=lambda ij: (
            (ij[0] - target_i) ** 2 + (ij[1] - target_j) ** 2,
            ij[0],
            ij[1],
        ),
    )


def _parse_preferred_center(preferred_center_label, nx: int) -> tuple[int, int] | None:
    """Return ``None`` for the auto sentinel or parse a ``"Cxx"`` label into (i, j)."""
    if preferred_center_label is None:
        return None
    if isinstance(preferred_center_label, str):
        normalised = preferred_center_label.strip()
        if normalised == "" or normalised.lower() in {"auto", "none"}:
            return None
    return _center_label_to_ij(preferred_center_label, nx)


def _center_coords_to_label_map(arr: np.ndarray, prefix: str) -> dict[tuple[float, float], str]:
    out = {}
    for idx, p in enumerate(arr):
        out[(round(float(p[0]), 10), round(float(p[1]), 10))] = f"{prefix}{idx}"
    return out


def _nearest_k_label(k_tree: KDTree, midpoint: np.ndarray) -> str:
    dist, idx = k_tree.query(midpoint)
    if dist > 1e-8:
        raise ValueError(f"Could not resolve midpoint {midpoint} to a kagome vertex; nearest dist={dist}")
    return f"K{int(idx)}"


def _build_side_labels(
    c_seq: list[str],
    kagome: np.ndarray,
    centers: np.ndarray,
    k_tree: KDTree,
) -> list[str]:
    labels = [c_seq[0]]
    for a, b in zip(c_seq[:-1], c_seq[1:]):
        pa = _resolve_label(a, kagome, centers)
        pb = _resolve_label(b, kagome, centers)
        labels.append(_nearest_k_label(k_tree, 0.5 * (pa + pb)))
        labels.append(b)
    return labels


def _concat_paths(*paths: list[str]) -> list[str]:
    out: list[str] = []
    for path in paths:
        if not out:
            out.extend(path)
        else:
            out.extend(path[1:])
    return out


def build_kp_paths(
    nx: int,
    ny: int,
    *,
    m: int,
    preferred_center_label: str | None = None,
    a: float = 1.0,
) -> tuple[list[list[str]], list[list[str]], str]:
    if m <= 0:
        raise ValueError("m must be positive")

    kagome = generate_kagome_lattice(nx, ny, a)
    centers = kagome_hex_centers(nx, ny, a)
    k_tree = KDTree(kagome)
    preferred_ij = _parse_preferred_center(preferred_center_label, nx)
    ci, cj = _choose_center_ij(nx, ny, int(m), preferred_ij)
    center_label = _center_label(nx, (ci, cj))

    def clab(i: int, j: int) -> str:
        return f"C{j * nx + i}"

    # Corners in clockwise order starting from top-left.
    side_center_seqs = [
        [clab(ci - m + t, cj + m) for t in range(m + 1)],     # TL -> TR
        [clab(ci + t, cj + m - t) for t in range(m + 1)],     # TR -> R
        [clab(ci + m, cj - t) for t in range(m + 1)],         # R -> BR
        [clab(ci + m - t, cj - m) for t in range(m + 1)],     # BR -> BL
        [clab(ci - t, cj - m + t) for t in range(m + 1)],     # BL -> L
        [clab(ci - m, cj + t) for t in range(m + 1)],         # L -> TL
    ]
    side_paths = [_build_side_labels(seq, kagome, centers, k_tree) for seq in side_center_seqs]

    branch_center_seqs = [
        [clab(ci - t, cj + t) for t in range(m + 1)],         # center -> TL
        [clab(ci, cj - t) for t in range(m + 1)],             # center -> BL
        [clab(ci + t, cj) for t in range(m + 1)],             # center -> R
    ]
    branch_paths = [_build_side_labels(seq, kagome, centers, k_tree) for seq in branch_center_seqs]

    return side_paths, branch_paths, center_label


def _polygon_labels_for_regions(side_paths: list[list[str]], branch_paths: list[list[str]]) -> dict[str, list[str]]:
    labels_A = _concat_paths(
        list(reversed(side_paths[5])),
        list(reversed(side_paths[4])),
        list(reversed(branch_paths[1])),
        branch_paths[0],
    )
    labels_B = _concat_paths(
        side_paths[0],
        side_paths[1],
        list(reversed(branch_paths[2])),
        branch_paths[0],
    )
    labels_C = _concat_paths(
        branch_paths[2],
        side_paths[2],
        side_paths[3],
        list(reversed(branch_paths[1])),
    )
    return {"A": labels_A, "B": labels_B, "C": labels_C}


def _polygon_points(labels: list[str], kagome: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.array([_resolve_label(label, kagome, centers) for label in labels], dtype=np.float64)


def build_kp_region_masks(
    nx: int,
    ny: int,
    *,
    m: int,
    a: float = 1.0,
    preferred_center_label: str | None = None,
) -> KPRegionSpec:
    kagome = generate_kagome_lattice(nx, ny, a)
    centers = kagome_hex_centers(nx, ny, a)
    bond_pos = generate_kagome_bond_lattice(nx, ny, a)

    side_paths, branch_paths, center_label = build_kp_paths(
        nx, ny, m=m, preferred_center_label=preferred_center_label, a=a
    )
    poly_labels = _polygon_labels_for_regions(side_paths, branch_paths)
    region_masks: dict[str, np.ndarray] = {}
    region_indices: dict[str, np.ndarray] = {}
    atomic_masks: dict[str, np.ndarray] = {}

    for name in ("A", "B", "C"):
        polygon = _polygon_points(poly_labels[name], kagome, centers)
        inside = Path(polygon).contains_points(bond_pos, radius=-1e-9)
        mask = np.ascontiguousarray(inside.astype(np.uint8))
        atomic_masks[name] = mask
        region_masks[name] = mask
        region_indices[name] = np.flatnonzero(mask).astype(np.int32)

    for name, parts in {
        "AB": ("A", "B"),
        "BC": ("B", "C"),
        "CA": ("C", "A"),
        "ABC": ("A", "B", "C"),
    }.items():
        mask = np.zeros_like(atomic_masks["A"], dtype=np.uint8)
        for part in parts:
            mask = np.maximum(mask, atomic_masks[part])
        region_masks[name] = mask
        region_indices[name] = np.flatnonzero(mask).astype(np.int32)

    return KPRegionSpec(
        nx=int(nx),
        ny=int(ny),
        m=int(m),
        center_label=str(center_label),
        region_masks=region_masks,
        region_indices=region_indices,
        site_orders=None,
        outer_paths=side_paths,
        branch_paths=branch_paths,
    )


def attach_kp_site_orders(spec: KPRegionSpec, bond_sites) -> KPRegionSpec:
    site_orders = {
        name: np.asarray(
            build_boundary_first_site_order(mask, bond_sites),
            dtype=np.int32,
        )
        for name, mask in spec.region_masks.items()
    }
    return KPRegionSpec(
        nx=spec.nx,
        ny=spec.ny,
        m=spec.m,
        center_label=spec.center_label,
        region_masks={k: v.copy() for k, v in spec.region_masks.items()},
        region_indices={k: v.copy() for k, v in spec.region_indices.items()},
        site_orders=site_orders,
        outer_paths=[list(path) for path in spec.outer_paths],
        branch_paths=[list(path) for path in spec.branch_paths],
    )


def build_region_ladder(region_name: str, region_mask, *, bond_sites, site_order=None) -> RegionLadderSpec:
    mask = np.asarray(region_mask, dtype=np.uint8).reshape(-1)
    if site_order is None:
        order = np.asarray(build_boundary_first_site_order(mask, bond_sites), dtype=np.int32)
    else:
        order = np.asarray(site_order, dtype=np.int32).reshape(-1)

    masks = [np.zeros_like(mask, dtype=np.uint8)]
    current = np.zeros_like(mask, dtype=np.uint8)
    for site in order:
        current = current.copy()
        current[int(site)] = 1
        masks.append(current)

    neighbors = []
    for idx in range(len(masks)):
        row = []
        if idx - 1 >= 0:
            row.append(idx - 1)
        if idx + 1 < len(masks):
            row.append(idx + 1)
        neighbors.append(row)

    return RegionLadderSpec(
        region_name=str(region_name),
        site_order=order,
        masks=masks,
        neighbors=neighbors,
        target_ensemble=len(masks) - 1,
    )


def build_kp_ladders(spec: KPRegionSpec, *, bond_sites) -> dict[str, RegionLadderSpec]:
    enriched = spec if spec.site_orders is not None else attach_kp_site_orders(spec, bond_sites)
    return {
        name: build_region_ladder(
            name,
            enriched.region_masks[name],
            bond_sites=bond_sites,
            site_order=enriched.site_orders[name],
        )
        for name in KP_REGION_NAMES
    }
