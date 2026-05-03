"""Shared helpers for the KP TEE MPI entry points.

Both ``kp_tee_ratio_mpi`` and ``kp_tee_expanded_mpi`` reuse the same lattice /
Hamiltonian CLI surface, the same geometry-payload writer, and the same
per-rank vs total flag resolver.  They live here so neither method-specific
entry point grows accidental coupling to the other.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.kp.kp_geometry import DEFAULT_LATTICE, LATTICE_NAMES


def _geometry_payload(spec) -> dict:
    return {
        "center_label": str(spec.center_label),
        "region_indices": {
            name: np.asarray(values, dtype=np.int32).reshape(-1).tolist()
            for name, values in spec.region_indices.items()
        },
        "region_sizes": {
            name: int(np.sum(np.asarray(mask, dtype=np.uint8)))
            for name, mask in spec.region_masks.items()
        },
        "site_orders": None if spec.site_orders is None else {
            name: np.asarray(values, dtype=np.int32).reshape(-1).tolist()
            for name, values in spec.site_orders.items()
        },
        "outer_paths": [[str(item) for item in path] for path in spec.outer_paths],
        "branch_paths": [[str(item) for item in path] for path in spec.branch_paths],
    }


def _write_geometry_json(path, *, spec, params: dict) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"params": dict(params), "geometry": _geometry_payload(spec)},
                   indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return out_path


def _resolve_total_per_rank(per_rank: int, total: int, n_ranks: int, *,
                            name: str, sentinel: int = -1) -> int:
    """Resolve a per-rank vs total count flag.

    - If only ``per_rank`` is set (``> sentinel``) it is used as-is.
    - If only ``total`` is set, returns ``ceil(total / n_ranks)``.
    - If both are set (both > sentinel), raises.
    - If neither, returns ``sentinel`` so downstream can treat as "not provided".
    """
    has_per_rank = int(per_rank) > sentinel
    has_total = int(total) > sentinel
    if has_per_rank and has_total:
        raise ValueError(
            f"specify either --{name} (per rank) or --{name}_total (sum across ranks), not both"
        )
    if has_total:
        if int(n_ranks) <= 0:
            raise ValueError("n_ranks must be positive")
        return -(-int(total) // int(n_ranks))  # ceil div
    return int(per_rank)


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add CLI flags shared by every KP TEE entry point.

    Lattice geometry, Hamiltonian, projector depth, RNG seed, output dir,
    and the universal ``--block_size`` knob.
    """
    parser.add_argument("--lattice", choices=list(LATTICE_NAMES), default=DEFAULT_LATTICE)
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--Omega", type=float, default=1.0)
    parser.add_argument("--Rb", type=float, default=1.2)
    parser.add_argument("--delta_min", type=float, default=0.0)
    parser.add_argument("--delta_max", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neighbor_cutoff", type=int, default=-1)
    parser.add_argument("--delta_groups", type=int, default=0)
    parser.add_argument("--preferred_center_label", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--block_size", type=int, default=-1)
    return parser


def normalize_common_args(args: argparse.Namespace) -> dict:
    """Convert sentinel values from ``add_common_args`` into Python-native form."""
    neighbor_cutoff = None if int(args.neighbor_cutoff) < 0 else int(args.neighbor_cutoff)
    block_size = None if int(args.block_size) < 0 else int(args.block_size)
    preferred_str = str(args.preferred_center_label).strip().lower()
    preferred = None if preferred_str in {"", "auto", "none"} else args.preferred_center_label
    return {
        "neighbor_cutoff": neighbor_cutoff,
        "block_size": block_size,
        "preferred_center_label": preferred,
    }
