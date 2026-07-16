"""Production runner for one device-resident CUDA QAQMC chain.

The process is intentionally independent: Slurm can launch one process per
GPU without requiring CUDA-aware MPI.  Each process writes a rank-local HDF5
stream and an atomic operator/RNG checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np

import qaqmc_cuda
from src.engines.qaqmc_cuda import QAQMC_Rydberg_CUDA
from src.mpi.qaqmc_mpi import (
    _build_occ2_sf_geometry,
    _build_occ2_sf_geometry_tri,
    _build_occ_q_grid,
    _build_occ_sf_geometry,
    _build_occ_sf_geometry_tri,
    _build_vbs_triangles,
    _build_vbs_triangles_tri,
    _lattice_observables,
    _snap_delta_points,
)
from src.rydberg.lattices import (
    generate_kagome_bond_lattice,
    generate_kagome_bond_triangle_lattice,
    lattice_box_vectors,
)


def _append(dataset: h5py.Dataset, value: np.ndarray) -> None:
    row = len(dataset)
    dataset.resize(row + 1, axis=0)
    dataset[row] = value


def _dataset(group: h5py.Group, name: str, tail: tuple[int, ...]) -> h5py.Dataset:
    if name in group:
        return group[name]
    chunk = (1,) + tuple(max(1, size) for size in tail)
    return group.create_dataset(
        name,
        shape=(0,) + tail,
        maxshape=(None,) + tail,
        chunks=chunk,
        dtype=np.float64,
    )


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def _validate_or_create_manifest(path: Path, configuration: dict[str, object]) -> None:
    payload = json.dumps(configuration, sort_keys=True, default=_json_default)
    signature = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    record = json.dumps(
        {"sha256": signature, "configuration": configuration},
        sort_keys=True, default=_json_default, indent=2,
    )
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("sha256") != signature:
            raise RuntimeError(
                "run parameters/site permutation differ from the existing CUDA checkpoint"
            )
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(record, encoding="utf-8")
    os.replace(temporary, path)


def _write_bin(
    path: Path,
    sample_count: int,
    mean: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> None:
    with h5py.File(path, "a") as handle:
        if "params" not in handle:
            params = handle.create_group("params")
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, np.integer, np.floating)):
                    params.attrs[key] = value
            params.attrs["metadata_json"] = json.dumps(
                metadata, sort_keys=True, default=_json_default
            )
        bins = handle.require_group("bins")
        if "sample_count" not in bins:
            bins.create_dataset(
                "sample_count", shape=(0,), maxshape=(None,), chunks=(128,),
                dtype=np.int64,
            )
        _append(bins["sample_count"], np.asarray(sample_count, dtype=np.int64))
        for key, value in mean.items():
            value = np.asarray(value, dtype=np.float64)
            _append(_dataset(bins, key, value.shape), value)
        handle.flush()


def _completed_samples(path: Path) -> int:
    if not path.exists():
        return 0
    with h5py.File(path, "r") as handle:
        if "bins/sample_count" not in handle:
            return 0
        return int(np.asarray(handle["bins/sample_count"]).sum())


def _truncate_to_checkpoint(path: Path, target_samples: int) -> int:
    """Drop an HDF5 bin flushed after the last atomic GPU checkpoint."""
    if not path.exists():
        return 0
    with h5py.File(path, "r+") as handle:
        if "bins/sample_count" not in handle:
            return 0
        bins = handle["bins"]
        counts = np.asarray(bins["sample_count"], dtype=np.int64)
        cumulative = np.concatenate(([0], np.cumsum(counts)))
        matches = np.flatnonzero(cumulative == target_samples)
        if not len(matches):
            raise RuntimeError(
                "checkpoint sweep does not align with an HDF5 bin boundary"
            )
        keep = int(matches[-1])
        for dataset in bins.values():
            dataset.resize(keep, axis=0)
        handle.flush()
        return target_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lattice", choices=["kagome_bond", "kagome_bond_triangle"],
                        default="kagome_bond")
    parser.add_argument("--boundary", choices=["open", "periodic"], default="periodic")
    parser.add_argument("--nx", type=int, default=6)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--a", type=float, default=4.0)
    parser.add_argument("--M", type=int, default=2_760_000)
    parser.add_argument("--Omega", type=float, default=1.0)
    parser.add_argument("--Rb", type=float, default=2.4)
    parser.add_argument("--delta-min", type=float, default=-2.0)
    parser.add_argument("--delta-max", type=float, default=6.0)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--neighbor-cutoff", type=int, default=-1)
    parser.add_argument("--delta-groups", type=int, default=600)
    parser.add_argument("--profile-step", type=int, default=10_000)
    parser.add_argument("--n-equil", type=int, default=4_000)
    parser.add_argument("--n-samples", type=int, default=100_000)
    parser.add_argument("--checkpoint", type=int, default=200,
                        help="raw samples per output bin/checkpoint")
    parser.add_argument("--occ-sf-grid-n", type=int, default=0,
                        help="BZ q-grid side; 0 disables occupation-SF matrices")
    parser.add_argument("--occ-sf-deltas", type=float, nargs="*", default=[],
                        help="forward-ramp deltas for occupation-SF matrices")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--device", type=int, default=0,
                        help="index within CUDA_VISIBLE_DEVICES")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--no-permute-site-labels", action="store_true")
    args = parser.parse_args()

    rank = int(os.environ.get("SLURM_PROCID", "0")) if args.rank is None else args.rank
    local_rank = int(os.environ.get("SLURM_LOCALID", str(args.device)))
    device = 0 if "CUDA_VISIBLE_DEVICES" in os.environ else local_rank
    args.run_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.run_dir / f"rank{rank}.h5"
    checkpoint_path = args.run_dir / f"rank{rank}.checkpoint.npz"
    manifest_path = args.run_dir / f"rank{rank}.manifest.json"

    if args.lattice == "kagome_bond":
        canonical_pos = generate_kagome_bond_lattice(args.nx, args.ny, args.a)
    else:
        if args.boundary == "periodic":
            raise ValueError("kagome_bond_triangle only supports open boundaries")
        canonical_pos = generate_kagome_bond_triangle_lattice(args.nx, args.ny, args.a)
    n_sites = len(canonical_pos)
    box = (lattice_box_vectors(args.lattice, args.nx, args.ny, args.a, N=n_sites)
           if args.boundary == "periodic" else None)

    (bulk, loop_sets, string_sets, loop_meta, string_meta,
     vertex_sets, ijk_map) = _lattice_observables(
        args.lattice, args.nx, args.ny, boundary=args.boundary
    )
    rng = np.random.default_rng(args.seed + rank * 1_000_003)
    if args.no_permute_site_labels:
        engine_to_canonical = np.arange(n_sites, dtype=np.int32)
    else:
        engine_to_canonical = rng.permutation(n_sites).astype(np.int32)
    canonical_to_engine = np.empty(n_sites, dtype=np.int32)
    canonical_to_engine[engine_to_canonical] = np.arange(n_sites, dtype=np.int32)
    positions = canonical_pos[engine_to_canonical]
    remap = lambda values: canonical_to_engine[np.asarray(values, dtype=np.int32)]

    chain_seed = args.seed + rank * 1_000_003
    manifest_configuration: dict[str, object] = {
        "format_version": 1,
        "rank": rank, "N": n_sites, "M": args.M, "Omega": args.Omega,
        "Rb": args.Rb, "delta_min": args.delta_min, "delta_max": args.delta_max,
        "epsilon": args.epsilon, "neighbor_cutoff": args.neighbor_cutoff,
        "delta_groups": args.delta_groups, "profile_step": args.profile_step,
        "occ_sf_grid_n": args.occ_sf_grid_n,
        "occ_sf_deltas": list(args.occ_sf_deltas),
        "n_equil": args.n_equil, "seed": chain_seed,
        "lattice": args.lattice, "boundary": args.boundary,
        "nx": args.nx, "ny": args.ny, "a": args.a,
        "positions_sha256": hashlib.sha256(
            np.ascontiguousarray(positions, dtype=np.float64).tobytes()
        ).hexdigest(),
        "engine_to_canonical": engine_to_canonical.tolist(),
    }
    if (output_path.exists() or checkpoint_path.exists()) and not manifest_path.exists():
        raise RuntimeError("existing CUDA output/checkpoint has no parameter manifest")
    _validate_or_create_manifest(manifest_path, manifest_configuration)
    t0 = time.perf_counter()
    simulation = QAQMC_Rydberg_CUDA(
        n_sites, args.M, Omega=args.Omega, Rb=args.Rb,
        delta_min=args.delta_min, delta_max=args.delta_max,
        pos=positions, epsilon=args.epsilon, seed=chain_seed,
        neighbor_cutoff=args.neighbor_cutoff, delta_groups=args.delta_groups,
        box_vectors=box, device=device, verbose=True,
    )
    engine = simulation.engine
    engine.set_bulk_sites(remap(bulk))
    engine.set_observable_sites(
        [remap(values) for values in loop_sets + vertex_sets],
        [remap(values) for values in string_sets],
    )
    vbs = (_build_vbs_triangles_tri(args.nx, args.ny, ijk_map)
           if args.lattice == "kagome_bond_triangle"
           else _build_vbs_triangles(args.nx, args.ny))
    if vbs is not None:
        corners, parity, vbs_sign, ss_sign, ref00, ref10 = vbs
        engine.set_vbs_triangles(
            remap(corners).reshape(-1, 3), parity, vbs_sign, ss_sign, ref00, ref10
        )

    completed = _completed_samples(output_path)
    if checkpoint_path.exists():
        engine.load_checkpoint(checkpoint_path)
        checkpoint_samples = engine.sweep_id - args.n_equil
        if checkpoint_samples < 0:
            raise RuntimeError(
                f"checkpoint sweep {engine.sweep_id} precedes requested equilibration"
            )
        if checkpoint_samples < completed:
            completed = _truncate_to_checkpoint(output_path, checkpoint_samples)
            print(f"[rank {rank}] discarded an uncommitted trailing HDF5 bin", flush=True)
        elif checkpoint_samples > completed:
            raise RuntimeError(
                "checkpoint is newer than HDF5 output; restore both files from "
                "the same filesystem snapshot"
            )
        print(f"[rank {rank}] resumed at sample {completed}/{args.n_samples}", flush=True)
    elif completed:
        raise RuntimeError("rank HDF5 exists without its CUDA checkpoint")
    else:
        engine.run_steps(args.n_equil)
        engine.save_checkpoint(checkpoint_path)
        print(f"[rank {rank}] equilibration complete", flush=True)

    n_points = engine.M_total // args.profile_step
    profile_p = (np.arange(n_points, dtype=np.int64) + 1) * args.profile_step - 1
    profile_delta = np.where(
        profile_p < args.M,
        args.delta_min + (args.delta_max - args.delta_min) * profile_p / args.M,
        args.delta_max - (args.delta_max - args.delta_min) * (profile_p - args.M) / args.M,
    )
    occ_config = None
    if args.occ_sf_grid_n > 0 and args.occ_sf_deltas:
        occ_indices = _snap_delta_points(profile_delta, args.occ_sf_deltas, "forward")
        q_points, q_fractional = _build_occ_q_grid(args.occ_sf_grid_n, args.a)
        if args.lattice == "kagome_bond_triangle":
            cell_R, basis, in_bulk = _build_occ_sf_geometry_tri(
                args.nx, args.ny, args.a, ijk_map, bulk
            )
            cell_R2, basis2 = _build_occ2_sf_geometry_tri(
                args.nx, args.ny, args.a, ijk_map
            )
        else:
            cell_R, basis, in_bulk = _build_occ_sf_geometry(
                args.nx, args.ny, args.a, args.boundary
            )
            cell_R2, basis2 = _build_occ2_sf_geometry(args.nx, args.ny, args.a)
        occ_config = {
            "indices": occ_indices,
            "q_points": q_points,
            "q_fractional": q_fractional,
            "cell_R": np.asarray(cell_R)[engine_to_canonical],
            "basis": np.asarray(basis)[engine_to_canonical],
            "in_bulk": np.asarray(in_bulk)[engine_to_canonical],
            "cell_R2": np.asarray(cell_R2)[engine_to_canonical],
            "basis2": np.asarray(basis2)[engine_to_canonical],
        }
    metadata: dict[str, object] = {
        "rank": rank, "N": n_sites, "M": args.M, "Omega": args.Omega,
        "Rb": args.Rb, "delta_min": args.delta_min, "delta_max": args.delta_max,
        "epsilon": args.epsilon, "profile_step": args.profile_step,
        "n_equil": args.n_equil, "seed": chain_seed,
        "lattice": args.lattice, "boundary": args.boundary,
        "nx": args.nx, "ny": args.ny,
        "loop_meta": loop_meta, "string_meta": string_meta,
        "n_vertex_copies": len(vertex_sets),
        "device": qaqmc_cuda.device_info()[device],
        "profile_p": profile_p.tolist(), "profile_delta": profile_delta.tolist(),
        "engine_to_canonical": engine_to_canonical.tolist(),
        "occ_sf_grid_n": args.occ_sf_grid_n,
        "occ_sf_profile_indices": ([] if occ_config is None
                                   else occ_config["indices"].tolist()),
        "occ_sf_q_fractional": ([] if occ_config is None
                                else occ_config["q_fractional"].tolist()),
    }

    checkpoint_every = max(1, args.checkpoint)
    while completed < args.n_samples:
        count = min(checkpoint_every, args.n_samples - completed)
        accum: dict[str, np.ndarray] = {}
        bin_start = time.perf_counter()
        for _ in range(count):
            engine.mc_step()
            states = engine.profile_states(args.profile_step)
            measured = engine.measure_states(states)
            if occ_config is not None:
                selected_states = states[occ_config["indices"]]
                sf = engine.occupation_sf_matrices(
                    selected_states, occ_config["cell_R"], occ_config["basis"],
                    occ_config["q_points"], occ_config["in_bulk"], n_basis=6,
                )
                measured["occ_S_full_re"] = sf["S_full"].real
                measured["occ_S_full_im"] = sf["S_full"].imag
                measured["occ_S_bulk_re"] = sf["S_bulk"].real
                measured["occ_S_bulk_im"] = sf["S_bulk"].imag
                measured["occ_nprof"] = selected_states
                sf2 = engine.occupation_sf_matrices(
                    selected_states, occ_config["cell_R2"], occ_config["basis2"],
                    occ_config["q_points"], occ_config["basis2"] >= 0, n_basis=6,
                )
                measured["occ_S_tri_re"] = sf2["S_full"].real
                measured["occ_S_tri_im"] = sf2["S_full"].imag
            for key, value in measured.items():
                value = np.asarray(value, dtype=np.float64)
                if value.size == 0:
                    continue
                if key not in accum:
                    accum[key] = np.zeros_like(value)
                accum[key] += value
        mean = {key: value / count for key, value in accum.items()}
        _write_bin(output_path, count, mean, metadata)
        engine.save_checkpoint(checkpoint_path)
        completed += count
        elapsed = time.perf_counter() - bin_start
        print(
            f"[rank {rank}] samples={completed}/{args.n_samples} "
            f"bin_s={elapsed:.3f} step_ms={1000.0 * elapsed / count:.3f} "
            f"resident_mib={engine.device_bytes / 2**20:.1f}",
            flush=True,
        )

    print(
        f"[rank {rank}] finished in {time.perf_counter() - t0:.1f}s: {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
