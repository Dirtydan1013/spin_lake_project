"""End-to-end smoke test for the MPI-aware KP TEE entry point.

When run as ``python ...`` this exercises the rank=1 path of the wrappers.
When run as ``mpiexec -n K python ...`` it exercises the multi-rank reduction
path inside ``run_ratio_mpi`` / ``run_expanded_mpi``.  Both must produce the
same set of artefacts and finite γ / S_2 values on rank 0.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.kp.kp_geometry import KP_REGION_NAMES
from src.mpi.kp_tee_job_mpi import run_expanded_job_mpi, run_ratio_job_mpi
from src.tee.compose_tee import load_kp_result_hdf5
from src.tee.reweighting import load_expanded_result_hdf5


# Smallest kagome lattice that admits an m=1 KP cut: 7×7 → 6 atoms per region.
# Tiny M, n_therm, n_measure keep the test fast (~seconds).
COMMON = dict(
    nx=7, ny=7, m=1, M=50, a=1.0,
    Omega=1.0, Rb=1.2, delta_min=0.0, delta_max=1.0,
    epsilon=0.01, seed=42, neighbor_cutoff=1, delta_groups=5,
    preferred_center_label=None,
)


def _test_run_ratio_job_mpi_writes_expected_artefacts():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    with tempfile.TemporaryDirectory() as tmpdir:
        # rank 0 owns tmpdir; broadcast its name so all ranks agree.
        tmpdir = comm.bcast(tmpdir, root=0)
        out_dir = Path(tmpdir) / "ratio"

        payload = run_ratio_job_mpi(
            **COMMON,
            n_therm=50, n_measure=200, measure_stride=1, block_size=50,
            output_dir=str(out_dir), comm=comm,
        )
        comm.Barrier()

        if rank == 0:
            assert payload is not None
            assert "gamma" in payload and np.isfinite(payload["gamma"])
            assert "gamma_err" in payload and np.isfinite(payload["gamma_err"])

            geometry_json = json.loads(Path(payload["geometry_path"]).read_text())
            assert geometry_json["params"]["method"] == "ratio"
            assert geometry_json["params"]["n_ranks"] == comm.Get_size()
            assert set(geometry_json["geometry"]["region_indices"]) == set(KP_REGION_NAMES)

            kp = load_kp_result_hdf5(payload["kp_result_path"])
            assert set(kp.region_summaries) == set(KP_REGION_NAMES)
            for name, summary in kp.region_summaries.items():
                assert summary.S_2 is not None and np.isfinite(summary.S_2)
                assert int(summary.sites_ordered.size) == int(geometry_json["geometry"]["region_sizes"][name])

            ratio_steps_dir = out_dir / "ratio_steps"
            for name in KP_REGION_NAMES:
                size = int(geometry_json["geometry"]["region_sizes"][name])
                for step in range(size):
                    p = ratio_steps_dir / f"{name}_step{step:03d}.h5"
                    assert p.exists(), f"missing per-step artefact {p}"


def _test_run_expanded_job_mpi_writes_expected_artefacts():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = comm.bcast(tmpdir, root=0)
        out_dir = Path(tmpdir) / "expanded"

        # Run a subset of regions to keep the test under a few seconds even at
        # rank=1.  All three atomic regions are independent; this is enough to
        # exercise the per-region MPI loop.
        regions = ["A", "B", "C"]
        payload = run_expanded_job_mpi(
            **COMMON,
            regions=regions,
            output_dir=str(out_dir),
            autotune_steps_per_iter=200, autotune_max_iters=3, autotune_tol=1.5,
            autotune_method="transition_matrix", autotune_damping=0.7,
            n_steps=1000, block_size=100,
            target_s2_err=None, batch_steps=None, max_steps=None, min_steps=0,
            estimator="collection",
            comm=comm,
        )
        comm.Barrier()

        if rank == 0:
            assert payload is not None
            assert "geometry_path" in payload
            geometry_json = json.loads(Path(payload["geometry_path"]).read_text())
            assert geometry_json["params"]["method"] == "expanded"
            assert geometry_json["params"]["n_ranks"] == comm.Get_size()
            for name in regions:
                entry = payload[name]
                assert np.isfinite(entry["s2"])
                assert entry["s2_err"] >= 0.0
                loaded = load_expanded_result_hdf5(entry["path"])
                assert loaded["params"]["region_name"] == name
                assert int(loaded["params"]["target_ensemble"]) == int(geometry_json["geometry"]["region_sizes"][name])
                assert loaded["result"].log_z.size == int(geometry_json["geometry"]["region_sizes"][name]) + 1


def main():
    _test_run_ratio_job_mpi_writes_expected_artefacts()
    _test_run_expanded_job_mpi_writes_expected_artefacts()
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"KP TEE-job MPI integration smoke test passed (n_ranks={MPI.COMM_WORLD.Get_size()})")


if __name__ == "__main__":
    main()
