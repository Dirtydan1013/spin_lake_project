"""Driver-level plumbing test for the drag-ladder phase of
src.mpi.qaqmc_string_work_mpi (single in-process rank).

Statistical correctness of the estimators is gated at engine level in
tests/engines/integration/test_qaqmc_string_drag_vs_ed.py; here we check the
driver wiring: phase sequencing, per-pass aggregation, HDF5 layout, and the
anchor x ladder curve composition.
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.mpi.qaqmc_string_work_mpi import run_string_work_mpi
from src.rydberg.lattices import generate_1d_chain


def test_string_drag_driver_single_rank():
    import h5py

    N, M = 5, 16
    pos = np.asarray(generate_1d_chain(N), dtype=np.float64)
    grid = [12, 8]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "drag_mpi.h5")
        out = run_string_work_mpi(
            N=N, M=M, Omega=1.0, Rb=1.2, delta_min=0.0, delta_max=1.5,
            epsilon=0.01, pos=pos, string_sites=[2],
            K_values=[50], schedule="cosine",
            n_trajectories=64, n_thermalize=200, decorrelation_steps=20,
            neighbor_cutoff=1, delta_groups=32, seed=11,
            filepath=path,
            checkpoint_every_trajectories=32, checkpoint_dir=os.path.join(tmpdir, "ck"),
            drag_grid=grid, drag_mirror=True,
            drag_samples_per_rung=40, drag_sweeps_between_samples=1,
            drag_burn_per_rung=2, drag_slots_per_rung=2,
            drag_repeats=2, drag_thermalize=100, drag_equil_at_anchor=20,
            verbose=False,
        )
        drag = out["drag"]
        assert drag is not None
        assert list(drag["m_grid"]) == grid
        assert drag["n_passes"] == 2
        assert drag["log_r_passes"].shape == (2, 2)
        assert np.all(np.isfinite(drag["log_r_mean"]))
        assert np.all(drag["log_r_sem"] > 0)
        np.testing.assert_allclose(
            drag["deltas"], [0.0 + 1.5 * m / M for m in grid])
        # curve composition: log_o_curve == anchor log_o_c + ladder log_r
        res_k = out["K_results"][50]
        cur = drag["curves"][50]
        np.testing.assert_allclose(
            cur["log_o_curve"], res_k["log_o_c"] + drag["log_r_mean"])
        assert np.all(cur["log_o_curve_sem"] >= drag["log_r_sem"])

        # per-pass drag chunks exist (crash-safety flushes)
        drag_rank_file = os.path.join(tmpdir, "ck", "drag", "rank0.h5")
        assert os.path.exists(drag_rank_file)
        with h5py.File(drag_rank_file, "r") as f:
            assert "chunk0" in f and "chunk1" in f
            assert f["chunk0"]["log_r"].shape == (2,)

        # HDF5 layout
        with h5py.File(path, "r") as f:
            dg = f["drag"]
            assert bool(dg.attrs["mirror"]) is True
            assert int(dg.attrs["slots_per_rung"]) == 2
            assert dg["log_r_left_passes"].shape == (2, 2)
            kg = f["drag"]["curves"]["K50"]
            np.testing.assert_allclose(kg["o_curve"][...],
                                       np.exp(cur["log_o_curve"]))
            assert "log_o_c_sem_boot" in f["K_results"]["K50"].attrs


def test_string_driver_without_drag_unchanged():
    # No drag flags -> payload absent and the legacy result dict intact.
    N, M = 5, 8
    pos = np.asarray(generate_1d_chain(N), dtype=np.float64)
    out = run_string_work_mpi(
        N=N, M=M, Omega=1.0, Rb=1.2, delta_min=0.0, delta_max=1.5,
        epsilon=0.01, pos=pos, string_sites=[2],
        K_values=[20], schedule="cosine",
        n_trajectories=16, n_thermalize=50, decorrelation_steps=5,
        neighbor_cutoff=1, delta_groups=16, seed=3, verbose=False,
    )
    assert out["drag"] is None
    assert 20 in out["K_results"]


def main():
    test_string_drag_driver_single_rank()
    print("drag driver single-rank plumbing passed")
    test_string_driver_without_drag_unchanged()
    print("no-drag legacy path passed")


if __name__ == "__main__":
    main()
