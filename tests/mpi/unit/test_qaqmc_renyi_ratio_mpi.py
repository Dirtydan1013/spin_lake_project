import os
import sys
import tempfile

import numpy as np
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tee.qaqmc_renyi_ratio import load_ratio_result_hdf5
from src.mpi.qaqmc_renyi_ratio_mpi import run_ratio_mpi


class _FakeEngine:
    def __init__(self, ratio):
        self.visit_count_low = 10
        self.visit_count_high = int(round(float(ratio) * self.visit_count_low))
        self.pos = np.arange(4, dtype=np.float64).reshape(-1, 1)
        self.delta_schedule = np.linspace(0.0, 1.0, 8, dtype=np.float64)

    def set_topology_pair(self, lower_mask, upper_mask, diff_site):
        self.lower_mask = np.array(lower_mask, dtype=np.uint8)
        self.upper_mask = np.array(upper_mask, dtype=np.uint8)
        self.diff_site = int(diff_site)

    def get_visit_counts(self):
        return np.array([self.visit_count_low, self.visit_count_high], dtype=np.int64)

    def run_steps(self, n_steps):
        self.n_steps = int(n_steps)

    def reset_visit_counts(self):
        self.was_reset = True


def _test_single_rank_mpi_driver_with_fake_engine():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ratio_mpi.h5")
        result = run_ratio_mpi(
            N=4,
            M=4,
            A_mask=np.array([1, 0, 0, 0], dtype=np.uint8),
            next_site=2,
            n_therm=3,
            n_measure=10,
            filepath=path,
            region_name="A",
            step_index=1,
            comm=MPI.COMM_SELF,
            engine_factory=lambda rank: _FakeEngine(0.4),
        )
        assert np.isclose(result.ratio, 0.4)
        loaded = load_ratio_result_hdf5(path)
        assert loaded["visit_count_low_per_rank"].tolist() == [10]
        assert loaded["visit_count_high_per_rank"].tolist() == [4]
        assert loaded["pos"].shape == (4, 1)
        assert loaded["delta_schedule"].shape == (8,)


def main():
    _test_single_rank_mpi_driver_with_fake_engine()
    print("QAQMC Renyi ratio MPI unit checks passed")


if __name__ == "__main__":
    main()
