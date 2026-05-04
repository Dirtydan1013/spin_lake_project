import os
import sys
import tempfile

import numpy as np
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tee.qaqmc_renyi_ratio import load_ratio_result_hdf5
from src.mpi.qaqmc_renyi_ratio_mpi import run_ratio_mpi


def _test_single_rank_real_ratio_mpi():
    pos = np.arange(6, dtype=np.float64).reshape(-1, 1)
    a_mask = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ratio_real_mpi.h5")
        result = run_ratio_mpi(
            N=6,
            M=8,
            Omega=1.0,
            Rb=1.2,
            delta_min=0.0,
            delta_max=1.0,
            pos=pos,
            seed=19,
            neighbor_cutoff=1,
            A_mask=a_mask,
            next_site=2,
            n_therm=2,
            n_measure=8,
            filepath=path,
            region_name="A",
            step_index=0,
            comm=MPI.COMM_SELF,
        )
        loaded = load_ratio_result_hdf5(path)
        assert np.isclose(loaded["result"].ratio, result.ratio)
        assert loaded["visit_count_low_per_rank"].tolist() == [result.visit_count_low]
        assert loaded["visit_count_high_per_rank"].tolist() == [result.visit_count_high]
        assert loaded["A_mask"].tolist() == a_mask.tolist()
        assert loaded["A_kp1_mask"].tolist() == [0, 1, 1, 0, 0, 0]
        assert loaded["next_site"] == 2
        assert loaded["delta_schedule"].shape == (16,)


def main():
    _test_single_rank_real_ratio_mpi()
    print("QAQMC Renyi ratio MPI integration checks passed")


if __name__ == "__main__":
    main()
