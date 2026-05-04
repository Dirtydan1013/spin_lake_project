import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tee.compose_tee import load_kp_result_hdf5, save_kp_result_hdf5
from src.rydberg.hamiltonian import build_rydberg_vij
from src.rydberg.lattices import generate_1d_chain
from src.tee.qaqmc_renyi_ratio import KPRatioRunner, RegionRatioRunner


def _test_real_kp_runner_small_chain():
    pos = generate_1d_chain(6)
    _, _, _, _, bond_sites, _ = build_rydberg_vij(
        N=6, Omega=1.0, Rb=1.2, pos=pos, verbose=False, neighbor_cutoff=1
    )
    region_masks = {
        "A": np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8),
        "B": np.array([0, 0, 1, 0, 0, 0], dtype=np.uint8),
        "C": np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8),
        "AB": np.array([0, 1, 1, 0, 0, 0], dtype=np.uint8),
        "BC": np.array([0, 0, 1, 1, 0, 0], dtype=np.uint8),
        "CA": np.array([0, 1, 0, 1, 0, 0], dtype=np.uint8),
        "ABC": np.array([0, 1, 1, 1, 0, 0], dtype=np.uint8),
    }
    region_runner = RegionRatioRunner(
        N=6,
        M=8,
        Omega=1.0,
        Rb=1.2,
        delta_min=0.0,
        delta_max=1.0,
        pos=pos,
        seed=31,
        neighbor_cutoff=1,
    )
    kp_runner = KPRatioRunner(region_runner=region_runner)
    result = kp_runner.run_kp(region_masks, bond_sites=bond_sites, n_therm=2, n_measure=6)
    assert set(result.kp_result.region_summaries) == set(region_masks)
    assert np.isfinite(result.kp_result.gamma)
    assert result.kp_result.gamma_err >= 0.0

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "kp_result.h5")
        save_kp_result_hdf5(path, result.kp_result)
        loaded = load_kp_result_hdf5(path)
        assert np.isclose(loaded.gamma, result.kp_result.gamma)
        assert np.isclose(loaded.gamma_err, result.kp_result.gamma_err)


def main():
    _test_real_kp_runner_small_chain()
    print("QAQMC Renyi KP runner integration checks passed")


if __name__ == "__main__":
    main()
