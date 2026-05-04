import os
import sys
import tempfile

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tee.compose_tee import aggregate_ratios, compose_kp


def _write_ratio_file(path, region_name, step_index, next_site, ratio, ratio_err):
    with h5py.File(path, "w") as h5:
        params = h5.create_group("params")
        params.attrs["region_name"] = region_name
        params.attrs["step_index"] = int(step_index)
        ratio_group = h5.create_group("ratio")
        ratio_group.create_dataset("next_site", data=int(next_site))
        ratio_group.create_dataset("ratio", data=float(ratio))
        ratio_group.create_dataset("ratio_err", data=float(ratio_err))


def _test_hdf5_aggregation_and_kp():
    with tempfile.TemporaryDirectory() as tmpdir:
        specs = [
            ("A", 0, 1, 0.5, 0.05),
            ("B", 0, 2, 0.5, 0.05),
            ("C", 0, 3, 0.5, 0.05),
            ("AB", 0, 1, 0.25, 0.025),
            ("AB", 1, 2, 0.25, 0.025),
            ("BC", 0, 2, 0.25, 0.025),
            ("BC", 1, 3, 0.25, 0.025),
            ("CA", 0, 1, 0.25, 0.025),
            ("CA", 1, 3, 0.25, 0.025),
            ("ABC", 0, 1, 0.125, 0.0125),
            ("ABC", 1, 2, 0.125, 0.0125),
            ("ABC", 2, 3, 0.125, 0.0125),
        ]
        paths = []
        for idx, spec in enumerate(specs):
            path = os.path.join(tmpdir, f"ratio_{idx}.h5")
            _write_ratio_file(path, *spec)
            paths.append(path)

        summaries = aggregate_ratios(paths)
        assert summaries["AB"].sites_ordered.tolist() == [1, 2]
        assert summaries["ABC"].ratios.tolist() == [0.125, 0.125, 0.125]
        gamma, gamma_err = compose_kp(summaries)
        assert np.isfinite(gamma)
        assert np.isfinite(gamma_err)


def main():
    _test_hdf5_aggregation_and_kp()
    print("Compose TEE integration checks passed")


if __name__ == "__main__":
    main()
