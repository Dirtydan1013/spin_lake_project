import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.engines.qaqmc_renyi import QAQMCRenyiRydberg


def _explicit_midpoint(types, sites, M, N):
    state = np.zeros(N, dtype=np.int32)
    for p in range(M):
        if types[p] == -1:
            state[sites[p]] ^= 1
    return state


def _check_midpoint_consistency(mask):
    engine = QAQMCRenyiRydberg(N=6, M=12, seed=11)
    engine.set_A_mask(np.asarray(mask, dtype=np.uint8))
    engine.set_indicator_site(0)
    engine.reset_indicator()
    engine.run_steps(20)

    for replica in (0, 1):
        types = engine.get_op_types(replica)
        sites = engine.get_op_sites(replica)
        explicit = _explicit_midpoint(types, sites, engine.M, engine.N)
        got = engine.get_state_at_M(replica)
        assert np.array_equal(got, explicit)

    for site in range(engine.N):
        paths = engine.get_site_paths(site)
        for key in ("channel_0", "channel_1"):
            assert paths[key][0] == 0
            assert paths[key][-1] == 0


def main():
    _check_midpoint_consistency(np.zeros(6, dtype=np.uint8))
    joined = np.zeros(6, dtype=np.uint8)
    joined[[1, 3]] = 1
    _check_midpoint_consistency(joined)
    print("Renyi QAQMC integration checks passed")


if __name__ == "__main__":
    main()
