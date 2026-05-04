"""
Probe: does setting A_mask = {0} actually enforce sigma^0_0(M) == sigma^1_0(M)?
If the SWAP boundary is active, the indicator at site 0 should always be 1
(i.e. replicas match on site 0 every sample).
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.rydberg.lattices import generate_1d_chain
from src.engines.qaqmc_renyi import QAQMCRenyiRydberg


def main():
    N = 4
    pos = generate_1d_chain(N)

    for mask_desc, mask_list in [
        ("A=empty (no SWAP)", [0, 0, 0, 0]),
        ("A={0}  (SWAP on site 0)", [1, 0, 0, 0]),
        ("A={0,1}", [1, 1, 0, 0]),
    ]:
        engine = QAQMCRenyiRydberg(
            N=N, M=100, Omega=1.0, Rb=1.2,
            delta_min=0.0, delta_max=1.5, pos=pos, seed=1234, neighbor_cutoff=1,
        )
        mask = np.array(mask_list, dtype=np.uint8)
        engine.set_A_mask(mask)
        engine.set_indicator_site(0)
        engine.run_steps(500)  # thermalize
        engine.reset_indicator()
        engine.run_steps(5000)
        indicator_avg = engine.get_indicator_avg()
        match_frac_site0 = indicator_avg
        print(f"{mask_desc}: P(match at site 0) = {match_frac_site0:.4f}")

        # Also check site 1 match
        engine.set_indicator_site(1)
        engine.reset_indicator()
        engine.run_steps(5000)
        m1 = engine.get_indicator_avg()
        print(f"    P(match at site 1) = {m1:.4f}")

        # Dump actual midpoint states from a few samples
        engine.set_indicator_site(0)
        engine.reset_indicator()
        states0 = []
        states1 = []
        for _ in range(20):
            engine.run_steps(10)
            states0.append(engine.get_state_at_M(0).tolist())
            states1.append(engine.get_state_at_M(1).tolist())
        print(f"    sample replica-0 state_at_M: {states0[:5]}")
        print(f"    sample replica-1 state_at_M: {states1[:5]}")
        matches0 = sum(1 for s0, s1 in zip(states0, states1) if s0[0] == s1[0])
        matches1 = sum(1 for s0, s1 in zip(states0, states1) if s0[1] == s1[1])
        print(f"    out of 20 spot-check samples: match site 0 = {matches0}, match site 1 = {matches1}")
        print()


if __name__ == "__main__":
    main()
