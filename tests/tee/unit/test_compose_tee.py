import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tee.compose_tee import compose_kp, summarize_region


def _test_region_summary_formula():
    summary = summarize_region(
        "A",
        sites_ordered=[1, 2, 3],
        ratios=[0.5, 0.25, 0.8],
        ratio_errs=[0.05, 0.01, 0.08],
    )
    expected_s2 = -(np.log(0.5) + np.log(0.25) + np.log(0.8))
    expected_err = np.sqrt((0.05 / 0.5) ** 2 + (0.01 / 0.25) ** 2 + (0.08 / 0.8) ** 2)
    assert np.isclose(summary.S_2, expected_s2)
    assert np.isclose(summary.S_2_err, expected_err)


def _test_compose_kp_formula():
    summaries = {
        "A": summarize_region("A", [0], [0.5], [0.05]),
        "B": summarize_region("B", [0], [0.5], [0.05]),
        "C": summarize_region("C", [0], [0.5], [0.05]),
        "AB": summarize_region("AB", [0], [0.25], [0.025]),
        "BC": summarize_region("BC", [0], [0.25], [0.025]),
        "CA": summarize_region("CA", [0], [0.25], [0.025]),
        "ABC": summarize_region("ABC", [0], [0.125], [0.0125]),
    }
    gamma, gamma_err = compose_kp(summaries)
    expected_gamma = (
        summaries["A"].S_2 + summaries["B"].S_2 + summaries["C"].S_2
        - summaries["AB"].S_2 - summaries["BC"].S_2 - summaries["CA"].S_2
        + summaries["ABC"].S_2
    )
    expected_err = np.sqrt(sum(item.S_2_err ** 2 for item in summaries.values()))
    assert np.isclose(gamma, expected_gamma)
    assert np.isclose(gamma_err, expected_err)


def main():
    _test_region_summary_formula()
    _test_compose_kp_formula()
    print("Compose TEE unit checks passed")


if __name__ == "__main__":
    main()
