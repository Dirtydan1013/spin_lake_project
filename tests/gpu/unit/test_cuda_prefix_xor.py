"""White-box tests for the CUDA packed prefix-XOR state propagation."""

from __future__ import annotations

import numpy as np
import pytest

qaqmc_cuda = pytest.importorskip("qaqmc_cuda")
if not qaqmc_cuda.is_available():
    pytest.skip("CUDA device is not available", allow_module_level=True)


def _cpu_prefix(op_types: np.ndarray, op_sites: np.ndarray, n_sites: int) -> np.ndarray:
    words = (n_sites + 63) // 64
    out = np.zeros((len(op_types), words), dtype=np.uint64)
    state = np.zeros(words, dtype=np.uint64)
    one = np.uint64(1)
    for p, (op_type, site) in enumerate(zip(op_types, op_sites, strict=True)):
        out[p] = state
        if op_type == -1:
            word = int(site) // 64
            bit = int(site) % 64
            state[word] ^= one << np.uint64(bit)
    return out


@pytest.mark.parametrize("n_sites", [1, 64, 65, 216, 384])
@pytest.mark.parametrize("length", [1, 255, 256, 257, 1025])
def test_prefix_xor_matches_cpu_across_tile_and_word_boundaries(n_sites, length):
    rng = np.random.default_rng(90210 + n_sites + length)
    op_types = rng.choice(np.array([-1, 1, 2], np.int32), size=length,
                          p=[0.35, 0.35, 0.30]).astype(np.int32)
    op_sites = rng.integers(0, n_sites, size=length, dtype=np.int32)

    expected = _cpu_prefix(op_types, op_sites, n_sites)
    actual = qaqmc_cuda.prefix_xor_states(op_types, op_sites, n_sites)

    assert actual.dtype == np.uint64
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


def test_prefix_xor_all_diagonal_is_zero():
    op_types = np.tile(np.array([1, 2], np.int32), 300)
    op_sites = np.arange(len(op_types), dtype=np.int32) % 7
    actual = qaqmc_cuda.prefix_xor_states(op_types, op_sites, 7)
    np.testing.assert_array_equal(actual, np.zeros_like(actual))


@pytest.mark.parametrize(
    "types,sites,n_sites,message",
    [
        (np.array([0], np.int32), np.array([0], np.int32), 2, "-1, 1, or 2"),
        (np.array([-1], np.int32), np.array([2], np.int32), 2, "out of range"),
        (np.array([-1], np.int32), np.array([-1], np.int32), 2, "out of range"),
    ],
)
def test_prefix_xor_rejects_invalid_operator_strings(types, sites, n_sites, message):
    with pytest.raises(ValueError, match=message):
        qaqmc_cuda.prefix_xor_states(types, sites, n_sites)


def test_prefix_xor_is_repeatable():
    rng = np.random.default_rng(71)
    types = rng.choice(np.array([-1, 1, 2], np.int32), 777)
    sites = rng.integers(0, 216, 777, dtype=np.int32)
    first = qaqmc_cuda.prefix_xor_states(types, sites, 216)
    second = qaqmc_cuda.prefix_xor_states(types, sites, 216)
    np.testing.assert_array_equal(first, second)
