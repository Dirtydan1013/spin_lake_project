"""Transactional HDF5 continuation tests for trajectory engines."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from src.mpi.chunk_io import (
    RankChunkWriter,
    array_fingerprint,
    checkpoint_tree_has_committed_chunks,
    collective_resume_decision,
    compact_operator_checkpoint,
    load_checkpointed_rank_chunks,
    rank_file,
)


_DATASETS = ("work", "attempts")


def test_array_fingerprint_binds_dtype_shape_and_content():
    values = np.arange(6, dtype=np.int32)
    assert array_fingerprint(values) == array_fingerprint(values.copy())
    assert array_fingerprint(values) != array_fingerprint(values.reshape(2, 3))
    assert array_fingerprint(values) != array_fingerprint(values.astype(np.int64))
    assert array_fingerprint(values) != array_fingerprint(values + 1)
    assert array_fingerprint(None) == "none"


@pytest.mark.parametrize(
    ("largest_site", "expected_dtype"),
    [(12, np.uint8), (12_000, np.uint16), (70_000, np.uint32)],
)
def test_operator_checkpoint_compaction_is_lossless(largest_site, expected_dtype):
    types = np.array([-1, 1, 2], dtype=np.int32)
    sites = np.array([0, 1, largest_site], dtype=np.int32)
    narrow_types, narrow_sites = compact_operator_checkpoint(types, sites)
    assert narrow_types.dtype == np.int8
    assert narrow_sites.dtype == expected_dtype
    np.testing.assert_array_equal(narrow_types.astype(np.int32), types)
    np.testing.assert_array_equal(narrow_sites.astype(np.int32), sites)


def test_operator_checkpoint_compaction_rejects_invalid_ranges():
    with pytest.raises(ValueError, match="fit int8"):
        compact_operator_checkpoint(np.array([129]), np.array([0]))
    with pytest.raises(ValueError, match="fit uint32"):
        compact_operator_checkpoint(np.array([1]), np.array([-1]))


def _write(writer: RankChunkWriter, index: int, begin: int) -> None:
    work = np.arange(begin, begin + 2, dtype=np.float64)
    attempts = np.arange(begin + 10, begin + 12, dtype=np.int64)
    writer.write_chunk(
        index,
        datasets={"work": work, "attempts": attempts},
        attrs={"n_trajectories": 2},
        checkpoint_datasets={
            "op_types": np.array([1, -1, 1], dtype=np.int32),
            "op_sites": np.array([0, 1, 0], dtype=np.int32),
        },
        checkpoint_attrs={"sweep_id": begin + 2, "topology_id": begin + 3},
        prune_previous_checkpoints=True,
    )


def test_checkpointed_chunks_append_and_restore_exact_state(tmp_path):
    run_dir = tmp_path / "run"
    attrs = {"backend": "cuda", "K": 20, "mask": np.array([0, 1], np.uint8)}
    with RankChunkWriter(run_dir, 0, run_attrs=attrs) as writer:
        _write(writer, 0, 0)
    with RankChunkWriter(run_dir, 0, run_attrs=attrs, resume=True) as writer:
        _write(writer, 1, 2)

    loaded = load_checkpointed_rank_chunks(run_dir, 0, _DATASETS)
    assert loaded["completed"] == 4
    assert loaded["next_chunk"] == 2
    np.testing.assert_array_equal(loaded["datasets"]["work"], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(loaded["datasets"]["attempts"], [10, 11, 12, 13])
    np.testing.assert_array_equal(
        loaded["checkpoint"]["datasets"]["op_types"], [1, -1, 1]
    )
    assert loaded["checkpoint"]["attrs"]["sweep_id"] == 4
    assert loaded["checkpoint"]["attrs"]["topology_id"] == 5
    with h5py.File(rank_file(run_dir, 0), "r") as handle:
        assert "checkpoint" not in handle["chunk0"]
        assert "checkpoint" in handle["chunk1"]


def test_resume_ignores_and_cleans_unpublished_pending_transaction(tmp_path):
    run_dir = tmp_path / "run"
    attrs = {"backend": "cuda", "K": 20}
    with RankChunkWriter(run_dir, 0, run_attrs=attrs) as writer:
        _write(writer, 0, 0)
    with h5py.File(rank_file(run_dir, 0), "a") as handle:
        pending = handle.create_group("_pending_chunk1")
        pending.create_dataset("work", data=np.array([999.0]))

    loaded = load_checkpointed_rank_chunks(run_dir, 0, _DATASETS)
    assert loaded["completed"] == 2
    assert loaded["next_chunk"] == 1
    with RankChunkWriter(run_dir, 0, run_attrs=attrs, resume=True) as writer:
        assert "_pending_chunk1" not in writer._h5
        _write(writer, 1, 2)
    assert load_checkpointed_rank_chunks(run_dir, 0, _DATASETS)["completed"] == 4


def test_resume_rejects_incompatible_run_attributes(tmp_path):
    run_dir = tmp_path / "run"
    with RankChunkWriter(run_dir, 0, run_attrs={"K": 20}) as writer:
        _write(writer, 0, 0)
    with pytest.raises(ValueError, match="attribute mismatch for K"):
        RankChunkWriter(run_dir, 0, run_attrs={"K": 21}, resume=True)


def test_loader_rejects_chunk_without_matching_sample_lengths(tmp_path):
    run_dir = tmp_path / "run"
    with RankChunkWriter(run_dir, 0, run_attrs={"K": 20}) as writer:
        writer.write_chunk(
            0,
            datasets={
                "work": np.array([1.0]),
                "attempts": np.array([1], dtype=np.int64),
            },
            attrs={"n_trajectories": 2},
            checkpoint_datasets={"op_types": np.array([1], dtype=np.int32)},
        )
    with pytest.raises(ValueError, match="length does not match"):
        load_checkpointed_rank_chunks(run_dir, 0, _DATASETS)


def test_pruned_checkpoint_storage_is_bounded_across_many_chunks(tmp_path):
    """Raw samples grow with chunks; large continuation state must not."""
    run_dir = tmp_path / "run"
    rng = np.random.default_rng(83)
    # Incompressible enough to expose accidental one-full-state-per-chunk
    # growth.  The transactional writer may briefly need old+new state, so the
    # high-water mark is two payloads rather than one.
    state = rng.integers(
        0, np.iinfo(np.int32).max, size=1 << 18, dtype=np.int32
    )
    with RankChunkWriter(run_dir, 0, run_attrs={"backend": "cuda"}) as writer:
        for chunk in range(8):
            writer.write_chunk(
                chunk,
                datasets={
                    "work": np.array([float(chunk)]),
                    "attempts": np.array([chunk], dtype=np.int64),
                },
                attrs={"n_trajectories": 1},
                checkpoint_datasets={"op_sites": state},
                checkpoint_attrs={"sweep_id": chunk + 1},
                prune_previous_checkpoints=True,
            )

    with h5py.File(rank_file(run_dir, 0), "r") as handle:
        owners = [
            name for name in handle
            if name.startswith("chunk") and "checkpoint" in handle[name]
        ]
        assert owners == ["chunk7"]
    # A broken implementation retaining eight copies is ~8 MiB.  Allow room
    # for gzip/HDF metadata and the atomic old+new high-water mark.
    assert rank_file(run_dir, 0).stat().st_size < 4 * state.nbytes


def test_checkpoint_tree_detects_only_published_chunks(tmp_path):
    run_dir = tmp_path / "tree" / "region" / "K2"
    run_dir.mkdir(parents=True)
    with h5py.File(rank_file(run_dir, 0), "w") as handle:
        handle.create_group("_pending_chunk0")
    assert not checkpoint_tree_has_committed_chunks(tmp_path / "tree")
    with h5py.File(rank_file(run_dir, 0), "a") as handle:
        handle.create_group("chunk0")
    assert checkpoint_tree_has_committed_chunks(tmp_path / "tree")


class _CollectiveComm:
    def __init__(self, states):
        self.states = states

    def gather(self, _value, root=0):
        return self.states

    def bcast(self, value, root=0):
        return value


def test_collective_resume_requires_all_active_ranks_to_agree():
    assert collective_resume_decision(
        _CollectiveComm([True, True, None]), rank=0, active=True,
        completed=1, allow_all_missing=False, label="gate"
    )
    assert not collective_resume_decision(
        _CollectiveComm([False, False]), rank=0, active=True,
        completed=0, allow_all_missing=True, label="gate"
    )
    with pytest.raises(RuntimeError, match="partial MPI checkpoint"):
        collective_resume_decision(
            _CollectiveComm([True, False]), rank=0, active=True,
            completed=1, allow_all_missing=True, label="gate"
        )
    with pytest.raises(FileNotFoundError, match="no committed CUDA checkpoint"):
        collective_resume_decision(
            _CollectiveComm([False, False]), rank=0, active=True,
            completed=0, allow_all_missing=False, label="gate"
        )
