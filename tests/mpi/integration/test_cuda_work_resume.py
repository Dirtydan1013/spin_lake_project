"""Interrupted CUDA work runs resume with the exact uninterrupted sample stream.

The fake engines exercise MPI orchestration, atomic HDF5 chunks, rolling
operator-state restoration and Philox counters without requiring a GPU.
Kernel-level checkpoint copies remain covered by ``tests/gpu``.
"""

from __future__ import annotations

from types import SimpleNamespace
import zlib

import numpy as np
import pytest

import src.mpi.qaqmc_renyi_work_mpi as renyi_mpi
import src.mpi.qaqmc_string_work_mpi as string_mpi


def test_kp_region_seed_is_independent_of_python_hash_randomization():
    expected = 7 + 7919 * (zlib.crc32(b"A-full") & 0xFFFF)
    assert renyi_mpi._stable_region_seed(7, "A-full") == expected
    assert renyi_mpi._stable_region_seed(7, "A-full") != (
        renyi_mpi._stable_region_seed(7, "A-bulk")
    )


class _Comm:
    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def Barrier(self):
        return None

    def bcast(self, value, root=0):
        return value

    def gather(self, value, root=0):
        return [value]

    def reduce(self, value, op=None, root=0):
        return value


_FAKE_MPI = SimpleNamespace(COMM_WORLD=_Comm(), MAX=object())


class _Interrupted(RuntimeError):
    pass


class _Counters:
    def __init__(self):
        self.sweep_id = 0
        self.topology_id = 0


class _FakeRenyiEngine:
    interrupt_after: int | None = None

    def __init__(self, **_kwargs):
        self._backend = _Counters()
        self._cpp_engine = self
        self.state = 0
        self.calls = 0

    def set_region_pair(self, _start, _end):
        return None

    def set_lambda_schedule(self, _schedule):
        return None

    def thermalize(self, count):
        self.state += int(count)

    def run_trajectories(self, count, _decorrelation):
        if self.interrupt_after is not None and self.calls >= self.interrupt_after:
            raise _Interrupted("simulated scheduler kill")
        self.calls += 1
        begin = self.state
        values = np.arange(begin, begin + count, dtype=np.float64) / 10.0
        self.state += int(count)
        self._backend.sweep_id += 3 * int(count)
        self._backend.topology_id += 2 * int(count)
        return SimpleNamespace(
            work_samples=values,
            final_swap_counts=np.arange(begin, begin + count, dtype=np.int32),
            unjoined_counts_per_traj=np.zeros(count, dtype=np.int32),
            topology_attempts_per_traj=np.full(
                count, self._backend.sweep_id, dtype=np.int64),
            topology_accepts_per_traj=np.full(
                count, self._backend.topology_id, dtype=np.int64),
        )

    def export_start_config(self):
        state = np.array([self.state, 1, 1, 1], dtype=np.int32)
        sites = np.zeros(4, dtype=np.int32)
        return state.copy(), sites.copy(), state.copy(), sites.copy()

    def import_start_config(self, types0, _sites0, _types1, _sites1):
        self.state = int(types0[0])


class _InterruptingRenyiEngine(_FakeRenyiEngine):
    interrupt_after = 1


class _FakeStringBackend(_Counters):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.has_checkpoint = False

    def restore_device_checkpoint(self):
        return None

    def save_device_checkpoint(self):
        self.has_checkpoint = True

    def get_operator_string(self):
        return (
            np.array([self.owner.state, 1, 1, 1], dtype=np.int32),
            np.zeros(4, dtype=np.int32),
        )

    def set_op_string(self, types, _sites):
        self.owner.state = int(types[0])


class _FakeStringEngine:
    interrupt_after: int | None = None

    def __init__(self, **_kwargs):
        self.state = 0
        self.calls = 0
        self._length = 0
        self._checkpoint_mask = None
        self._eng = _FakeStringBackend(self)

    def set_string_sites(self, sites, _m_star):
        self._length = len(sites)

    def set_lambda_schedule(self, _schedule):
        return None

    def _full_mask(self):
        return (1 << self._length) - 1

    def thermalize(self, count, direction="forward"):
        self.state += int(count)
        self._checkpoint_mask = 0 if direction == "forward" else self._full_mask()
        self._eng.save_device_checkpoint()

    def run_trajectories(self, count, _decorrelation, **kwargs):
        if self.interrupt_after is not None and self.calls >= self.interrupt_after:
            raise _Interrupted("simulated scheduler kill")
        self.calls += 1
        begin = self.state
        samples = np.arange(begin, begin + count, dtype=np.float64) / 10.0
        self.state += int(count)
        self._eng.sweep_id += 3 * int(count)
        self._eng.topology_id += 2 * int(count)
        self._eng.has_checkpoint = True
        direction = kwargs.get("direction", "forward")
        self._checkpoint_mask = (
            0 if direction == "forward" else self._full_mask()
        )
        return SimpleNamespace(log_j_samples=samples)


class _InterruptingStringEngine(_FakeStringEngine):
    interrupt_after = 1


def _run_renyi(monkeypatch, engine_type, checkpoint_dir, *, resume=False,
               epsilon=0.01, k_values=None):
    monkeypatch.setattr(renyi_mpi, "MPI", _FAKE_MPI)
    monkeypatch.setattr(
        renyi_mpi, "_engine_type_for_backend", lambda _backend: engine_type
    )
    monkeypatch.setattr(renyi_mpi, "_cuda_device_for_rank", lambda _comm: 0)
    return renyi_mpi.run_work_mpi(
        N=2, M=2, Omega=1.0, Rb=0.0,
        delta_min=0.0, delta_max=1.0, epsilon=epsilon,
        pos=np.arange(2, dtype=np.float64).reshape(-1, 1),
        A_start_mask=np.zeros(2, dtype=np.uint8),
        A_end_mask=np.array([1, 0], dtype=np.uint8),
        K_values=([2] if k_values is None else list(k_values)),
        n_trajectories=4, n_thermalize=0,
        decorrelation_steps=0, compute_ed=False,
        checkpoint_every_trajectories=2,
        checkpoint_dir=str(checkpoint_dir), backend="cuda", resume=resume,
        permute_site_labels=False, verbose=False,
    )


def _run_string(monkeypatch, engine_type, checkpoint_dir, *, resume=False,
                k_values=None):
    monkeypatch.setattr(string_mpi, "MPI", _FAKE_MPI)
    monkeypatch.setattr(
        string_mpi,
        "_engine_type_and_schedule_for_backend",
        lambda _backend: (
            engine_type,
            lambda count: np.linspace(0.0, 1.0, count + 1),
        ),
    )
    monkeypatch.setattr(string_mpi, "_cuda_device_for_rank", lambda _comm: 0)
    return string_mpi.run_string_work_mpi(
        N=2, M=2, Omega=1.0, Rb=0.0,
        delta_min=0.0, delta_max=1.0, epsilon=0.01,
        pos=np.arange(2, dtype=np.float64).reshape(-1, 1),
        string_sites=[0],
        K_values=([2] if k_values is None else list(k_values)),
        schedule="linear",
        n_trajectories=4, n_thermalize=0, decorrelation_steps=0,
        checkpoint_every_trajectories=2,
        checkpoint_dir=str(checkpoint_dir), backend="cuda", resume=resume,
        permute_site_labels=False, verbose=False,
    )


def test_renyi_interrupted_resume_matches_uninterrupted_samples(monkeypatch, tmp_path):
    baseline = _run_renyi(
        monkeypatch, _FakeRenyiEngine, tmp_path / "baseline"
    )
    with pytest.raises(_Interrupted):
        _run_renyi(monkeypatch, _InterruptingRenyiEngine, tmp_path / "resume")
    resumed = _run_renyi(
        monkeypatch, _FakeRenyiEngine, tmp_path / "resume", resume=True
    )

    expected = baseline["K_results"][2]
    actual = resumed["K_results"][2]
    for key in (
        "work_samples", "final_swap_counts", "unjoined_counts_per_traj",
        "topology_attempts_per_traj", "topology_accepts_per_traj",
    ):
        np.testing.assert_array_equal(actual[key], expected[key])


def test_string_interrupted_resume_matches_uninterrupted_samples(monkeypatch, tmp_path):
    baseline = _run_string(
        monkeypatch, _FakeStringEngine, tmp_path / "baseline"
    )
    with pytest.raises(_Interrupted):
        _run_string(monkeypatch, _InterruptingStringEngine, tmp_path / "resume")
    resumed = _run_string(
        monkeypatch, _FakeStringEngine, tmp_path / "resume", resume=True
    )

    np.testing.assert_array_equal(
        resumed["K_results"][2]["log_j_samples"],
        baseline["K_results"][2]["log_j_samples"],
    )


def test_resume_rejects_changed_hamiltonian(monkeypatch, tmp_path):
    run_dir = tmp_path / "resume"
    with pytest.raises(_Interrupted):
        _run_renyi(monkeypatch, _InterruptingRenyiEngine, run_dir)
    with pytest.raises(ValueError, match="attribute mismatch for epsilon"):
        _run_renyi(
            monkeypatch, _FakeRenyiEngine, run_dir,
            resume=True, epsilon=0.02,
        )


def test_resume_rejects_missing_committed_checkpoint(monkeypatch, tmp_path):
    with pytest.raises(FileNotFoundError, match="no committed CUDA checkpoint"):
        _run_renyi(
            monkeypatch, _FakeRenyiEngine, tmp_path / "missing-renyi",
            resume=True,
        )
    with pytest.raises(FileNotFoundError, match="no committed CUDA checkpoint"):
        _run_string(
            monkeypatch, _FakeStringEngine, tmp_path / "missing-string",
            resume=True,
        )


def test_resume_starts_later_unstarted_k_only_after_prior_k_is_found(
        monkeypatch, tmp_path):
    renyi_dir = tmp_path / "renyi"
    _run_renyi(monkeypatch, _FakeRenyiEngine, renyi_dir, k_values=[2])
    renyi = _run_renyi(
        monkeypatch, _FakeRenyiEngine, renyi_dir,
        resume=True, k_values=[2, 3],
    )
    assert set(renyi["K_results"]) == {2, 3}

    string_dir = tmp_path / "string"
    _run_string(monkeypatch, _FakeStringEngine, string_dir, k_values=[2])
    string = _run_string(
        monkeypatch, _FakeStringEngine, string_dir,
        resume=True, k_values=[2, 3],
    )
    assert set(string["K_results"]) == {2, 3}
