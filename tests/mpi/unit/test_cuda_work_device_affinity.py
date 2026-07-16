"""Node-local CUDA affinity tests for both work MPI drivers."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from src.mpi.qaqmc_renyi_work_mpi import _cuda_device_for_rank as renyi_device
from src.mpi.qaqmc_string_work_mpi import _cuda_device_for_rank as string_device


class _LocalComm:
    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.freed = False

    def Get_rank(self) -> int:
        return self.rank

    def Free(self) -> None:
        self.freed = True


class _WorldComm:
    def __init__(self, local_rank: int) -> None:
        self.local = _LocalComm(local_rank)

    def Split_type(self, _kind):
        return self.local


@pytest.mark.parametrize("selector", [string_device, renyi_device])
def test_single_visible_slurm_gpu_always_uses_local_device_zero(monkeypatch, selector):
    monkeypatch.setitem(
        sys.modules, "qaqmc_cuda", SimpleNamespace(device_info=lambda: [{"index": 0}])
    )
    comm = _WorldComm(local_rank=7)
    assert selector(comm) == 0
    assert comm.local.freed


@pytest.mark.parametrize("selector", [string_device, renyi_device])
def test_all_visible_gpus_use_shared_memory_local_rank(monkeypatch, selector):
    monkeypatch.setitem(
        sys.modules,
        "qaqmc_cuda",
        SimpleNamespace(device_info=lambda: [{"index": i} for i in range(3)]),
    )
    comm = _WorldComm(local_rank=5)
    assert selector(comm) == 2
    assert comm.local.freed


@pytest.mark.parametrize("selector", [string_device, renyi_device])
def test_no_visible_gpu_fails_before_splitting_communicator(monkeypatch, selector):
    monkeypatch.setitem(
        sys.modules, "qaqmc_cuda", SimpleNamespace(device_info=lambda: [])
    )
    comm = _WorldComm(local_rank=0)
    with pytest.raises(RuntimeError, match="no GPU is visible"):
        selector(comm)
    assert not comm.local.freed
