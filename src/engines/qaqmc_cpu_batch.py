"""Independent CPU QAQMC chains sharing one immutable model in one process."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.engines.qaqmc import QAQMC_Rydberg
from src.mpi.driver_util import RANK_SEED_STRIDE


class QAQMCSharedModelBatch:
    """Run B standard QAQMC chains while storing model tables only once.

    The C++ ``mc_step`` binding releases the GIL. A persistent Python thread
    pool therefore maps one worker to each independent chain; all mutable
    operator/event/RNG state remains private while grouped-alias and geometry
    arrays are referenced through one ``QAQMCModelData`` object.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        seed: int = 42,
        seed_stride: int = RANK_SEED_STRIDE,
        **engine_kwargs: Any,
    ) -> None:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not bool(engine_kwargs.get("use_cpp", True)):
            raise ValueError("shared-model batches require use_cpp=True")
        # Model construction may use OpenMP, but concurrent chain transitions
        # should remain one host thread per chain.
        engine_kwargs = dict(engine_kwargs)
        engine_kwargs["use_cpp"] = True
        engine_kwargs["n_jobs"] = 1
        engine_kwargs["backend"] = "thread"
        engine_kwargs["omp_threads"] = 1
        engine_kwargs["verbose"] = False

        first = QAQMC_Rydberg(seed=int(seed), **engine_kwargs)
        if first._cpp_engine is None:
            raise RuntimeError("qaqmc_cpp is unavailable")
        model = first._cpp_engine.model_data
        chains = [first]
        for lane in range(1, batch_size):
            chains.append(QAQMC_Rydberg(
                seed=int(seed) + lane * int(seed_stride),
                model_data=model,
                **engine_kwargs,
            ))

        self.chains = chains
        self.batch_size = batch_size
        self.model_data = model
        self._pool = ThreadPoolExecutor(
            max_workers=batch_size,
            thread_name_prefix="qaqmc-chain",
        )
        self._closed = False

    @property
    def shared_model_bytes(self) -> int:
        return int(self.model_data.logical_bytes)

    @property
    def dominant_chain_bytes(self) -> list[int]:
        return [
            int(chain._cpp_engine.memory_breakdown["per_chain_capacity_bytes"])
            for chain in self.chains
        ]

    @property
    def dominant_resident_bytes(self) -> int:
        return self.shared_model_bytes + sum(self.dominant_chain_bytes)

    def mc_step(self) -> None:
        if self._closed:
            raise RuntimeError("batch is closed")
        futures = [
            self._pool.submit(chain._cpp_engine.mc_step)
            for chain in self.chains
        ]
        for future in futures:
            future.result()

    def run_steps(self, count: int) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        for _ in range(int(count)):
            self.mc_step()

    def run_profiles(self, *args, **kwargs) -> list[dict[str, Any]]:
        """Run the established C++ profile production loop on every lane.

        Observable geometry must be configured on each ``chain._cpp_engine``
        before calling this method. The binding releases the GIL around each
        transition/profile capture, so lane kernels execute concurrently while
        each lane returns the unchanged single-chain result schema.
        """
        if self._closed:
            raise RuntimeError("batch is closed")
        futures = [
            self._pool.submit(chain._cpp_engine.run_profile, *args, **kwargs)
            for chain in self.chains
        ]
        return [future.result() for future in futures]

    def close(self) -> None:
        if not self._closed:
            self._pool.shutdown(wait=True)
            self._closed = True

    def __enter__(self) -> "QAQMCSharedModelBatch":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
