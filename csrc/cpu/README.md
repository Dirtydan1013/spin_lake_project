# QAQMC CPU backend

The CPU backend is isolated under `csrc/cpu` so it can coexist with the CUDA
backend without mixing implementation files or Python module registration.
The compiled Python extension remains `qaqmc_cpp`; existing Python engines,
MPI drivers, checkpoints, and Slurm scripts do not need import changes.

## Module map

| Path | Responsibility |
| --- | --- |
| `include/qaqmc_core.hpp` | Public standard QAQMC model, chain, and off-diagonal API |
| `include/qaqmc_renyi_core.hpp` | Public two-replica Rényi API |
| `include/qaqmc_renyi_work_core.hpp` | Public nonequilibrium Rényi-work API |
| `include/sse_core.hpp` | Public finite-temperature SSE API |
| `detail/qaqmc_core.cpp` | Standard diagonal, event, cluster, profile implementation |
| `detail/qaqmc_off_diagonal_core.*` | String seam and half-line transitions |
| `detail/qaqmc_renyi_core.cpp` | Replica/channel transitions and topology updates |
| `detail/qaqmc_renyi_work_core.cpp` | Rényi-work protocol implementation |
| `detail/sse_core.cpp` | SSE diagonal and cluster implementation |
| `detail/sse_off_diagonal_core.*` | SSE string-work transition implementation |
| `bindings/module.cpp` | Minimal `qaqmc_cpp` module entry point |
| `bindings/bindings_*.cpp` | One pybind registration unit per public engine |
| `bindings/fragments/qaqmc_*.inc` | Standard-engine bindings grouped by observable domain |

Public headers live in `include/` (compiled with `csrc/cpu` as the include
root, so C++ code includes them as `include/qaqmc_core.hpp` etc.).  The old
root-level `csrc/*.hpp` compatibility shims were removed after the backend
merges; nothing may include through `csrc/` root any more.

## Dependency direction

```text
Python engines / MPI / scripts
              |
              v
         qaqmc_cpp module
              |
              v
      bindings/bindings_*.cpp
              |
              v
        public CPU headers
              |
              v
        detail implementation
```

- Python code must depend on the extension API, never on a C++ source path.
- Binding files include public headers only; they do not include detail `.cpp`
  files or reach into private storage.
- Standard, Rényi, Rényi-work, and SSE keep separate registration units.
- Observable binding fragments remain part of the standard binding translation
  unit because they continue one `py::class_` fluent expression; the fragments
  are organizational and do not create new runtime boundaries.
- `qaqmc_cuda_bridge.inc` owns the one-time CPU→CUDA model export schema. It
  reconstructs the stable int32 alias arrays from compact CPU storage without
  adding permanent per-chain memory.
- The shared-model CPU batch runner remains a Python orchestration layer over
  `QAQMCEngine`; it does not duplicate transition code.

## CPU/CUDA merge contract

- CPU builds produce `qaqmc_cpp`; optional CUDA builds produce `qaqmc_cuda`.
- The backends may share Python orchestration and result schemas, but neither
  backend includes the other's implementation headers.
- CMake owns separate CPU engine/module source lists so the CUDA option can be
  merged without editing individual CPU paths again.
- Legacy Python imports and public class names are intentionally unchanged.

### Known branch-overlap resolutions

When merging the current `gpu_version` branch:

1. Keep `QAQMC_CPU_ENGINE_SOURCES` and `QAQMC_CPU_MODULE_SOURCES` from this
   branch, then append the GPU branch's optional `QAQMC_ENABLE_CUDA` target.
2. Do not restore the old monolithic `csrc/bindings.cpp`. Its GPU model-export
   additions now live in `bindings/fragments/qaqmc_cuda_bridge.inc` and support
   the compact CPU alias representation.
3. Do not reapply the GPU branch's old `csrc/qaqmc_core.hpp` accessors. The
   maintained public implementation is `csrc/cpu/include/qaqmc_core.hpp`; the root
   header is only a compatibility include.
4. Combine both `tests/conftest.py` requirements: pin an explicitly selected
   CPU build and also expose `build_cuda` for GPU tests.
5. After resolving, run CPU engines/MPI tests with `QAQMC_TEST_BUILD_DIR` and
   the complete real-GPU suite before accepting the merge.
