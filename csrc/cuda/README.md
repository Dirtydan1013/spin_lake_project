# QAQMC CUDA backend

This directory keeps the CUDA backend split by responsibility.  The public
headers contain stable host-facing interfaces; implementation-only state and
kernels live under `detail/`; Python bindings do not reach into implementation
details.

## Module map

| Module | Responsibility |
| --- | --- |
| `qaqmc_cuda_api.cuh` / `qaqmc_cuda_scan.cu` | Runtime discovery and the public packed-prefix primitive |
| `qaqmc_cuda_diagonal.cuh` / `.cu` | Standard QAQMC engine and off-diagonal string extension |
| `qaqmc_cuda_renyi.cuh` / `.cu` | Two-replica Rényi-work engine |
| `qaqmc_cuda_scan.cuh` | Compatibility umbrella for callers that need all public APIs |
| `bindings_cuda.cpp` | Minimal Python module entry point |
| `bindings_cuda_runtime.cpp` | Runtime and prefix-scan Python bindings |
| `bindings_cuda_diagonal.cpp` | `DiagonalEngine` Python bindings |
| `bindings_cuda_renyi.cpp` | `RenyiEngine` Python bindings |

The implementation-only headers are narrower:

| Detail module | Responsibility |
| --- | --- |
| `qaqmc_cuda_common.cuh` | Error handling, device buffers, RNG, and shared scalar helpers |
| `qaqmc_cuda_scan_primitives.cuh` | Packed-state prefix scan and propagation primitives |
| `qaqmc_cuda_diagonal_kernels.cuh` | Diagonal update, event construction, and cluster kernels |
| `qaqmc_cuda_offdiagonal_kernels.cuh` | String-seam and half-line topology kernels |
| `qaqmc_cuda_renyi_transition_kernels.cuh` | Two-replica propagation and Rényi transition kernels |
| `qaqmc_cuda_renyi_topology_kernels.cuh` | Rényi mask/topology proposal kernels |
| `qaqmc_cuda_prefix_kernels.cuh` | Standalone public prefix-XOR launch kernels |
| `qaqmc_cuda_diagonal_state.cuh` | Private `DiagonalEngine::Impl` device state |
| `qaqmc_cuda_renyi_state.cuh` | Private `RenyiEngine::Impl` device state |

## Dependency direction

```text
Python binding modules
        |
        v
public API headers
        |
        v
engine host translation units
        |
        +--> private engine state
        |
        +--> owning kernel module
                 |
                 v
       scan primitives / common utilities
```

Keep dependencies flowing downward.  In particular:

- Public headers must not expose CUDA allocation types or engine `Impl` fields.
- Binding files may depend only on public headers.
- A new kernel belongs in the module for the update it implements; shared
  machinery belongs in `common` or `scan_primitives` only when two engines use it.
- Off-diagonal string logic remains an extension of `DiagonalEngine`; Rényi
  replica-transition logic remains isolated from it.
- Device functions are currently consumed within their owning CUDA translation
  unit, so separable CUDA compilation is intentionally disabled.

This split is organizational only: it does not change the Markov transition,
RNG keys, acceptance ratios, or device-memory layout.
