# QAQMC CUDA backend

This directory keeps the CUDA backend split by responsibility.  The public
headers contain stable host-facing interfaces; implementation-only state and
kernels live under `detail/`; Python bindings do not reach into implementation
details.

## Module map

| Module | Responsibility |
| --- | --- |
| `include/api.cuh` / `src/scan.cu` | Runtime discovery and the public packed-prefix primitive |
| `include/diagonal.cuh` / `src/diagonal.cu` | Standard QAQMC engine and off-diagonal string extension |
| `include/renyi.cuh` / `src/renyi.cu` | Two-replica Rényi-work engine |
| `src/batch_diagonal.cu` | Multi-chain standard/string host orchestration |
| `src/batch_renyi.cu` | Multi-chain two-replica host orchestration |
| `bindings/module.cpp` | Minimal Python module entry point |
| `bindings/bindings_runtime.cpp` | Runtime and prefix-scan Python bindings |
| `bindings/bindings_diagonal.cpp` | `DiagonalEngine` Python bindings |
| `bindings/bindings_renyi.cpp` | `RenyiEngine` Python bindings |
| `bindings/bindings_batch_diagonal.cpp` | `BatchedDiagonalEngine` Python bindings |
| `bindings/bindings_batch_renyi.cpp` | `BatchedRenyiEngine` Python bindings |

The implementation-only headers are narrower:

| Detail module | Responsibility |
| --- | --- |
| `common.cuh` | Error handling, device buffers, RNG, and shared scalar helpers |
| `scan_primitives.cuh` | Packed-state prefix scan and propagation primitives |
| `diagonal_kernels.cuh` | Diagonal update, event construction, and cluster kernels |
| `offdiagonal_kernels.cuh` | String-seam and half-line topology kernels |
| `renyi_transition_kernels.cuh` | Two-replica propagation and Rényi transition kernels |
| `renyi_topology_kernels.cuh` | Rényi mask/topology proposal kernels |
| `prefix_kernels.cuh` | Standalone public prefix-XOR launch kernels |
| `diagonal_state.cuh` | Private `DiagonalEngine::Impl` device state |
| `renyi_state.cuh` | Private `RenyiEngine::Impl` device state |

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
- Batch translation units orchestrate existing single-chain transitions.  They
  must not duplicate transition kernels or change a chain's update ordering.
- Device functions are currently consumed within their owning CUDA translation
  unit, so separable CUDA compilation is intentionally disabled.

## Multi-chain execution model

`BatchedDiagonalEngine` and `BatchedRenyiEngine` own `B` independent mutable
chain states in one CUDA process.  Their immutable Hamiltonian, grouped-alias,
and bond-envelope buffers are held by one reference-counted
`DeviceHamiltonian`, so they are allocated once per batch.  Every chain keeps
its own Philox seed, sweep/topology counters, operator strings, seam/mask state,
and device checkpoint.

Batch calls dispatch the existing chain transition from separate host threads.
The CUDA target is compiled with per-thread default streams, allowing kernels
from independent chains to overlap without changing the established kernel
signatures or within-chain transition order.  `B=1` therefore remains exactly
compatible with the single-chain engine for the same seed and counters.

The split into host modules is organizational.  The batch feature changes
allocation ownership and launch concurrency, but does not change a chain's
Markov transition, RNG keys, acceptance ratios, or mutable device layout.
