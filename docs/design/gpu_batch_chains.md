# QAQMC CUDA batch chains

## What is implemented

All three CUDA workflows now have a true one-process, multi-chain path:

- standard QAQMC: `QAQMC_Rydberg_CUDA_Batch`;
- off-diagonal string work: `QAQMCStringWorkRydbergCUDABatch`;
- two-replica Rényi work: `QAQMCRenyiWorkRydbergCUDABatch`.

The low-level extension exposes `qaqmc_cuda.BatchedDiagonalEngine` and
`qaqmc_cuda.BatchedRenyiEngine`.  Standard and string work share the diagonal
batch engine because their transition core is the same; string work adds an
independent seam sector and topology counter per lane.  Rényi work stores two
operator strings and an independent region mask per lane.

This is not multiple Python processes pointed at one GPU.  One process owns the
batch, immutable model tables exist once, and independent CUDA streams overlap
the existing per-chain kernels.  Mutable state and checkpoints remain private
to each lane.

## Determinism and invariants

- Lane zero uses the original seed, so `B=1` is bit-exact with the established
  single-chain CUDA wrapper.
- Other lanes use deterministic 64-bit Philox keys separated by the golden-ratio
  stride `0x9E3779B97F4A7C15`.
- Sweep and topology counters are arrays of length `B`.
- Within one chain, diagonal then cluster update order is unchanged.
- String seam masks and Rényi masks/checkpoints can be set and restored per lane.
- A final trajectory wave may contain fewer than `B` requested samples; only
  active lanes are returned.

## Memory model

After lazy event/cluster workspace allocation,

```text
batch VRAM = shared immutable model + B * mutable chain state.
```

For the production probe `N=216`, `M=2,760,000`, 23,220 bonds and 600 delta
groups on the 32 GiB V100:

| Workflow | Shared model | Mutable per chain | B=4 resident | B=8 resident |
| --- | ---: | ---: | ---: | ---: |
| standard | 321.2 MiB | 684.3 MiB | 3058.3 MiB | 5795.5 MiB |
| string work | 321.2 MiB | 684.3 MiB | 3058.3 MiB | 5795.5 MiB |
| Rényi work | 321.2 MiB | 1115.9 MiB | 4784.7 MiB | 9248.2 MiB |

These are engine-accounted allocations after one full MC step, not total
process usage from `nvidia-smi`.  CUDA context/library overhead and safety
margin still have to be reserved when selecting a production batch size.

## V100 throughput result

Slurm job `26720` ran the production-size transition probe on a V100 PCIe 32
GiB.  The reported rate is completed independent chain steps per second:

| Workflow | B=1 | B=2 | B=4 | B=8 | Best gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard | 47.16 | 60.59 | 71.34 | 66.85 | 1.512x at B=4 |
| string work | 49.66 | 60.78 | 70.73 | 65.89 | 1.424x at B=4 |
| Rényi work | 23.25 | 29.47 | 34.81 | 32.97 | 1.497x at B=4 |

`B=4` is the present V100 recommendation.  `B=8` fits easily but loses
throughput because the GPU is already saturated; capacity is not the same as a
good operating point.  A100 and the second V100 measurement is represented by
three-GPU job `26722`, currently pending because another job owns one of the
three GPUs; all three cannot be allocated together yet.

## Reproducing the probe

```bash
sbatch scripts/bench/benchmark_qaqmc_batch_cuda.sh
sbatch scripts/bench/benchmark_qaqmc_batch_cuda_all_gpus.sh
```

`BATCH_SIZES`, `M`, `WARMUP`, `STEPS`, and `ENGINES` can be overridden through
environment variables.  The Python implementation is
`-m src.probes.qaqmc_batch_cuda`.

## Verification matrix

The GPU suite checks independent-lane exactness, `B=1` compatibility, shared
model accounting, per-lane seam/mask/topology/checkpoint behavior, profile
shape, complete work protocols, and partial final waves.  After the final clean
rebuild and source split, V100 job `26721` passed all 108 GPU tests plus the
standard, string-work, and Rényi probes with empty stderr.
