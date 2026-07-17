# Project

**Goal**: probe Z2 spin-liquid signatures (Kitaev-Preskill γ) in non-equilibrium Rydberg dynamics on cropped kagome-bond (ruby/link) lattices via QAQMC Renyi-2 entropy.

**Reference papers**: `paper/<title>/paper.md` ×4 (Semeghini experiment, Wang–Pollet QMC/RCSL, Mauron t-VMC TEE, Vu NQS spin lakes) — cross-comparison in `paper/COMPARISON.md`. Main anchor: Vu et al., arXiv:2512.09040 ("spin lakes"). Same model/parameters as our production runs: Rb = 2.4·a_nn (code: a=4.0 → intra-triangle spacing 1.0, Rb=2.4 ✓), full 1/r⁶, PBC, δ/Ω ramp from −2. Their M_VBS/M_SS (End Matter Eq. 5-6) == our `M_vbs`/`M_ss` exactly. Their NQS T=0 phase diagram (contested!): trivial → VBS (n=1/4, 2×2 cell, δ≈3.4) → SS stripe (n=1/3, δ≈5.2); "spin lake" = ramp-prepared QSL-like state (TEE→ln2, optimal Γ≈0.03–0.2Ω ending δ≈4.5) despite non-QSL ground state.

**Architecture**
- `csrc/` — C++ engines (pybind11 → `qaqmc_cpp`): `QAQMCEngine` (single-replica, δ-sweep), `QAQMCRenyiEngine` (two-replica S₂; modes PairToggle / Expanded / Work, swap cut generalized to `cut_`), `QAQMCRenyiWorkEngine` + string-work (Jarzynski ΔS₂ / O_C), `SSEEngine` (thermal reference).
- `src/engines/` — Python wrappers; `src/rydberg/` — lattices + V_ij; `src/analysis/` — ED/measurement.
- `src/tee/` — S₂ estimators: ratio method (per-site visit-count ratios) and expanded-ensemble reweighting (log_g autotune + jackknife); `compose_tee` does the KP γ sum.
- `src/kp/` — KP region masks / boundary-first site orders on (nx, ny, m); `src/mpi/` — production drivers (`kp_tee_{ratio,expanded}_mpi`, `qaqmc_mpi --mode profile`, `qaqmc_renyi_work_mpi`, `qaqmc_string_work_mpi`, `sse_mpi`) with per-rank chunk checkpointing + warm start (`src/mpi/chunk_io.py`).
- `scripts/` — deployment layer: `common/env.sh` (SLURM *and* bare-server; resolves NTASKS/CPT, mpiexec binding), `submit.sh` (sbatch or nohup), production scripts in `slurm_scripts/`. Env spec: `environment.yml` (conda-forge openmpi/mpi4py/compiler — no system MPI needed). Build: `-DQAQMC_ARCH=native|x86-64-v3|...` (default native; changing march breaks bit-reproducibility, not statistics).

**Physics status (2026-07-10, equilibrium SSE campaign)**
- Scan-order bias found & fixed (PR #2): δ-sweep chains phase-locked into ONE stripe pattern via the shared site-scan order (old data: M1:M2:M3=30:32:1, per-site ⟨n⟩ std 0.24). `--permute-site-labels` now default ON in all four drivers; rerun profile confirms M3 restored (24:15:16) and phase-lock gone (std 0.047 == equilibrium).
- Equilibrium SSE (6×6/8×8 torus, β=20–80, δ=3.5–5.5): NO symmetry-broken phase anywhere — Ψ_VBS≈0.17 (below 1/√25 baseline), Ψ_SS≈0.37 flat, connected S(q) at K/M short-ranged (flat in L and β), per-bin M "exclusivity" == shuffle-null except weak +2.2σ at δ=5.5; stripe-init (n≈1/4 sweep pattern) melts in ~200 sweeps at δ=5.5 β=20; density stays ≈1/4 at δ=5.5 (their SS needs n=1/3).
- ⇒ direct contradiction with the paper's NQS/iDMRG phase diagram, but consistent with the QMC literature (Wang–Pollet "renormalized classical spin liquid", their ref [29]). δ≈4.0–4.5 = SL-candidate window (Z_l(2) peak, |A_v|≈0.82, C_m≈0 all sizes, BFFM≈0) — the target for KP TEE runs.
- Diagnostics live in `plots/plot_diagonal/plot_m_domains.py` (per-bin M-domain analysis; use instead of unconnected S(q), which ≈ connected under domain averaging).

**Open threads (as of 2026-07-10 evening)** — experiment log lives in `docs/progress/INDEX.md` (one file per experiment; keep it updated, use `docs/progress/TEMPLATE.md`):
- E09 deep-β SSE DONE (analyzed 2026-07-11 on branch `z2_spin_lake`): β 20→320 (×16) at δ=4.25 — Ψ_VBS flat ≈0.16 (=1/√36 baseline), exclusivity == shuffle-null at ALL β (E06's +2.2σ at δ=5.5 β=20 did NOT grow: −0.6σ at β=160; the raw corr(M1,M2)=+0.27 there is an inter-rank common-mode artifact, −0.08 within-rank). Liquid indicators STRENGTHEN with β at δ=4.25 (Z_l(2) 0.446→0.512, |A_v| 0.816→0.832). Prediction hit; RCSL holds to β=320 → KP TEE @δ=4.25 safe at β=20–80. Details `docs/progress/experiments/E09_*.md`.
- Next big step: **KP TEE run at δ≈4.25** (the SL-candidate window; matches Vu et al.'s TEE peak) — the whole pipeline exists (`kp_tee_{ratio,expanded}_mpi`, `run_kagome_renyi_work.sh` KP mode).
- 8×8 δ=4.25 (loop-size scaling / perimeter law) and equilibrium-vs-NQS energy arbitration remain open ideas.

**Progress (recent commits on main, 2026-07)**
- `fbdd59d` remove leftover ASan artifacts (profile segfault closed: was a `cp` over the live .so, not code — deploy with `mv`)
- `d4babc1` portable deployment (env.sh / submit.sh / environment.yml / QAQMC_ARCH); branch `TEE_Modify_Autotune` fast-forwarded into `main` on 2026-07-09
- `c84e229` equilibration progress printing in all four MPI production drivers (`--equil-progress-every`, default 500)
- `8fb7ddd` warm start records + validates spatial boundary in the config
- `6d3a955`/`0d8f97d` open/periodic spatial boundary in all four engines; periodic kagome_bond profile drops the bulk restriction
- `0039c4b` SSE optimization + unified chunked storage (flat `rank{r}.h5` + `chunk{i}` + `final_config`)
- `faa856e` swap boundary generalized `p < M` → `p < cut_` (work engine `set_cut`; default keeps symmetric path bit-exact)

Earlier: ratio/expanded entry points split; `--log_g_init` / `--skip_autotune` / `--warm_up_steps` decouple tune from production; physical-bound, ladder-consistency, cross-method, C3-symmetry tests (catch known shared-engine biases on 4×4 m=1 KP).
