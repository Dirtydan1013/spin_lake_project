# E11 — U(1) 線：平衡 SSE 的 RCSL 檢驗 pilot（2026-07-11）

## 目的

paper/u1（Geim et al.）的 RK 態對角觀測量 == 無窮溫古典 dimer 氣體（其 Methods Eq. 6）。
問：**平衡** Rydberg 熱態在 kagome 頂點晶格上，溫度落在
Ω⁶/δ⁵ ≪ T ≪ Δ_monomer 窗口時，對角觀測量是否重現 RK/古典 dimer 氣體
（Wang–Pollet RCSL 邏輯的 U(1)/bipartite 版）？以及 vdW tails 的
Boltzmann 加權（β·V_tail）何時破壞均勻性、把系統推向 nematic 選擇。

## 預測（跑之前登記）

1. β 小（β·V_nnn ≲ 1，V_nnn = V₀/27 ≈ 0.196）：dimer manifold 近均勻 →
   ⟨V⟩ → 0.29（worm MC 同尺寸參考值）、C_s^FM 衰減與 RK 參考一致、|Φ|²/T² → 0。
2. β 大（β·V_nnn ≫ 1）：Boltzmann 傾斜 → ⟨V⟩ 偏離（dimer 組態按 tail 能量
   加權，類比其 ED Fig 7b 的 β_eff 熱擬合）、可能出現 nematic 選擇。
3. monomer 密度隨 β 升高而下降（gap ~ O(δ)·Ω，βΔ ≫ 1 時凍結）；
   密度 → 1/3⁻。
4. 交叉點粗估：β* ~ 5–10（β·V_nnn ~ 1–2）。

## 設定

- `sse_mpi --lattice kagome`（頂點版，commit 5def3ce）、12×6 torus（N=216，
  PBC，與 paper 36×6 同縱橫比家族）、a=2.0（nn=1）、Rb=1.32（V₀=5.29Ω）、
  full 1/r⁶、δ=3.5。
- β ∈ {2, 4, 8, 16}；每 β：n_equil 5000、n_samples 8000、checkpoint 2000、
  100 snapshots/chunk。本機 pilot（與 E09 SLURM jobs 無關）。
- RK 參考：`src.u1.worm_rk` 12×6（同 torus）。
- 資料：`data/u1_sse_12x6_beta{B}_delta3.5/`；分析用
  `src.u1.honeycomb_dimer` snapshot 管線。
- ⚠️ 引擎 string/loop 存的是 ∏(1−2n) = (−1)^len·∏σᶻ_paper。

## 結果

（待填）

## 結論

（待填）

## 後續

（待填；→ E11 QAQMC sweep-rate 掃描 = 虛時間 hemidiabatic 窗口）
