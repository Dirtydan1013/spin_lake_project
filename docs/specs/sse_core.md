# 規格：csrc/cpu/include/sse_core.hpp / csrc/cpu/detail/sse_core.cpp

最後更新：2026-07-17（重寫；含 2026-07 優化後的資料結構、χ_F 估計器、
set_config warm start、periodic-τ off-diagonal seam、DiagonalObservables 共用）。

## 角色

`SSEEngine` 是有限溫度 Stochastic Series Expansion 引擎，模擬固定
Hamiltonian（單一 δ，不是 ramp）的 Rydberg model：

    H = −(Ω/2)Σᵢσˣᵢ + δΣᵢnᵢ + Σ_{i<j} V_ij nᵢnⱼ

它是整個 campaign 的**熱平衡對照**：QAQMC sweep 資料的每個結論都對照
SSE 平衡態（scan-order bias、RCSL、深 β、thermal entropy ladder 都靠它）。
與 QAQMC 的關鍵差異：可變有效長度 operator string（identity slot 可插拔、
`M_` 自動成長）、**periodic** imaginary-time boundary（trace）。

Python：`src/engines/sse.py`；drivers：`src/mpi/sse_mpi.py`（含 `--chi-f`）、
`src/mpi/sse_entropy_mpi.py`（β-ladder 熱熵）、`src/mpi/sse_string_work_mpi.py`
（thermal X_C string work）。

## 物件 / 函式

| 名稱 | 種類 | 可見性 | 用途 |
| --- | --- | --- | --- |
| `SSEEngine` | class | public | 主引擎 |
| `mc_step()` | method | public | diagonal update + cluster update + `adjust_M_if_needed()` |
| `measure_energy()` | method | public | `−n_ops/β + Σ_b cij_b`（energy_shift_ 修正） |
| `measure_density()` / `measure_mz()` | method | public | τ=0 態的 density / staggered mz |
| `measure_chi_f_terms(g_left, g_right)` | method | public | χ_F（WLT）：β/2 兩半的 Σ ∂δ ln W_p；**cut 用 j~Binomial(n_ops,½) exact 抽樣**（slot M/2 cut 有 anti-bunching bias）；消耗 RNG ⇒ 開啟會改變 trajectory bit 流（統計不變） |
| `diag_obs` | member | public | `DiagonalObservables`（與 QAQMC profile 共用的幾何/量測：loop/string/A_v/VBS/occ-SF），對 τ=0 態量測 |
| `get_state/op_types/op_sites` | method | public | int32 陣列存取 |
| `get_rng_state` / `set_rng_state` | method | public | RNG checkpoint |
| `set_config(state, types, sites)` | method | public | warm start：τ=0 spin 態 + 完整 op string（長度成為新 M_）；驗證非法輸入 throw |
| seam 系列（`set_string_sites`、`set_seam_mask`(=consistent)、`attempt_string_toggle`、`topology_sweep`…） | method | public | thermal X_C — 轉發 `SSEOffDiagonalCore`；**注意 Python 面的 `set_seam_mask` 綁到的就是 consistent 版**（trace 的 closure parity 無 raw-setter 的合法使用場景） |

## 物理 / 演算法契約

- Operator 編碼：`0` identity（可被 diagonal update 填/清）、`-1` σˣ、
  `1` diagonal site、`2` diagonal bond。**與 QAQMC 的差異就在 type 0。**
- Bond weight 慣例與 QAQMC 完全共用（`compute_bond_W_inline`；
  `delta_i = δ/z_eff[i]`；cutoff=-1 時 z_eff≡N−1，重現 δ_b = δ/(N−1)）。
- Periodic τ：cluster segment 可 wrap（wrap 的翻轉會改 `state_`）；
  這是與 QAQMC frozen-boundary 的本質差異。
- Diagonal update 是 identity↔diagonal 的 insertion/removal：插入接受率
  `∝ β·norm_N / (M − n_ops)`、移除為其倒數（`inv_beta_norm_` 免除法）。
- χ_F 契約：`χ_F = ½(⟨G_L G_R⟩ − ⟨G_L⟩⟨G_R⟩)`，G = 半條 string 的
  Σ ∂δ ln W_p（`bond_dlnW_` 預表含 cij 分支）；組裝在分析層。
- seam（thermal X_C）：per-site `parity(σˣ ops) XOR seam bit == even`
  由 trace 強加；half-line walk 可繞過 τ=0（wrapped commit 額外翻
  `state_[site]`）；唯一 invalid proposal 是該站全串無 single-site op。
  seam snapshot 更新按方向分流（right→plus、left→minus、wrap→state_）。
- RNG 流（2026-07 重寫後）與更早引擎**不** bit-compatible — 驗證一律對
  ED / 統計，不對舊 bytes。

## 模組輸入 / 輸出

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `N, Omega, delta, Rb, beta, epsilon` | scalar | Hamiltonian + 溫度（單一 δ！） |
| `seed, pos, neighbor_cutoff, box/n_box` | — | 同 QAQMC（periodic 空間邊界同樣支援） |

| 輸出 | Type / Shape | 意義 |
| --- | --- | --- |
| `measure_energy()` | double | E 估計（含 shift 修正）；S(T) ladder 的積分元 |
| `g_left/g_right` | double | χ_F 兩半 G |
| `diag_obs` 量測 | 依設定 | density/Z_l/C_m_l/A_v/VBS/occ-SF（binding run() 迴圈逐 bin 聚合） |
| `get_n_ops()` / `get_M()` | int | 有效長度 / 容量 |

## 資料契約

- `op_types_/op_sites_` 長度 = M_（容量），有效 op 數 = n_ops_；
  對外 int32。
- alias 表：**單一表**（一組 β/δ 一份，非 per-slice），16-byte AoS
  `AliasEntry`。
- vertex lists / packed bond events / int8 `bond_spin_` 與 QAQMC 同款
  （`[p:32|b:31|endpoint:1]`）；`vertex_counts_valid_` 同語義，
  `set_config` 清除。
- `bond_dlnW_`：`(n_bonds_pad*4)` 的 ∂δ ln W 預表（W=0 處為 0）。
- warm-start 檔案格式：`src/mpi/chunk_io.py` 的 `final_config`
  （state + op strings + rng_state attr）。

## 狀態 / 不變量

- worldline closure（trace）：每站 σˣ op 數為偶（seam 啟用時 XOR seam
  bit 後為偶）。
- `n_ops_ ≤ M_`；`adjust_M_if_needed` 維持 M_ ≈ 1.33·n_ops（這正是
  slot-cut χ_F 有 bias 的原因 — slot 位置 anti-bunched）。
- `energy_shift_ = Σ_b cij_b` 恆定（建構時定）。
- 同 build 同 seed ⇒ bit-identical；`measure_chi_f_terms` 開關會改 bit
  流（消耗 RNG），統計等價。

## 整體流程（mc_step）

1. `diagonal_update()`：逐 slot — identity 嘗試插入（alias 提案 +
   envelope rejection）、diagonal 嘗試移除、`-1` 傳播 state；
   `p == m_star` 時 seam hook XOR。
2. `cluster_update()`：`build_vertex_lists()`（若 counts 失效）後逐站
   segment Metropolis，ratio-product 無 log；wrap segment 翻 `state_`。
3. `adjust_M_if_needed()`：n_ops 逼近容量時擴 M_（identity 填充）。

## 邊界情況

- 站點無 single-site op：該站唯一 segment 是整條 worldline，翻轉合法
  （trace 無固定邊界）— 與 QAQMC 相反。
- `set_config` 長度/索引/type 非法 → `std::runtime_error`。
- β 很小（ladder anchor β₀=3e-4）：n_ops→0，identity-dominated；
  anchor 解析式在 driver 層處理。
- seam `set_seam_mask_consistent` 對無 single-site op 的站：把第一個
  identity slot 轉成 `-1` 當 seed（caller 必須 re-equilibrate —
  trajectory reset 路徑都會）。

## 效能備註（2026-07-07 優化，3.35× mc_step / 6.9× cluster）

- ratio-product segment Metropolis（無 std::log）、packed int64 events、
  int8 bond_spin、AoS alias、fused vertex counting、division-free
  insert/remove 接受判準。
- 單一 alias 表（β/δ 固定）⇒ 記憶體 O(N+n_bonds)，與 M 無關。
- ladder driver 的 per-rung log 印的是 rank0 最後一個 chunk 的均值 —
  **雜訊是外觀問題**，能量單調性一律用全 ensemble 重算
  （docs/design/cpu_memory.md 之外，見 memory entropy-ladder 註記）。

## 驗收標準 / 測試

- thermal ED gate：density 0.7σ、energy 1.5σ（1D 小系統）。
- `tests/engines/unit/test_sse_chi_f.py`：χ_F vs ED（含 Binomial cut）。
- `tests/engines/unit/test_sse_string_work_ed.py`：thermal X_C vs ED +
  **fixed-λ sector balance**（sharp check）。
- warm-start round-trip（`tests/mpi/`）：set_config 後 thermalization
  可跳過且統計一致。

## 相關檔案

- `csrc/cpu/detail/sse_off_diagonal_core.hpp/.cpp` — periodic-τ seam
  （與 QAQMC 版的三個結構差異寫在該 header 頂註）
- `csrc/cpu/detail/diagonal_observables.hpp` — 共用 diagonal 幾何/量測
- `csrc/cpu/bindings/bindings_sse.cpp` — run() 驅動迴圈 + 量測旗標
- `src/mpi/sse_mpi.py`、`src/mpi/sse_entropy_mpi.py`、
  `src/mpi/sse_string_work_mpi.py`
- `docs/progress/experiments/` E09/E13/E14 — 用此引擎的主要物理結果

## 開放問題

- SSE 未接 shared-model / compact op 儲存（單表 alias 本來就小，M 相關
  的 op string 才是大頭）— 若 deep-β production 記憶體吃緊再套用
  standard-engine 的型別窄化。
