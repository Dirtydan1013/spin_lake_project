# 規格：csrc/cpu/include/qaqmc_core.hpp / csrc/cpu/detail/qaqmc_core.cpp

最後更新：2026-07-17（重寫，對應 PR #4/#5 merge 後的引擎；含 compact memory
layout、shared model、off-diagonal seam 元件化、CUDA bridge）。

## 角色

`QAQMCEngine` 是 single-replica、detuning-ramp 的 QAQMC 引擎：固定長度
operator string 表示 imaginary-time ramp `delta_min → delta_max → delta_min`，
open τ boundary、兩端固定 `|0…0⟩`，沒有 identity operator、沒有可變
expansion order。

它同時是整個 CPU backend 的地基：

- `RydbergVij` / `build_rydberg_vij` / `AliasEntry` / RNG helpers 被
  `QAQMCRenyiEngine`、`SSEEngine` 共用。
- Off-diagonal string（X_C seam）功能整個委託給
  `QAQMCOffDiagonalCore`（`detail/qaqmc_off_diagonal_core.hpp`），engine 只留
  one-line pass-through。
- `export_cuda_diagonal_data`（binding fragment `qaqmc_cuda_bridge.inc`）把
  model tables 以穩定 int32 schema 匯出給 optional CUDA backend。

Python 端經 `src/engines/qaqmc.py`（`QAQMC_Rydberg`）、
`src/engines/qaqmc_cpu_batch.py`（`QAQMCSharedModelBatch`）與
`src/mpi/qaqmc_mpi.py`（profile production driver）使用。

## 物件 / 函式

| 名稱 | 種類 | 可見性 | 用途 |
| --- | --- | --- | --- |
| `uniform01` / `randint` | function | header-local | mt19937_64 之上的快速 [0,1) / Lemire 無偏 randint；所有 CPU 引擎共用 |
| `RydbergVij` | struct | public | active bonds、V_ij、endpoint list、z_eff、1/z_eff |
| `build_rydberg_vij()` | function | public | 由座標 + cutoff 建 bond；`box`/`n_box` 支援 periodic minimum-image（n_box=0 = open） |
| `QAQMCGroupedAlias` | struct | public | 不可變 grouped proposal tables（compact SoA；alias index 自動 u16/u32） |
| `QAQMCModelData` | struct | public | **process 內可共用**的 immutable model（幾何、V_ij、alias、ramp 參數）；`logical_bytes()` 供記憶體審計 |
| `AliasTable` | struct | public | 舊式 per-slice alias tables（`build_qaqmc_alias_tables`；Renyi engine 仍用其建構路徑） |
| `AliasEntry` | struct | public | 16-byte AoS alias slot：`prob` / `alias` / `loc_kind = (loc<<1)|kind` |
| `QAQMCEngine` | class | public | 主引擎；兩個建構子（全參數 / `shared_ptr<QAQMCModelData>` + seed） |
| `mc_step()` | method | public | 一次 diagonal update + cluster update |
| `mc_step_profiled(profile_step)` | method | public | fused step+profile：回傳**上一步**的 profile 樣本（one-step lag，統計等價、零額外 O(M) pass；假設 seam 未啟用） |
| `diagonal_update()` / `cluster_update()` | method | private | 核心 update kernels（見下） |
| `measure_at_midpoint()` / `measure_profile(step)` | method | public | p=M 對稱點 / 沿 ramp 的 diagonal observables |
| `set_observable_sites` / `set_bulk_sites` | method | public | loop(Z_l)/string(C_m_l) 複本與 density bulk 子集 |
| `set_snapshot_point_indices` | method | public | 指定 profile 點傾印完整 state 向量 |
| `set_occ_sf_q_points` / `set_occ_sf_site_map` / `set_occ_sf_point_indices` / `set_occ2_sf_site_map` | method | public | sublattice-resolved occupation SF（兩套 unit cell） |
| `set_vbs_triangles` | method | public | VBS/SS 序參量（paper Eq. 5-6）的 up-triangle 幾何 |
| `set_dimer_sf_*` / `measure_dimer_sf()` | method | public | dimer S(q)：forward-ramp 指定 δ 點的 s_q |
| `set_bond_event_storage(mode)` | method | public | cluster event 表示法：`packed64`（預設）/ `p_bond16` / `p_only32` |
| `get_memory_breakdown()` | method | public | per-array logical/capacity bytes（RSS 審計） |
| `delta_at(p)` | method | public inline | O(1) 計算 δ(p)，**與舊 materialized schedule bit-identical**；`get_delta_schedule()` 是按需匯出 |
| `get_rng_state` / `set_rng_state` / `set_op_string` | method | public | checkpoint（RNG 序列化 + int32 op string 匯入） |
| `export_op_types` / `export_op_sites` | method | public | 以相容 int32 schema 匯出 compact 內部儲存 |
| seam pass-throughs（`set_string_sites`、`set_seam_mask[_consistent]`、`build/commit_half_line_proposal`、`attempt_string_toggle`、`topology_sweep`…） | method | public | 全部轉發給 `off_diag_`（QAQMCOffDiagonalCore） |
| `compute_bond_W_inline()` | static | public | 單 bond 四個 spin state 的 diagonal weights；QAQMC/Renyi/SSE 共用的權重慣例 |

## 物理 / 演算法契約

- Ramp：`delta_at(p)` 為兩段線性（p<M 上坡、p≥M 下坡），表達式即建構
  schedule 的原式 — **bit-identical 是契約**（materialized 陣列已移除）。
- τ open boundary：`p=0` 與 `p=M_total` 的態固定 `|0…0⟩`；cluster update 的
  首尾 segment frozen、不允許 wrap。
- 每個 slot 必為 `-1`（off-diagonal σˣ）/ `1`（diagonal site）/ `2`
  （diagonal bond）之一；diagonal update 是對 diagonal slot 的
  heat-bath 重抽樣（alias 提案 + envelope rejection），**不是** insertion/removal。
- Bond weight 慣例（`compute_bond_W_inline`，三引擎共用）：
  `delta_i = δ/z_eff[i]`、`raw = {0, δ_j, δ_i, −V_ij+δ_i+δ_j}`、
  `cij = max(0,−min raw) + ε·min(|raw₁|,|raw₂|,|raw₃|)`（排除恆為 0 的
  raw0），`W = raw + cij ≥ 0`。
- `state_at_M_` 是 p=M 對稱點的 spin state；`measure_at_midpoint` 用它。
- Off-diagonal seam（若啟用）：per-site worldline closure parity
  `parity(σˣ ops) == seam bit` 是兩端固定邊界強加的守恆量 — 所有 kernel
  保持它；**重設 sector 一律走 `set_seam_mask_consistent`**，raw
  `set_seam_mask` 只能配 `set_op_string` 還原成對記錄的 (ops, mask)。
- Shared model：`QAQMCModelData` 對 chain 是唯讀；同一 process 多個
  engine 共用一份**不改變任何 trajectory**（有測試釘著）。

## 模組輸入

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `N, Omega, delta_min, delta_max, Rb, M, epsilon` | scalar | Hamiltonian / ramp；`M_total = 2M` |
| `seed` | `uint64` | mt19937_64 seed（per-chain） |
| `pos` | `double*  (N, pos_dim)` | 座標 |
| `neighbor_cutoff` | `int` | bond cutoff；`-1` = all-to-all |
| `delta_groups` | `int` | grouped alias 組數（production 慣用 600；必須 > 0） |
| `box, n_box` | `double* (n_box, pos_dim)` | periodic supercell 向量；`n_box=0` = open |
| `model_data` | `shared_ptr<QAQMCModelData>` | 第二建構子：共用既有 model、只建 per-chain state |

## 模組輸出

| 輸出 | Type / Shape | 意義 |
| --- | --- | --- |
| `export_op_types/sites` | `int32[M_total]` | 相容 schema 的 operator string（內部是 int8 / u16‖u32） |
| `MidpointObservables` | struct | density + `Z_l_by_size[g]` + `C_m_l_by_size[g]` |
| `ProfileObservables` | struct | density/Z_l/C_m_l/s_q(6 階)/snapshots/occ-SF(兩套 cell)/M_vbs/M_ss，皆 [·][n_points] |
| `DimerSFSample` | struct | density[n_p]、s_q re/im/abs²（n_p×n_q row-major） |
| `get_memory_breakdown()` | map<string,u64> | 各主要 allocation 的 logical/capacity bytes |
| `time_diag/time_clus/mc_steps` | scalar | phase timing |

## 資料契約

- `op_types` 編碼 `-1/1/2`（見上）；內部 `OpType = int8_t`。
- `op_sites` 內部自動選型：`max_location ≤ 65535` → `uint16_t`
  （N=216 full-bond 落在此），否則 `uint32_t`；恰好配置一個 storage
  vector。對外 API/checkpoint 一律 int32。
- Grouped alias：`alias_prob` + `alias_idx16‖idx32`（`alias_u16` 旗標），
  location kind 由 sampled index 推導，不再存 `loc_kind` 陣列；
  `bond_W_max_all` 已移除（無 hot-path 讀者），保留 `bond_W_rmax_all`。
- Bond event 三種表示（`set_bond_event_storage`）：
  - `packed64`：int64 `[p:32|b:31|endpoint:1]`，p 在高位 ⇒ packed 序 == p 序，
    upper_bound 可直接搜 packed key；
  - `p_bond16`：`uint32 p` + `uint16 b`（6 B/event；**n_bonds > 65535 建構時
    直接拒絕**，不做 silent narrowing）；
  - `p_only32`：只存 `uint32 p`，b/endpoint 在 hot loop 由 op string 重建。
  三種 layout 的 event 順序、RNG 流、逐步 trajectory **exact 相同**。
- `bond_spin_`：per-bond-op spin index 0..3 的 int8 cache。
- `vertex_counts_valid_`：diagonal sweep 填的 count 對 cluster 的 1↔−1
  toggle 是 count-neutral，可跨 step 重用；`set_op_string` 清除。

## 狀態 / 不變量

- `len(op_types) == M_total`，永無 type 0。
- 從 `|0…0⟩` 經全部 `-1` ops 傳播回到 `|0…0⟩`（open-boundary closure）；
  seam 啟用時改為 per-site parity == seam bit。
- `delta_groups_ > 0`；group(p) 在 sweep 內以增量 tracker 計算（無
  per-slice map）。
- 同 build、同 seed、同呼叫序 ⇒ trajectory bit-identical（含三種 event
  layout、含 shared-model 建構子）。換 `-march` 只保證統計等價。

## 整體流程（mc_step）

1. `diagonal_update()`：從 p=0 起走一遍 operator string，用 prefix 傳播
   state；對每個 diagonal slot 依 group alias 表提案（site 直接接受；bond 以
   `W[s]·rmax` envelope rejection 重試至接受）；順路重建 vertex lists 與
   `bond_spin_`、抓 `state_at_M_`、在 `p == m_star` 呼叫 seam hook。
2. `cluster_update()`：逐 site，對其相鄰兩個 single-site op 之間的每個
   internal segment 做 segment Metropolis：走訪該 segment 的 bond events，
   累乘 W(new)/W(old)（**純 ratio product、以 1e±100 重正規化、零/inf 另計
   balance**，hot path 無 `std::log`）；接受則翻 segment（單站 op 1↔−1、
   更新 `bond_spin_`）。首尾 segment frozen（open BC）。
3. `mc_step_profiled` 把 profile 量測 fused 進 diagonal sweep（同一條 state
   trajectory），回傳上一步樣本。

## 邊界情況

- `n_bonds == 0`（例如 cutoff 過小）：n_bonds_pad=1，bond 提案分支不觸發。
- 站點無 single-site op：該站整條 worldline 無 internal segment，cluster
  不動它（open BC 下整條翻轉會改邊界態，禁止）。
- `set_op_string` 驗證 type/site/bond 範圍，非法輸入 throw（防 compact
  儲存被 narrowing 汙染）。
- `set_bond_event_storage("p_bond16")` 在 n_bonds > 65535 時 throw。

## 效能備註（不改變統計結果的設計）

- grouped alias（`delta_groups` 組共用表 + per-slice rejection 修正）取代
  per-slice 表：正確性 exact，記憶體 O(G) 而非 O(M)。
- int8 op types、u16 op sites、AoS AliasEntry、int8 bond_spin、packed
  vertex events：cache/記憶體頻寬優化；production `p_bond16` 建議值
  在大 M 省 15–18% RSS 且速度持平（見 docs/design/cpu_memory.md §1.4）。
- event scratch 的 retained headroom 有界（~12.5% slack 才 shrink），避免
  波動後長期持有 2× capacity。
- 移除 materialized delta schedule：production M=100M 省 1.6 GB/rank。
- `mc_step_profiled` 的 one-step lag 對平衡採樣統計恆等（已有 bit-exact
  對照測試）。

## 驗收標準

- 1D 小系統 vs ED gate（`tests/engines/` 的 vs-ED 測試）通過。
- 同 seed 20-step trajectory 與既有 build bit-identical（memory-layout
  重構時的硬 gate；任何例外需逐項證明 + 測試）。
- `get_delta_schedule()` 與 `delta_at(p)` bitwise 相同。
- checkpoint（RNG + op string）restore 後下一步 bit-exact。

## 測試

- `tests/engines/unit/test_qaqmc_cpu_memory_layout.py`：compact 表示、
  overflow fallback、三種 event layout exact、shared-model 不改 trajectory、
  checkpoint replay、scratch headroom。
- `tests/engines/unit/test_cpu_module_api.py`：module API 面。
- `tests/mpi/unit/test_qaqmc_profile_grid.py`：profile 網格（不再
  materialize 全 ramp）。
- vs-ED / C3 對稱性 / cross-method gates（既有）。

## 相關檔案

- `csrc/cpu/detail/qaqmc_off_diagonal_core.hpp/.cpp` — seam 元件（契約見
  該 header 註解與本 spec「物理契約」節）
- `csrc/cpu/detail/diagonal_observables.hpp` — 與 SSE 共用的 diagonal
  observable 幾何/量測
- `csrc/cpu/bindings/bindings_qaqmc.cpp` + `fragments/*.inc` — Python 面
  （profile/dimer/occ-SF/VBS/CUDA bridge 分片）
- `csrc/cuda/` — device-resident 對應實作（`docs/design/gpu_*.md`）
- `src/engines/qaqmc.py`、`src/engines/qaqmc_cpu_batch.py`、
  `src/mpi/qaqmc_mpi.py`

## 開放問題

- `p_only32` 在 64-rank 頻寬競爭下 slowdown 放大到 ~30%（單核 ~10%）；
  是否保留仍待未來 RAM-bound 需求檢驗。
- diag 階段 61% stall 在 alias 表 random load；剩餘的加速槓桿是
  state-dependent proposal（演算法層變更，需另行驗證）。
