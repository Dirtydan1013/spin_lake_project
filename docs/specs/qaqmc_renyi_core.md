# 規格：csrc/cpu/include/qaqmc_renyi_core.hpp / csrc/cpu/detail/qaqmc_renyi_core.cpp

最後更新：2026-07-17（重寫；含 cut 泛化 `cut_`、Mode::Work 單 bit toggle、
op-string checkpoint、packed channel events、OpenMP coloring cluster）。

## 角色

`QAQMCRenyiEngine` 是 two-replica QAQMC 引擎，用 swap trick 估計 Renyi-2
熵：兩條 replica operator string 在 swap 邊界 `cut_`（預設 M，可由
`set_cut` 移動）之後按 `A_mask` 把 replica 接到對方，形成 channel 結構。
支援三種模式：

- `Mode::PairToggle`：兩個 topology（`set_topology_pair`）之間 Metropolis
  切換，visit counts 給 ratio 估計（`src/tee/qaqmc_renyi_ratio.py`）。
- `Mode::Expanded`：ensemble ladder + `log_g` umbrella bias（expanded
  ensemble reweighting，`src/tee/reweighting.py`）。
- `Mode::Work`：topology/ensemble 更新整個交給外層
  `QAQMCRenyiWorkEngine`；本引擎只提供單 bit toggle 原語
  （`log_weight_ratio_for_toggle` / `apply_single_bit_toggle`）。

## 物件 / 函式

| 名稱 | 種類 | 可見性 | 用途 |
| --- | --- | --- | --- |
| `QAQMCRenyiEngine` | class | public | two-replica 引擎 |
| `Mode` | enum | public | PairToggle / Expanded / Work |
| `ReplicaState` | struct | public | 每 replica 的 op_types/op_sites（int32）/state_at_M |
| `Ensemble` | struct | public | ladder 節點：A_mask + size |
| `BondEvent` / `SiteEvent` | using(uint32) | public | bit-packed channel-space 事件：`[p:30|replica:1|endpoint:1]` / `[p:31|replica:1]`；p 高位 ⇒ 排序即 p 序 |
| `mc_step()` / `run_steps(n)` | method | public | diagonal + cluster（+ 依 mode 的 topology/ensemble/indicator 更新） |
| `set_A_mask` | method | public | 設定 swap 區域（PairToggle/Expanded 初始化路徑） |
| `set_cut(m_star)` | method | public | 移動 swap 邊界至 slice m_star ∈ [0, M_total]；**必須在 op string 於新舊映射下皆有效時呼叫**（實務：建構後立刻，或經 work engine 的 vacuum-reset） |
| `set_topology_pair(A_k, A_k+1, diff_site)` | method | public | PairToggle 的兩個 topology |
| `topology_toggle()` / `ensemble_switch()` | method | public | 依 mode 的 Metropolis 拓撲/ensemble 切換 |
| `set_ensemble_ladder(masks, neighbors, init)` / `set_log_g` | method | public | Expanded 模式 ladder |
| `visit/transition/collection counts` 系列 | method | public | 估計器統計（含 reset） |
| `set_mode(Mode)` | method | public | 切模式 |
| `set_A_mask_for_work(mask)` | method | public | Work 模式換 mask：**會 reproject 全部 site ops**（diagonal update 保留 offdiag types，不 reproject 會殘留舊邊界的 stale −1） |
| `log_weight_ratio_for_toggle(site)` | method | public | log[Z(A^site)/Z(A)]（現行 op string 下） |
| `apply_single_bit_toggle(site)` | method | public | `A_mask[site] ^= 1` + reproject 受影響 site ops（p ≥ cut）；Metropolis 判準由 caller 負責 |
| `save/restore_op_string_checkpoint()` | method | public | 兩 replica op string 的 checkpoint（**mask 不入 checkpoint**：caller 負責先把 A_mask 調回存檔時的 mask） |
| `get_checkpoint_op_types/sites(replica)` | method | public | warm-start 匯出 |
| `set_indicator_site` / `get_indicator_avg` | method | public | 單站 indicator（診斷） |
| `set_replica_op_string` / `recompute_midpoint_states` | method | public | 外部還原 op string（還原後必須 recompute） |
| `get_site_paths(site)` | method | public | 4 條 channel path（測試/診斷用） |
| `log_weight_ratio_for_site(site, from, to)` | method | public | PairToggle 的權重比 |

## 物理 / 演算法契約

- Channel 映射（核心不變量）：`p < cut_` 時 channel == replica；
  `p ≥ cut_` 時 `A_mask[site]==1` 的站 channel 與 replica 對調
  （`replica_for` / `channel_for_actual`）。swap trick 的 Z[A] 即此
  聯通結構下的配分函數。
- δ schedule、bond weight 慣例與 single-replica 引擎完全相同
  （`delta_at` bit-identical 契約、`compute_bond_W_inline` 轉呼叫
  QAQMCEngine 版本）。
- `log_g` 是 umbrella bias，不是 entropy；估計器在 Python 層由 visit
  counts / 轉移統計組裝。
- Work 模式的 toggle 只在 `p ≥ cut_` 影響 site ops；toggle 後受影響站的
  site ops 必須 reproject 到新 channel（`apply_single_bit_toggle` 內建）。
- 兩 replica 各有獨立 `mt19937_64`（`rngs_[2]`）；cluster update 另有
  per-site RNG streams（見效能節）— **RNG 流的配置是重現性契約的一部分**。

## 模組輸入 / 輸出

輸入同 single-replica（N/Ω/δ範圍/Rb/M/ε/seed/pos/cutoff/delta_groups/box）。
主要輸出：

| 輸出 | Type / Shape | 意義 |
| --- | --- | --- |
| `get_op_types/sites(replica)` | `int32[M_total]` | 各 replica operator string（此引擎**未** compact 化，仍 int32） |
| `get_visit_counts()` | `int64[2]` | PairToggle 兩 topology 的訪問數 |
| `get_visit_counts_ext()` / `get_transition_counts()` / `get_collection_counts()` | vectors | Expanded ladder 統計 |
| `get_operator_counts()` | `int64[3]` | (-1/1/2) op 計數 |
| `diag_*` 計數器、`time_*` | scalar | 提案/接受診斷與 phase timing |

## 資料契約

- Channel-space vertex lists：flat index = `channel * N + site`；
  `ch_site_bond_list_`（`BondEvent` uint32）與 `ch_site_op_list_`
  （`SiteEvent`）按 p 升冪填入。16-byte struct 版本已廢除
  （production M_total≈4.5M 時 ~9M events × 16B 爆 L3）。
- `bond_spin_by_replica_`：int8 `[2 * M_total]`。
- `W_by_op_`：per-bond-op raw-W 快取 `[(replica*M_total+p)*4 + w]`，
  cluster 開始時由 `build_bond_spins_from_ops()` 填；segment Metropolis
  累乘 raw ratio product（無 log）。
- op-string checkpoint（`op_ckpt_*`）只存兩 replica 的 types/sites；
  **不存 mask**。

## 狀態 / 不變量

- `cut_ ∈ [0, M_total]`，預設 M；對稱點 state 抓取與 A-mask swap 都以
  `cut_` 為界。
- 任何時刻兩 replica 的 op string 在現行 (A_mask, cut) 映射下 channel
  path 閉合（`recompute_midpoint_states` 重建快取後必須一致）。
- Expanded ladder 的相鄰 mask 恰差一個 site。
- `set_cut` 之後、op string 未重設前不得繼續 sampling（work engine 的
  `set_cut` 會 vacuum-reset 並作廢 checkpoint）。

## 整體流程（mc_step）

1. `diagonal_update()`：兩 replica 各掃一遍；channel 結構只影響 p ≥ cut 的
   site-op channel 歸屬；grouped alias（AoS `AliasEntry`）提案 + envelope
   rejection；順路更新 midpoint states。
2. `cluster_update()`：channel-space —
   `build_channel_vertex_lists()` + `build_bond_spins_from_ops()`，然後按
   **greedy bond-graph coloring** 分色：同色站彼此無 bond，OpenMP 平行跑
   per-(channel,site) 的 segment Metropolis 不會在 bond-spin 寫入上 race；
   每站用自己的 `site_rngs_[site]` stream。
3. 依 mode：PairToggle → `topology_toggle()`；Expanded →
   `ensemble_switch()`；Work → 什麼都不做（外層驅動）。
4. indicator（若設定）累積。

## 邊界情況

- `A_mask` 全 0：兩 replica 完全獨立（Z = Z₁Z₂）；全 1：p ≥ cut 完全對調。
- `set_A_mask_for_work` / `apply_single_bit_toggle` 之外直接改 mask =
  未定義行為（stale site ops）。
- `set_replica_op_string` 後未 `recompute_midpoint_states` → 快取不一致。
- cut 在端點（0 或 M_total）：channel 映射退化為全段 swap / 全段獨立，
  compact-theorem 測試涵蓋。

## 效能備註

- coloring + per-site RNG 使 cluster 可 OpenMP 平行；色組在建構時算一次。
- packed uint32 events、int8 bond_spin、raw-W 快取：cluster
  build/sweep 的頻寬優化（perf 記錄 6.9× cluster 加速的一部分）。
- grouped alias 同 single-replica；`delta_groups=0` 走舊 per-slice 表。

## 驗收標準 / 測試

- 4×4 m=1 KP 上 ratio 與 expanded 兩法互證（cross-method gate）。
- physical-bound / ladder-consistency / C3 對稱性 gates。
- `tests/engines/unit/test_qaqmc_renyi_compact_theorem.py`：
  compact 邊界公式 == 暴力 channel-path 枚舉；單站 toggle 零 ratio 不需
  reprojection 的定理；interacting bond weights 在 toggle 下 exact 不變；
  端點 cut。
- Work-mode 的 K-independence / fixed-λ sector gates 在 work-engine spec。

## 相關檔案

- `csrc/cpu/include/qaqmc_renyi_work_core.hpp` — 外層 work 引擎（Mode::Work 的唯一 driver）
- `csrc/cpu/bindings/bindings_renyi.cpp`
- `src/tee/qaqmc_renyi_ratio.py`、`src/tee/reweighting.py` — 兩種 S₂ 估計器
- `src/mpi/kp_tee_ratio_mpi.py`、`src/mpi/kp_tee_expanded_mpi.py`
- `csrc/cuda/include/renyi.cuh` — device-resident 對應

## 開放問題

- Renyi 引擎的 op string 尚未 compact 化（int8/u16）— PR #5 第一階段
  範圍只含 standard engine；若 Renyi production 記憶體成為瓶頸，同樣的
  型別窄化可以套用（需重跑 bit-exact gate）。
