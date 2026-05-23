# 規格：csrc/qaqmc_renyi_core.hpp / csrc/qaqmc_renyi_core.cpp

## 角色
`QAQMCRenyiEngine` 是 C++ 端的 two-replica QAQMC Renyi 引擎。它負責在固定長度的 QAQMC `operator string` 上模擬兩個 replicas，並依照目前的 Renyi topology mask 決定兩個 replicas 在 imaginary-time 中點之後如何交換 channel。

這個檔案支援兩種模式：

- `PairToggle`: 在兩個只差一個 site 的 topology 之間切換，用於 ratio estimator。
- `Expanded`: 在一組 ensemble masks 形成的 graph/ladder 上切換，用於 expanded ensemble reweighting。

Python 端主要透過 `src/engines/qaqmc_renyi.py` 包裝這個引擎，再由 `src/tee/qaqmc_renyi_ratio.py`、`src/tee/reweighting.py` 和 MPI driver 使用。

## 物件 / 函式

| 名稱 | 種類 | 可見性 | 用途 |
| --- | --- | --- | --- |
| `QAQMCRenyiEngine` | class | public | Two-replica QAQMC Renyi / expanded ensemble 引擎。 |
| `Mode` | enum | public nested | `PairToggle` 或 `Expanded`。 |
| `ReplicaState` | struct | public nested | 單一 replica 的 `op_types`、`op_sites`、`state_at_M`。 |
| `Ensemble` | struct | public nested | 一個 expanded ensemble sector 的 `A_mask` 與 region size。 |
| `OffdiagPaths` | struct | public nested | 對某個 mask 建出的 offdiagonal path 壓縮表示。 |
| `BondEvent` / `SiteEvent` | type alias | public | Packed `uint32_t` channel-space vertex event。 |
| `mc_step()` | method | public | 執行一次完整的 diagonal update、cluster update、topology/ensemble switch。 |
| `run_steps(n_steps)` | method | public | 重複呼叫 `mc_step()`。 |
| `set_A_mask(mask)` | method | public | 設定固定 Renyi mask。 |
| `set_topology_pair(A_k, A_kp1, diff_site)` | method | public | 設定 ratio estimator 使用的兩個 topology。 |
| `set_ensemble_ladder(masks, neighbors, initial_ensemble)` | method | public | 設定 expanded ensemble masks 與 proposal graph neighbors。 |
| `set_log_g(log_g)` | method | public | 設定 expanded ensemble umbrella weights。 |
| `topology_toggle()` | method | public | 在 `PairToggle` 模式下嘗試切換 topology。 |
| `ensemble_switch()` | method | public | 在 `Expanded` 模式下嘗試切換 ensemble sector。 |
| `diagonal_update()` | method | private | 在目前 Renyi mask 下，對兩個 replicas 的 diagonal slots 重新抽樣 site/bond operators。 |
| `cluster_update()` | method | private | 在 current topology 的 channel-space vertex lists 上做 segment Metropolis update。 |
| `build_channel_vertex_lists()` | method | private | 依照目前 `A_mask_` 建立 channel/site vertex lists。 |
| `build_bond_spins_from_ops()` | method | private | 建立 bond-spin cache 和 per-bond-op log-weight cache，供 cluster sweep 使用。 |
| `build_offdiag_paths()` | method | private | 對指定 topology mask 建立 offdiagonal paths，用於 topology weight ratio 和 reproject。 |
| `reproject_site_ops_for_mask_with_paths()` | method | private | Topology mask 改變後，將 site operators 重新投影到新 mask 的合法表示。 |
| `log_weight_ratio_for_site(site, from, to)` | method | public | 回傳某 site 在兩個 topology 間的 local log weight ratio。 |
| `get_visit_counts()` | accessor | public | `PairToggle` 的 topology visit counts。 |
| `get_visit_counts_ext()` | accessor | public | Expanded ensemble 的 sector visit counts。 |
| `get_transition_counts()` | accessor | public | Expanded ensemble 實際 accepted/rejected transition counts。 |
| `get_collection_counts()` | accessor | public | Expanded ensemble virtual-transition collection estimator counts。 |
| `get_time_*()` / `get_diag_*()` | accessor | public | Profiling 與 proposal diagnostics。 |

## 物理 / 演算法契約

- 引擎永遠只維護兩個 physical replicas，不是每個 ensemble 各自一組 replicas。
- `A_mask[site] == 1` 表示該 site 屬於 Renyi swap region。
- Imaginary time 總長度是 `M_total = 2 * M`。對 `p < M`，channel 與 replica 相同；對 `p >= M`，若 `A_mask[site] == 1`，channel 與 replica 對調。
- Topology mask 是 boundary condition，不是事後分類標籤。`diagonal_update()`、`cluster_update()`、topology/ensemble switch 都必須用目前的 `A_mask_` 解讀 channel/replica。
- `log_g` 是 expanded ensemble 的 umbrella bias weight，不是 entropy，也不是 `log Z`。
- `collection_count` 儲存 virtual transition estimator。Python 端會先 row-normalize，再取 stationary distribution，最後用 `log(pi) - log_g` 得到相對 `log Z`。
- `PairToggle` 和 `Expanded` 都是 topology switch/reweighting。`PairToggle` 只在兩個 topology 間估計 ratio；`Expanded` 在一個 ensemble graph 上同時估計多個 sectors 的相對 partition functions。

## 輸入

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `N` | `int` | Site 數。 |
| `Omega` | `double` | Rabi frequency。 |
| `delta_min`, `delta_max` | `double` | QAQMC ramp 的 detuning 範圍。 |
| `Rb` | `double` | Rydberg blockade radius parameter。 |
| `M` | `int` | Half schedule length；引擎使用 `M_total = 2M`。 |
| `epsilon` | `double` | Diagonal bond weight shift 的 safety margin。 |
| `seed` | `uint64_t` | RNG seed；兩個 replicas 使用不同 stream。 |
| `pos` | `double*`, shape `(N, pos_dim)` | Site coordinates。 |
| `neighbor_cutoff` | `int` | Rydberg interaction bond cutoff；`-1` 代表不截斷。 |
| `delta_groups` | `int` | Grouped alias table 數量；`0` 代表每個 slice 獨立 table。 |
| `A_mask` | `uint8_t[N]` | Renyi swap region mask。 |
| `masks` | `vector<vector<uint8_t>>`, each length `N` | Expanded ensemble sectors。 |
| `neighbors` | `vector<vector<int>>` | Expanded ensemble proposal graph。 |
| `log_g` | `vector<double>`, length `n_ensembles` | Umbrella weights。 |

## 輸出

| 輸出 | Type / Shape | 意義 |
| --- | --- | --- |
| `op_types(replica)` | `int32[M_total]` | 單一 replica 的 operator type string。 |
| `op_sites(replica)` | `int32[M_total]` | 單一 replica 的 operator site/bond index string。 |
| `state_at_M(replica)` | `int32[N]` | 從 operator string 重建出的 midpoint spin state。 |
| `visit_count` | `int64[2]` | `PairToggle` topology visits。 |
| `visit_count_ext` | `int64[n_ensembles]` | Expanded ensemble visits。 |
| `transition_count` | `int64[n_ensembles, n_ensembles]` flattened row-major | 實際 sampled ensemble transitions。 |
| `collection_count` | `double[n_ensembles, n_ensembles]` flattened row-major | Collection estimator 使用的 virtual transition weights。 |
| `time_diag`, `time_clus_build`, `time_clus_sweep`, `time_topology`, `time_ensemble` | `double` seconds | Profiling timers。 |
| `diag_update_slots`, `diag_*_proposals`, `diag_bond_accepts` | `int64` | Diagonal update proposal diagnostics。 |

## 資料契約

- `op_types[p]` 的編碼沿用 QAQMC core convention：
  - `-1`: offdiagonal spin flip operator。
  - `1`: site diagonal operator。
  - `2`: bond diagonal operator。
- `op_sites[p]` 的意義由 `op_types[p]` 決定：
  - `-1` 和 `1`: site index。
  - `2`: `vij_.bond_sites_flat` 裡的 bond index。
- `vij_.bond_sites_flat` 是 row-major `(n_bonds, 2)`，flattened 後形如 `[si0, sj0, si1, sj1, ...]`。
- Expanded `transition_count` 和 `collection_count` 是 row-major flattened matrix，index 是 `from * n_ens + to`。
- `BondEvent` layout 是 `[p:30 bits][replica:1 bit][endpoint:1 bit]`。
- `SiteEvent` layout 是 `[p:31 bits][replica:1 bit]`。
- Channel-space site index 是 `channel * N + site`，其中 `channel in {0, 1}`。
- Grouped alias tables 是 proposal envelopes。正確性要求 grouped bond envelope weight 對 group 中每個 slice 都至少等於 true slice weight。

## 狀態 / 不變量

- `M_total_ == 2 * M_`。
- `replicas_[0]` 和 `replicas_[1]` 的 operator strings 長度都必須是 `M_total_`。
- 傳給 `set_A_mask`、`set_topology_pair`、`set_ensemble_ladder` 的所有 masks 長度都必須是 `N_`。
- 在 `Expanded` 模式下，`A_mask_ == ensembles_[cur_ens_].A_mask`。
- 在 `PairToggle` 模式下，`A_mask_ == A_masks_[cur_topology_]`。
- `set_log_g()` 後，`len(log_g_) == len(ensembles_)`。
- `set_ensemble_ladder()` 會重設 `visit_count_ext_`、`transition_count_`、`collection_count_` 的大小與內容。
- `ens_neighbors_` 裡的 neighbor indices 都必須是合法 ensemble indices。
- 若相鄰 masks 只差一個 site，`ensemble_switch()` 可以只 reproject 該 site；否則必須 reproject 所有改變的 topology paths。
- Cluster update 或外部 operator-string restoration 後，必須呼叫 `recompute_midpoint_states_from_ops()`，讓 `state_at_M` 保持一致。

## 行為

1. 建構子建立 Rydberg interaction data、detuning schedule、初始 two-replica operator strings、alias tables、cluster update 的 color groups，以及 midpoint states。
2. `set_topology_pair()` 將引擎切到 `PairToggle`，儲存兩個 masks，記錄 `diff_site`，並重設 pair visit counters。
3. `set_ensemble_ladder()` 將引擎切到 `Expanded`，儲存所有 masks 和 neighbor lists，初始化 `cur_ens`，重設 expanded counters，並把 operator strings reproject 到初始 mask。
4. `diagonal_update()` 掃過每個 time slice。既有 diagonal slots 會從 per-slice alias table 或 grouped alias table 重新抽樣。Bond proposals 透過 true bond weight 除以 proposal envelope 做 rejection correction。
5. `cluster_update()` 在目前 `A_mask_` 下建立 channel-space vertex lists，產生 bond-spin 和 log-weight caches，接著對每個 site/channel 做 segment Metropolis updates。碰到 imaginary-time endpoints 的 boundary segments 不會 wrap。
6. 在 `PairToggle` 中，`topology_toggle()` 會用 `diff_site` 評估目前 configuration 在兩個 masks 下的 weight，並用 Barker-style probability `ratio / (1 + ratio)` 接受切換。
7. 在 `Expanded` 中，`ensemble_switch()` 會評估目前 ensemble 的所有 neighbor proposals，將 virtual transition weights 加到 `collection_count`，再均勻抽一個 neighbor，並以 Metropolis probability 接受。
8. `mc_step()` 一定先執行 diagonal update 和 cluster update，接著依模式執行 topology toggle 或 expanded ensemble switch，最後累積 indicator statistics 和 timers。

Expanded ensemble 從 `i -> j` 的 proposal acceptance：

```text
log_a = log_weight_ratio(i -> j)
      + log_g[j] - log_g[i]
      + log(degree(i) / degree(j))

accept_prob = min(1, exp(log_a))
```

Collection estimator 記錄 virtual transition weight：

```text
collection_count[i, j] += propose_prob(i -> j) * accept_prob(i -> j)
collection_count[i, i] += 1 - sum_{j neighbor of i} propose_prob(i -> j) * accept_prob(i -> j)
```

## 函數規格

### `QAQMCRenyiEngine::QAQMCRenyiEngine(...)`

**種類：** constructor  
**可見性：** public

**用途**  
建立 two-replica QAQMC Renyi engine，初始化兩個 replicas、detuning schedule、alias tables、cluster scratch arrays 和 topology state。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| physical parameters | scalars | `N`, `Omega`, `delta_min`, `delta_max`, `Rb`, `M`, `epsilon`。 |
| `pos`, `neighbor_cutoff` | array/scalar | 建立 Rydberg interaction graph。 |
| `seed` | `uint64_t` | 初始化兩個 replica RNG streams。 |
| `delta_groups` | `int` | 控制 grouped alias proposal table。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `replicas_` | two `ReplicaState`s | 兩個 physical replicas 的 operator strings。 |
| `delta_sched_`, `vij_` | arrays/struct | Ramp schedule 和 interaction graph。 |
| `alias_` 或 `grp_alias_` | structs | Diagonal proposal tables。 |
| `A_mask_`, `mode_` | state | 初始 topology state。 |

**演算法流程**

1. 建立 single-replica QAQMC 所需的 schedule、bonds、proposal tables。
2. 初始化兩個 replicas 的 operator strings 和 midpoint states。
3. 初始化 topology/mode state、timers、diagnostic counters。
4. 建立 OpenMP cluster update 需要的 site coloring 和 scratch arrays。

**邊界情況**

- `delta_groups > 0` 時使用 grouped proposal envelopes。

**不變量**

- 兩個 replicas 的 operator string 長度都等於 `M_total_`。

**測試**

- Constructor smoke test：檢查兩個 replicas、mask 長度、table shapes。

### `QAQMCRenyiEngine::set_A_mask(mask)`

**種類：** method  
**可見性：** public

**用途**  
設定固定 Renyi swap mask，供非 expanded / 非 pair setup 的基本 Renyi topology 使用。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `mask` | `uint8_t[N]` | Renyi swap region。 |
| `len` | `int` | 必須等於 `N_`。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `A_mask_` | `uint8_t[N]` | Current topology mask。 |

**演算法流程**

1. 檢查 mask 長度。
2. 複製 mask 到 `A_mask_`。
3. Reproject operator strings 並重建 midpoint states。

**邊界情況**

- 長度不符時 raise。

**不變量**

- `A_mask_.size() == N_`。

**測試**

- 設定人工 mask 後檢查 `get_A_mask()` 和 `state_at_M` consistency。

### `QAQMCRenyiEngine::set_topology_pair(A_k, A_kp1, diff_site)`

**種類：** method  
**可見性：** public

**用途**  
設定 ratio estimator 使用的兩個 topology，並切換到 `PairToggle` mode。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `A_k`, `A_kp1` | `uint8_t[N]` | 兩個 topology masks。 |
| `diff_site` | `int` | 兩個 masks 差異的 site。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `A_masks_`, `A_mask_` | masks | PairToggle 的兩個 masks 和 current mask。 |
| `cur_topology_`, `diff_site_` | scalars | 目前 topology index 和差異 site。 |
| `visit_count_` | `int64[2]` | 重設 pair visit counters。 |

**演算法流程**

1. 檢查 masks 長度和 `diff_site`。
2. 儲存兩個 topology masks。
3. 設定 `mode_ = PairToggle` 和初始 topology。
4. Reproject operator strings 到 current mask。

**邊界情況**

- `diff_site` 不合法時應 raise 或使 toggle 無效。

**不變量**

- `A_mask_ == A_masks_[cur_topology_]`。

**測試**

- 單 site 差異 masks 的 setup 和 visit counter reset。

### `QAQMCRenyiEngine::set_ensemble_ladder(masks, neighbors, initial_ensemble)`

**種類：** method  
**可見性：** public

**用途**  
設定 expanded ensemble 的 sectors 和 proposal graph，並切換到 `Expanded` mode。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `masks` | `vector<vector<uint8_t>>` | 每個 ensemble sector 的 topology mask。 |
| `neighbors` | `vector<vector<int>>` | Proposal graph adjacency。 |
| `initial_ensemble` | `int` | 初始 sector index。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `ensembles_`, `ens_neighbors_` | vectors | Expanded ensemble graph。 |
| `cur_ens_`, `A_mask_`, `mode_` | state | Current sector 和 current mask。 |
| `visit_count_ext_`, `transition_count_`, `collection_count_`, `log_g_` | arrays | 重設 expanded ensemble counters/weights。 |

**演算法流程**

1. 檢查 masks 非空、masks/neighbors 數量一致、indices 合法。
2. 計算每個 ensemble 的 region size。
3. 設定 `cur_ens_ = initial_ensemble` 和 current `A_mask_`。
4. 初始化 counters 和 `log_g_`。
5. 依初始 mask reproject site operators 並重建 midpoint states。

**邊界情況**

- 空 graph、mask 長度錯、neighbor index out of range 都會 raise。

**不變量**

- `A_mask_ == ensembles_[cur_ens_].A_mask`。
- Counter arrays 大小等於 `n_ensembles` 或 `n_ensembles^2`。

**測試**

- Three-node ladder setup：檢查 counters shape、current mask、neighbor validation。

### `QAQMCRenyiEngine::set_log_g(log_g)`

**種類：** method  
**可見性：** public

**用途**  
設定 expanded ensemble umbrella bias weights。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `log_g` | `vector<double>` | 長度必須等於 ensemble 數量。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `log_g_` | `vector<double>` | Current umbrella weights。 |

**演算法流程**

1. 確認已呼叫 `set_ensemble_ladder()`。
2. 檢查 `log_g.size() == ensembles_.size()`。
3. 複製到 `log_g_`。

**邊界情況**

- 未設定 ladder 或長度不符時 raise。

**不變量**

- `len(log_g_) == len(ensembles_)`。

**測試**

- 長度 mismatch validation。

### `QAQMCRenyiEngine::mc_step()`

**種類：** method  
**可見性：** public

**用途**  
執行一次完整 two-replica Renyi QAQMC Monte Carlo step。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| internal state | engine fields | 目前 replicas、mask、mode、RNG、tables。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `replicas_` | arrays | 更新後的 two-replica operator strings。 |
| `cur_topology_` 或 `cur_ens_` | scalar | 可能切換 topology/sector。 |
| visit/transition/collection counters | arrays | 依 mode 更新。 |
| timers, indicators | diagnostics | 更新 profiling 和 indicator statistics。 |

**演算法流程**

1. 呼叫 `diagonal_update()`。
2. 呼叫 `cluster_update()`。
3. `Expanded` mode 呼叫 `ensemble_switch()` 並累積 `visit_count_ext_`。
4. 否則呼叫 `topology_toggle()` 並累積 `visit_count_`。
5. 更新 indicator 和 timers。

**邊界情況**

- 若沒有 expanded ladder，`ensemble_switch()` 會直接返回。

**不變量**

- `A_mask_` 必須和 current topology/ensemble 一致。

**測試**

- PairToggle/Expanded smoke tests，檢查 counters 更新。

### `QAQMCRenyiEngine::diagonal_update()`

**種類：** method  
**可見性：** private

**用途**  
在目前 Renyi topology mask 下，對兩個 replicas 的 diagonal slots 重新抽樣。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `replicas_` | two operator strings | 兩個 physical replicas。 |
| `A_mask_` | `uint8_t[N]` | Current topology boundary condition。 |
| `alias_` / `grp_alias_`, `delta_sched_`, `vij_` | tables/arrays | Proposal 和 true bond weights。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `replicas_[r].op_types/sites` | arrays | 更新 diagonal slots。 |
| proposal diagnostics | counters | `diag_*` counters。 |

**演算法流程**

1. 建立 channel-space spin state。
2. 對每個 time slice 和 replica 檢查 operator type。
3. Offdiagonal operator 只在 slice 結尾 propagate channel spin。
4. Diagonal slot 從 alias/grouped alias proposal 新 site/bond。
5. Bond proposal 依目前 mask 將 physical replica 映射到 channel，計算 true `w_actual`，再用 envelope rejection correction 接受。

**邊界情況**

- `delta_groups_ > 0` 時 proposal 來自 group table。
- Bond proposal 若 `w_max <= 0` 會被拒絕並重抽。

**不變量**

- 更新時必須使用 current `A_mask_` 的 channel/replica mapping。

**測試**

- 固定 mask 下和 single-replica QAQMC 的 local proposal consistency。

### `QAQMCRenyiEngine::cluster_update()`

**種類：** method  
**可見性：** private

**用途**  
在 current topology 的 channel-space worldlines 上做 segment Metropolis update。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `replicas_`, `A_mask_` | arrays/mask | Two-replica operator strings 和 boundary condition。 |
| `color_groups_` | vector | OpenMP parallel sweep 的 site coloring。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `replicas_[r].op_types` | arrays | Single-site ops 在 `1` 和 `-1` 間切換。 |
| `state_at_M` | arrays | Cluster update 後重建 midpoint states。 |
| `time_clus_build_`, `time_clus_sweep_` | diagnostics | Build/sweep timers。 |

**演算法流程**

1. 呼叫 `build_channel_vertex_lists()`。
2. 呼叫 `build_bond_spins_from_ops()` 建立 bond-spin/log-weight caches。
3. 依 color groups 平行處理 sites。
4. 對每個 channel/site 的 internal segments 做 Metropolis flip。
5. 根據 segment flip parity 切換 single-site op type。
6. 重建 midpoint states。

**邊界情況**

- `n_sops < 2` 的 channel/site 沒有 internal segment 可翻。
- Boundary-touching segments 不 proposal。

**不變量**

- Cluster update 必須使用 current `A_mask_` 的 channel-space vertex lists。

**測試**

- 人工 mask/operator string 檢查 channel mapping 和 no-wrap behavior。

### `QAQMCRenyiEngine::ensemble_switch()`

**種類：** method  
**可見性：** public

**用途**  
在 expanded ensemble graph 上 proposal sector switch，並更新 transition/collection statistics。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `cur_ens_`, `ensembles_`, `ens_neighbors_` | graph state | Current sector 和 proposal neighbors。 |
| `log_g_` | `double[n_ens]` | Umbrella weights。 |
| current operator strings | replicas | 用來評估 topology weight ratio。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `cur_ens_`, `A_mask_` | state | 可能切換到 proposed sector。 |
| `transition_count_` | matrix | 實際 sampled transition。 |
| `collection_count_` | matrix | Virtual transition estimator。 |

**演算法流程**

1. 取得 current ensemble 的 neighbor list。
2. 對每個 neighbor 計算 affected site(s) 的 `log_weight_ratio`。
3. 加上 `log_g[to] - log_g[from]` 和 proposal degree correction。
4. 將每個 virtual proposal 的 `propose_prob * accept_prob` 加到 `collection_count_`。
5. 隨機抽一個 neighbor，依 Metropolis probability 接受或拒絕。
6. 若接受，更新 `cur_ens_`、`A_mask_`，並 reproject operator strings。
7. 更新 `transition_count_`。

**邊界情況**

- 沒有 neighbors 時記錄 self transition/self collection。
- Proposed topology weight invalid 時 acceptance 為 0。
- Current invalid、proposed valid 時用 large log ratio 強制接受。

**不變量**

- 接受後 `A_mask_ == ensembles_[cur_ens_].A_mask`。
- `collection_count_` row contribution 應接近一次 virtual transition normalization。

**測試**

- Three-node ladder test：檢查 transition/collection shapes、self weight、degree correction。

### `QAQMCRenyiEngine::topology_toggle()`

**種類：** method  
**可見性：** public

**用途**  
在 `PairToggle` mode 下於兩個 topology masks 之間切換。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `A_masks_`, `cur_topology_`, `diff_site_` | topology state | PairToggle 的兩個 masks 和差異 site。 |
| current operator strings | replicas | 評估兩個 topology 的 weight。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `cur_topology_`, `A_mask_` | state | 可能切換到另一個 topology。 |
| site operators | arrays | 接受後 reproject affected site。 |

**演算法流程**

1. 若 `diff_site_ < 0` 或兩個 masks 相同，直接返回。
2. 對 current/proposed mask 建立 offdiag paths。
3. 計算 `diff_site_` 的 `log_to - log_from`。
4. 用 Barker-style probability `ratio / (1 + ratio)` 接受。
5. 接受後更新 topology 和 `A_mask_`，並 reproject affected site。

**邊界情況**

- Masks 相同時不做任何事。

**不變量**

- 接受後 `A_mask_ == A_masks_[cur_topology_]`。

**測試**

- Pair ratio regression test。

### `QAQMCRenyiEngine::log_weight_ratio_for_site(site, from_topology, to_topology)`

**種類：** method  
**可見性：** public

**用途**  
計算某個 site 在兩個 topology masks 間的 local log weight ratio。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `site` | `int` | Affected site。 |
| `from_topology`, `to_topology` | `int` | Topology indices。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| return value | `double` | `log W_to - log W_from`。 |

**演算法流程**

1. 取出兩個 topology masks。
2. 建立對應 offdiag paths。
3. 分別計算該 site 在兩個 masks 下的 log weight。
4. 回傳差值。

**邊界情況**

- 若某 topology weight invalid，會反映為非常小的 log weight。

**不變量**

- 不修改 Markov chain state。

**測試**

- 對只差一個 site 的 masks 比對 `ensemble_switch()` 內部 ratio。

### `QAQMCRenyiEngine::build_offdiag_paths(mask, paths)`

**種類：** method  
**可見性：** private

**用途**  
依照 topology mask 建立每個 channel/site 的 offdiagonal path 表示，用於 weight ratio 和 reproject。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `mask` | `uint8_t[N]` | 要分析的 topology mask。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `paths` | `OffdiagPaths` | Compressed path data。 |

**演算法流程**

1. 依 mask 的 channel/replica mapping 掃過 offdiagonal events。
2. 對每個 channel/site 累積 event count。
3. 建立 head/list 壓縮表示。

**邊界情況**

- 某些 channel/site 可以沒有 offdiagonal events。

**不變量**

- Path representation 必須和 `replica_for_with_mask()` / `channel_for_actual_with_mask()` 一致。

**測試**

- 人工 offdiagonal string 下檢查 path count/head/list。

### `QAQMCRenyiEngine::reproject_site_ops_for_mask_with_paths(...)`

**種類：** method  
**可見性：** private

**用途**  
Topology mask 改變後，將 site diagonal/offdiagonal operators 重新投影到新 mask 的合法 channel/replica 表示。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `mask` | `uint8_t[N]` | New topology mask。 |
| `paths` | `OffdiagPaths` | New mask 的 offdiag paths。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `replicas_[r].op_types/sites` | arrays | Reproject 後的 site operators。 |

**演算法流程**

1. 依新 mask 重新解讀 channel/replica paths。
2. 對受影響 site 重新分配 site operators。
3. 保持 operator string 在新 topology 下合法。

**邊界情況**

- 若只有一個 `diff_site`，可使用 site-local reproject path 降低成本。

**不變量**

- Reproject 後必須能重建合法 midpoint states。

**測試**

- Mask switch 後檢查 `recompute_midpoint_states_from_ops()` 不產生不一致。

## 邊界情況

- `set_ensemble_ladder()` 會在 masks 為空、mask count 和 neighbor count 不同、`initial_ensemble` 超出範圍、任一 mask 長度不等於 `N`、或 neighbor index 不合法時 raise。
- `set_log_g()` 若在 `set_ensemble_ladder()` 前呼叫，或輸入長度不等於 ensemble 數量，會 raise。
- `set_indicator_site()` 會在 site index 不合法時 raise。
- 若某個 ensemble 沒有 neighbors，`ensemble_switch()` 會記錄 self transition 和 self collection weight。
- 若 proposed topology 對 affected path 的 weight 為零或不合法，接受率為零。
- 若 current topology 的 path weight 不合法，但 proposed topology 合法，move 會用非常大的 log ratio 強制接受。
- 對 grouped alias tables，如果 envelope 太小，rejection correction 就不再 valid；因此 grouped table builder 必須維持 envelope invariant。

## 效能備註

- 引擎只維護兩個 replicas，但用 channel-space vertex lists 讓 cluster update 可以依照目前 Renyi boundary condition 運作，不需要為每個 ensemble sector 複製一組 replicas。
- `BondEvent` 和 `SiteEvent` pack 成 `uint32_t`，降低 cluster build/sweep 的 memory bandwidth。
- `build_bond_spins_from_ops()` 把 sequential channel-state propagation 和 parallel log-weight cache construction 分開。
- OpenMP cluster sweep 使用 greedy bond-graph coloring：同色 sites 不共享 bond，所以 per-site segment updates 可以平行執行，不會 race on bond-spin cache writes。
- `delta_groups > 0` 把 alias-table memory 從 per-slice tables 降到 per-group proposal envelopes。由於 bond proposals 會用 `w_actual / w_envelope` 修正，理論上不應改變 sampled distribution。
- `time_clus_build` 和 `time_clus_sweep` 分開紀錄，因為 build bandwidth/log-cache cost 和 segment Metropolis cost 的 scaling 不同。

## 驗收標準

- `PairToggle` 模式在小系統上應重現 independent ratio estimator tests 的 pair ratio。
- `Expanded` 模式在 ensemble graph 和 runtime 足夠時，應產生 positive visits 和 connected collection transition matrix。
- 固定 seed 和參數時，在相同 binary 和 OpenMP configuration 下，`run_steps(n)` 應可重現。
- 設定 `delta_groups > 0` 可以改變 runtime/memory，但 measured ratios 應在統計誤差內不變。
- 接受 topology switch 後的 reproject 必須保留新 `A_mask_` 下的合法 operator strings。
- `collection_count` 的 row sums 應接近每次從該 ensemble 呼叫 `ensemble_switch()` 所貢獻的一次 transition weight。

## 測試

- Unit test `set_A_mask`、`set_topology_pair`、`set_ensemble_ladder` 的 mask validation。
- Unit test `diff_site_between_masks`：剛好一個 site 不同時回傳該 index；零個或多個 site 不同時回傳 `-1`。
- 小系統 exact 或 regression test，比較 `PairToggle` ratio 和 ED / known saved result。
- Expanded ensemble smoke test，使用 three-node ladder：所有 counters shape 正確、足夠 steps 後 visits 非零、collection rows 為 positive。
- Regression test，比較 `delta_groups=0` 和 moderate grouped setting 的 `log_z` 或 pair ratios 是否相容。
- Checkpoint-style test：呼叫 `set_replica_op_string()` 後再呼叫 `recompute_midpoint_states()`。

## 相關檔案

- `csrc/qaqmc_renyi_core.hpp`
- `csrc/qaqmc_renyi_core.cpp`
- `csrc/qaqmc_core.hpp`
- `csrc/bindings.cpp`
- `src/engines/qaqmc_renyi.py`
- `src/tee/qaqmc_renyi_ratio.py`
- `src/tee/reweighting.py`
- `src/mpi/qaqmc_renyi_ratio_mpi.py`
- `src/mpi/reweighting_mpi.py`
- `specs/src/tee/reweighting.md`
- `specs/data/hdf5_expanded_result.md`

## 開放問題

- 是否要把 PairToggle 的 Barker acceptance 明確保留為演算法選擇，還是未來統一成 Metropolis acceptance。
- `ensemble_switch()` 目前允許 neighbor masks 差多個 sites，並用所有差異 sites 的 log ratio 累加；實務上 KP ladder 多半是 single-site step。是否要在 Python 端或 C++ 端強制 graph edge 只差一個 site。
- `collection_count` 的 row-sum diagnostic 是否要在 C++ 端提供輕量 sanity check，避免 Python 端才發現某些 ensemble 沒有有效 collection weight。
- `delta_groups` 的 envelope invariant 目前依賴 builder 正確性；是否要提供 debug mode 檢查 sampled slice 的 `w_actual <= w_envelope`。
