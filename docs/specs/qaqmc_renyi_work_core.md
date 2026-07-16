# 規格：csrc/qaqmc_renyi_work_core.hpp / csrc/qaqmc_renyi_work_core.cpp

## 角色
`QAQMCRenyiWorkEngine` 是預計新增的 C++ 端 two-replica QAQMC nonequilibrium-work Renyi 引擎。它的目標是把 D'Emidio paper 裡的 nonequilibrium work / Jarzynski estimator 接到目前的 QAQMC Renyi topology machinery，並支援 KP TEE ladder 工作流程中**任意 nested mask 對 `(A_start, A_end)` 的縫合**。

這個引擎不取代目前的 expanded ensemble engine 或 ratio estimator。它提供另一條估計 Renyi entropy 差的路徑：

```text
λ = 0 → swap topology = A_start
    -> finite-time λ protocol over D = A_end \ A_start
    -> λ = 1 → swap topology = A_end
    -> Jarzynski estimator for ΔS2 = S2(A_end) - S2(A_start)
```

核心想法是：用兩個 physical replicas 表示 Renyi partition function，讓 `λ` 控制 **差集 `D = A_end \ A_start`** 中每個 site 是否插入 swap topology（`A_start` 中的 sites 永遠 join、`A_end` 外的 sites 永遠 split），沿 nonequilibrium trajectory 累積 dimensionless work `w`，最後用

```text
ΔS2 = S2(A_end) - S2(A_start) = -log < exp(-w) >
```

估計第二 Renyi entropy 差。當 `A_start = ∅` 時，`ΔS2 = S2(A_end) - 0 = S2(A_end)`，即 paper 主文討論的特例。

**對 KP workflow 的接法**：每個 ladder rung 對應一次 `set_region_pair(M_k, M_{k+1}) + run_trajectories(...)`，回傳的 `ΔS2_k` 累加成 `S2(M_K) = Σ_k ΔS2_k`，再丟給 KP composer 組 γ。對 composer 完全透明（同 ratio estimator 的輸出格式）。

## 物件 / 函式

| 名稱 | 種類 | 可見性 | 用途 |
| --- | --- | --- | --- |
| `QAQMCRenyiWorkEngine` | class | public | Two-replica QAQMC nonequilibrium-work Renyi engine。 |
| `WorkTrajectoryResult` | struct | public nested | 單條 trajectory 的 work、acceptance、final topology diagnostics。 |
| `WorkRunResult` | struct | public nested | 多條 trajectories 的 aggregate statistics。 |
| `set_region_pair(A_start_mask, A_end_mask)` | method | public | 設定縫合的起點/終點 region pair，必須 nested (`A_start ⊆ A_end`)，內部建 `D = A_end \ A_start`。 |
| `set_region(A_mask)` | method | public | Convenience：等同 `set_region_pair(zeros, A_mask)`，對應 paper 的 ∅→A 特例。 |
| `set_lambda_schedule(lambdas)` | method | public | 設定 nonequilibrium protocol 的 λ schedule。 |
| `thermalize(n_steps)` | method | public | 在 `λ = 0` 起點 sector（`B = ∅`，backend mask = `A_start`）熱化。 |
| `run_trajectory()` | method | public | 從目前 configuration 開始執行一條 λ trajectory，回傳 work。 |
| `run_trajectories(n, decorrelation_steps)` | method | public | 執行多條 trajectories 並回傳 aggregate statistics（含 `ΔS2`）。 |
| `accumulate_work(lambda_old, lambda_new)` | method | private | 依目前 topology subset `B ⊆ D` 累積 work increment。 |
| `topology_sweep_random_permutation(lambda)` | method | private | 對差集 `D` 的 sites 做 random permutation topology sweep。 |
| `propose_toggle_site(site, lambda)` | method | private | 對單一 `D` 內 site proposal insert/remove swap。 |
| `qaqmc_sweep_current_topology()` | method | private | 在目前 backend mask（= `A_start ∪ B`）下做 diagonal update + cluster update。 |
| `reset_to_start_sector()` | method | private | 將 current `B` 設為空、backend mask 設為 `A_start`，即 `λ = 0` 起點。 |
| `force_end_sector()` | method | private | 在 endpoint 需要時將 current `B` 設為 `D`、backend mask 設為 `A_end`。 |

## 物理 / 演算法契約

- 引擎維護兩個 physical replicas，不為每個 topology sector 維護獨立 replicas。
- `A_start_mask` 是 trajectory 起點 sector 的 swap region；`A_end_mask` 是終點 sector 的 swap region；兩者**必須 nested**：`A_start ⊆ A_end`。
- **差集** `D = A_end \ A_start` 是被 λ 控制的 sites 集合。`A_start` 中的 sites 整個 trajectory 永遠 join；`A_end` 之外的 sites 永遠 split；只有 `D` 中的 sites 動態切換。
- Trajectory 中的 current **動態** swap subset `B` 必須滿足 `B ⊆ D`。當前實際 backend swap mask（即 backend `QAQMCRenyiEngine` 看到的 `A_mask_`）為：

```text
A_swap_current = A_start ∪ B
```

- `λ` 控制 D 上的 topology mixture 外場：

```text
g(λ, B) = λ^{|B|} (1 - λ)^{|D|-|B|}
```

- `λ = 0` 時，合法 topology 只有 `B = ∅`（backend mask = `A_start`）。
- `λ = 1` 時，合法 topology 只有 `B = D`（backend mask = `A_end`）。
- Work 只在改變 `λ` 時累積；topology update 和 QAQMC update 本身不直接貢獻 work。
- 單條 trajectory 的 dimensionless work 定義為：

```text
Δw_m = - [log g(λ_{m+1}, B_m) - log g(λ_m, B_m)]
w = Σ_m Δw_m
```

其中 `B_m` 是更新 `λ_m -> λ_{m+1}` 前的 current dynamic subset。

- 第二 Renyi entropy 差的 estimator：

```text
ΔS2 = S2(A_end) - S2(A_start) = -log mean(exp(-w))
```

當 `A_start = ∅` 時 `S2(A_start) = 0`，`ΔS2 = S2(A_end)`，回到 paper 主文的 ∅→A 特例。

- 每個 `λ` step 內，topology update 使用 `random permutation sweep`：把 **差集 `D`** 的 sites 隨機打亂後，每個 site proposal 一次。`A_start` 與 `A_end` 之外的 sites 不參與 proposal。
- 每次 topology toggle 的 acceptance 必須包含：
  - QAQMC/Renyi topology weight ratio（透過 backend `log_weight_ratio_for_toggle` 計算）；
  - `log g(λ, B_new) - log g(λ, B_old)`；
  - proposal correction，single-site insert/remove 為對稱 proposal，correction = 0。
- 在 current backend mask = `A_start ∪ B` 下，diagonal update 和 cluster update 必須使用和 `QAQMCRenyiEngine` 相同的 channel/replica boundary condition。
- **`λ = 1` 端點處理（採 paper 的 "multiply by 1" prescription）**：trajectory 結束時若 `|B| < |D|`，D 中未接上的 sites 在最後一個 work increment 的 g-factor 應視為 1，亦即只對已合併 sites 累積 `log(λ_new/λ_old)`，跳過 `(1-λ)`-factor 中對應 `λ_new=1` 的發散項。這保留 trajectory 的 work 量測而不讓 `exp(-w)` 變 0；同時記錄該 trajectory 的 `unjoined_at_end_count` 作為 diagnostic。

## 整合 `QAQMCRenyiEngine` 的策略

Work engine 不另外維護 two-replica state，而是**內嵌一個 `QAQMCRenyiEngine` 物件**作為 backend，所有 channel-space machinery（vertex lists、offdiag paths、diagonal/cluster update、bond-spin cache、site-op reproject）都共用既有實作。

為支援動態 topology，須在 `QAQMCRenyiEngine` 新增：

| 新增成員 | 角色 |
| --- | --- |
| `Mode::Work` | 第三種 mode，與 `PairToggle` / `Expanded` 並列。`mc_step()` 在這個 mode 下用 current `A_mask_` 當 boundary，不做 topology toggle、不更新 visit counts。 |
| `set_A_mask_for_work(mask)` | Work-mode 專用 mask switch：設 `A_mask_`、`A_masks_[0/1]`、`mode_ = Work`、`diff_site_ = -1`、`reset_visit_counts`，並且**清掉 op_string 回 vacuum**（所有 op_types = 1、op_sites = 0、state_at_M = 0）。原因：trajectory 結束時 op_string 是舊 mask 下的合法 config（offdiag parity = 0 在舊 channel 下），切到新 mask 後 parity 在新 channel 下通常 ≠ 0 → 非法 sector；`reproject_site_ops_for_mask_with_paths` 對這狀況**等同 identity**（用「目前 -1 ops 建 paths」再判斷「這些 ops 是不是 -1」）所以救不了，而 `cluster_update` 保 parity，decorrelation 也出不來。Vacuum 是 trivially valid（沒任何 offdiag），後續 `decorrelation_steps × mc_step` 把它熱化到新 sector 平衡。 |
| `log_weight_ratio_for_toggle(int site)` | 計算把 current `A_mask_` 在 `site` 位翻轉後的 log-weight-ratio。內部 build 兩份 offdiag paths（current 跟 toggled）+ `log_weight_for_site_with_paths` 相減；不調用 `recompute_midpoint_states()`、不 rebuild 全域 alias / cluster cache。回傳 `-1e30` 表示 spin/path 不相容（拒絕）。 |
| `apply_single_bit_toggle(int site)` | 已決定 accept 後實際翻 `A_mask_[site]`，rebuild paths 並 reproject affected site 的 site operators (`reproject_site_ops_at_site_with_paths`)。注意 reproject 在 single-bit toggle 下對 op_types 是 identity，但 cache 一致性仍需 paths 重建。 |
| `set_mode(Mode)` | 切換 mode，僅改 `mode_` 旗標。一般情況下用 `set_A_mask_for_work` 而不是這個。 |

Work engine 對 backend 的對應關係：

```text
work-engine A_start_mask  = (起點 sector, 固定)       → 不直接給 renyi engine，用來構 D 跟初始 backend mask
work-engine A_end_mask    = (終點 sector, 固定)       → 同上
work-engine D_mask        = A_end \ A_start (固定)    → toggle 範圍
work-engine B_mask        = (D 內 current subset, 動態) → 跟 A_start 合併後給 renyi engine
backend renyi engine A_mask_ = A_start ∪ B (動態)     → 隨 B 變動同步更新
```

也就是 work engine 內部維護兩個固定 mask（`A_start_mask`, `A_end_mask`）跟一個動態的 `B_mask`（記錄 `D` 中哪些 sites 已加入 swap），實際傳給 backend 的 boundary 是 `A_start ∪ B`。每次 `B` 翻轉一個 bit，就呼叫 `apply_single_bit_toggle(site)` 同步翻 backend `A_mask_` 並 reproject。

效能考量：`log_weight_ratio_for_toggle` 一次只觸 single-bit 差異，cost 應與既有 `log_weight_ratio_for_site` 相當（O(local path length)）。每個 λ step 的 topology sweep 是 `O(|D| × per-site-ratio-cost)`，符合 v1 預期。`set_topology_pair` 的全 rebuild 路徑**不可用**，會壓垮 sampling。

當 `A_start = ∅` 時 `D = A_end`，所有公式退化到 paper 主文情形。當 `|D| = 1`（單 site rung，KP ratio ladder 的常見情境），work engine 退化成一條沿 λ ∈ [0,1] 累積單 bit 翻轉貢獻的 trajectory，仍然是合法的 Jarzynski estimator（雖然此時可能 ratio estimator 已經夠用，但形式上沒禁止）。

## 輸入

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `N`, `Omega`, `delta_min`, `delta_max`, `Rb`, `M`, `epsilon` | scalars | QAQMC physical parameters。 |
| `pos` | `double*`, `(N, pos_dim)` | Site coordinates。 |
| `neighbor_cutoff` | `int` | Rydberg interaction bond cutoff。 |
| `delta_groups` | `int` | Grouped alias table 數量。 |
| `A_start_mask` | `uint8_t[N]` | Trajectory 起點 sector 的 swap region。`A_start = ∅` 對應 paper 主文情形。 |
| `A_end_mask` | `uint8_t[N]` | Trajectory 終點 sector 的 swap region。必須滿足 `A_start ⊆ A_end`。 |
| `lambda_schedule` | `double[K+1]` | Nonequilibrium protocol 的 λ path。`lambda_schedule[0] = 0`、`lambda_schedule[K] = 1`。 |
| `n_topology_sweeps_per_lambda` | `int` | 每個 λ step 做幾次 topology sweep。**v1 預設 = 1**（對齊 paper）；> 1 是 future tuning hook，不影響 Jarzynski statistical exactness，但會增加每 λ step 成本並可能改善 work distribution variance。 |
| `n_qaqmc_sweeps_per_lambda` | `int` | 每個 λ step 做幾次 diagonal+cluster update。**v1 預設 = 1**（對齊 paper）；同上備註。 |
| `n_trajectories` | `int` | Jarzynski average 使用的 trajectory 數量。 |
| `decorrelation_steps` | `int` | trajectories 起點之間在 `λ=0` 起點 sector（backend mask = `A_start`）做多少 QAQMC decorrelation steps。 |
| `seed` | `uint64_t` | RNG seed。 |

## 輸出

| 輸出 | Type / Shape | 意義 |
| --- | --- | --- |
| `WorkTrajectoryResult.work` | `double` | 單條 trajectory 的 dimensionless work `w`。 |
| `WorkTrajectoryResult.exp_minus_work` | `double` | `exp(-w)`，供 Jarzynski average 使用。 |
| `WorkTrajectoryResult.final_swap_count` | `int` | Trajectory 結束時 `|B|`（D 內的動態 subset 大小）。 |
| `WorkTrajectoryResult.unjoined_at_end_count` | `int` | 結束於 `λ=1` 但 D 內仍未合併的 sites 數量（`|D| - |B|`），套用 "multiply by 1" prescription 的計數；> 0 表示 quench 太快。 |
| `WorkTrajectoryResult.topology_accepts/attempts` | `int64` | Topology proposal acceptance diagnostics。 |
| `WorkRunResult.mean_exp_minus_work` | `double` | `mean(exp(-w))` = `Z_{A_end} / Z_{A_start}` 估計值。 |
| `WorkRunResult.delta_s2` | `double` | `-log mean(exp(-w))` = `S2(A_end) - S2(A_start)`。當 `A_start = ∅` 時等於 `S2(A_end)`。 |
| `WorkRunResult.work_mean/work_var` | `double` | Work distribution diagnostics。 |
| `WorkRunResult.trajectory_count` | `int64` | 實際完成 trajectories 數量。 |

## 資料契約

- `A_start_mask`, `A_end_mask` 都是 length `N` 的 `uint8_t` array。`A_start ⊆ A_end` 必須在 `set_region_pair` 時檢查。
- 差集 `D_mask = A_end_mask & ~A_start_mask`（按位元）由引擎內部建構，外部不傳入。
- Current dynamic subset `B` 也用 length `N` 的 `uint8_t` mask 表示，且必須滿足：

```text
B_mask[i] == 1  =>  D_mask[i] == 1   (亦即 i ∈ D)
```

- `swap_count = |B|` 由 `B_mask` 的 bit sum 得到。
- Backend mask（傳給 `QAQMCRenyiEngine` 的 `A_mask_`）為：

```text
backend_A_mask[i] = A_start_mask[i] | B_mask[i]
```

- `log_g(λ, B)` 應以 log-space 計算，**自由度只算 D 上的 sites**：

```text
log_g = |B| * log(λ) + (|D|-|B|) * log(1-λ)
```

- Endpoint handling 不應直接在 `λ=0` 且 `|B|>0` 或 `λ=1` 且 `|B|<|D|` 時計算 log。
- `lambda_schedule` 必須單調非遞減，第一版只支援 forward protocol `0 -> 1`。
- `random permutation sweep` 的 permutation 只包含 `D_mask == 1` 的 sites（不是 A_end）。
- Trajectory 之間的 decorrelation 必須維持在 `λ = 0` + `B = ∅` sector，即 backend mask = `A_start`（對齊 paper：「always staying in the Z_start ensemble」，paper 主文的 `Z_∅` 是 A_start=∅ 的特例）。

## 狀態 / 不變量

- `B_mask ⊆ D_mask` 必須一直成立（等價於：backend `A_mask_` 永遠滿足 `A_start ⊆ backend_A_mask ⊆ A_end`）。
- `λ = 0` trajectory 起點必須是起點 sector：`B_mask = ∅`，backend `A_mask_ = A_start`。
- 若 protocol 結束在 `λ = 1`，理想 final topology 應是終點 sector：`B_mask = D_mask`、backend `A_mask_ = A_end`；若未達成，套 "multiply by 1" prescription 並記錄 `unjoined_at_end_count`。
- Work accumulator `w_current_` 在每條 trajectory 開始時歸零。
- Work increment 使用更新 λ 前的 current `B_mask`。
- 每次 topology toggle 接受後，必須透過 `apply_single_bit_toggle(site)` reproject site operators 並更新 backend `A_mask_`。
- QAQMC diagonal/cluster update 必須在 current backend mask = `A_start ∪ B` 所定義的 boundary condition 下執行（透過 backend `Mode::Work`）。
- `run_trajectory()` 結束後不應把 trajectory work 混入下一條 trajectory。

## 行為

1. 建構子初始化 two-replica QAQMC/Renyi core state（內嵌一個 `QAQMCRenyiEngine` 並設成 `Mode::Work`）、proposal tables、RNG 和 diagnostics。
2. `set_region_pair(A_start_mask, A_end_mask)` 檢查 nested 性、建立 `D_mask` 與 `D_sites` list；同步初始化 backend `A_mask_ = A_start`、`B_mask = ∅`。
3. `set_lambda_schedule()` 設定 forward protocol `λ_0 ... λ_K`。
4. `thermalize()` 在 `λ=0` 且 `B=∅` 的起點 sector（backend mask = `A_start`）下執行 QAQMC updates。
5. `run_trajectory()` 從目前 `λ=0` equilibrium-like configuration 開始：
   - `w = 0`、`unjoined_at_end_count = 0`。
   - 對每個 `λ_m -> λ_{m+1}`：
     1. 用目前 `B_m` 累積 work increment（endpoint-safe，見 `accumulate_work`）。
     2. 將 current λ 設為 `λ_{m+1}`。
     3. 做 `n_topology_sweeps_per_lambda` 次 random permutation topology sweep（permutation 在 `D_sites` 上）。
     4. 做 `n_qaqmc_sweeps_per_lambda` 次 backend `mc_step()`（在 `Mode::Work` 下）。
   - 回傳 `w` 和 diagnostics。
6. `run_trajectories()` 重複執行多條 trajectories，對 `exp(-w)` 做 log-sum-exp aggregation 並輸出 `delta_s2 = S2(A_end) - S2(A_start)`。

## 函數規格

### `QAQMCRenyiWorkEngine::set_region_pair(A_start_mask, A_end_mask)`

**種類：** method  
**可見性：** public

**用途**  
設定起點/終點 region pair，建立差集 `D` 及 topology sweep 使用的 site list，並把 backend 切到 `A_start` sector。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `A_start_mask` | `uint8_t[N]` | Trajectory 起點 sector 的 swap region。 |
| `A_end_mask` | `uint8_t[N]` | Trajectory 終點 sector 的 swap region；必須 nested `A_start ⊆ A_end`。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `A_start_mask_`, `A_end_mask_` | `uint8_t[N]` | 儲存起點/終點 region。 |
| `D_mask_` | `uint8_t[N]` | `A_end_mask_ & ~A_start_mask_`。 |
| `D_sites_` | `vector<int>` | 所有 `D_mask_[i] == 1` 的 site indices。 |
| `B_mask_` | `uint8_t[N]` | Reset 為 empty。 |
| backend `renyi_engine_.A_mask_` | `uint8_t[N]` | 設為 `A_start_mask_`（透過 `set_A_mask`）。 |

**演算法流程**

1. 檢查 mask 長度等於 `N_`。
2. 檢查 `A_start ⊆ A_end`：對所有 `i`，若 `A_start_mask[i] = 1` 則必須 `A_end_mask[i] = 1`，否則 raise。
3. 複製到 `A_start_mask_` / `A_end_mask_`。
4. 計算 `D_mask_` 與 `D_sites_`。
5. Reset `B_mask_` 為 empty。
6. 將 backend `renyi_engine_` 切到 `Mode::Work`，呼叫 `set_A_mask(A_start_mask_)` 並 reproject 全部 site operators。

**邊界情況**

- `A_start = A_end` (`D = ∅`) 合法但 `ΔS2 = 0`；第一版直接允許，trajectory work 永遠為 0。
- `A_start = ∅` 是 paper 主文情形（可透過 convenience helper `set_region(A_end)`）。

**不變量**

- `D_sites_.size() == |D|`。
- 設定後 `B_mask_ ⊆ D_mask_` 自動滿足（B=∅）。

**測試**

- 設定 `A_start = ∅, A_end = A` 後檢查 `D_sites_ == A_sites_`。
- 設定 nested pair 後檢查 `D_mask = A_end ^ A_start`。
- 非 nested 應 raise。

### `QAQMCRenyiWorkEngine::set_region(A_mask)`

**種類：** method  
**可見性：** public

**用途**  
Convenience wrapper：等同 `set_region_pair(zeros_mask, A_mask)`。對應 paper 主文的 ∅→A 特例。

**輸入 / 輸出 / 流程**：見 `set_region_pair`。

### `QAQMCRenyiWorkEngine::set_lambda_schedule(lambdas)`

**種類：** method  
**可見性：** public

**用途**  
設定 nonequilibrium work protocol 的 λ path。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `lambdas` | `double[K+1]` | 從 0 到 1 的 monotonic schedule。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `lambda_schedule_` | `vector<double>` | 儲存 protocol。 |

**演算法流程**

1. 檢查長度至少為 2。
2. 檢查 `λ` 單調非遞減。
3. 檢查第一版 protocol 是否從 0 到 1。
4. 儲存 schedule。

**邊界情況**

- 若 schedule 包含 endpoint 0/1，`log_g` 必須用 endpoint-safe policy。

**不變量**

- `lambda_schedule_[0] == 0`，`lambda_schedule_.back() == 1`。

**測試**

- 非單調 schedule validation。

### `QAQMCRenyiWorkEngine::accumulate_work(lambda_old, lambda_new)`

**種類：** method  
**可見性：** private

**用途**  
在改變 λ 時，依目前 `B_mask_` 累積 Jarzynski work increment。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `lambda_old`, `lambda_new` | `double` | 相鄰 protocol points。 |
| `B_mask_` | `uint8_t[N]` | 更新 λ 前的 D 上 current subset。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `w_current_` | `double` | 加上 `Δw`。 |
| `unjoined_at_end_count_` | `int` | `λ_new=1, b<|D|` 時更新。 |

**演算法流程**

1. 計算 `b = |B|` 和 `d = |D|`（**注意：不是 `|A_end|`，是差集大小**）。
2. `Δ_join = b * (log λ_new - log λ_old)`（已合併 sites 的貢獻）。`λ_old = 0` 時 `b = 0`（trajectory 起點不變量），自動取 0。
3. `Δ_split` 的計算需 endpoint-safe：
   - 若 `λ_new < 1`：`Δ_split = (d-b) * (log(1-λ_new) - log(1-λ_old))`。
   - 若 `λ_new = 1` 且 `b = d`：`Δ_split = 0`（D 全部合併，無 split factor）。
   - 若 `λ_new = 1` 且 `b < d`：採 paper 的 "multiply by 1" prescription，**將 split 因子視為 1**，即 `Δ_split = 0`，並設 `unjoined_at_end_count_ += (d - b)` 作 diagnostic。
4. 累積 `w_current_ += -(Δ_join + Δ_split)`。

**邊界情況**

- `λ_old = 0` 時必須有 `b = 0`（不變量保證；trajectory 起點）。
- 若 schedule 含 `λ_old = 0` 開頭，`Δ_join` 在 `b = 0` 時自動為 0，避免 `0 * log(0)`。第一版預設 `λ_old = 0` 出現時跳過 `log λ_old` 計算。
- `λ_new = 1` + `b < d` 的處理已併入流程步驟 3。

**不變量**

- Work increment 使用 topology update 前的 `B_mask_`。

**測試**

- 手算 small A：`|A|=2`，`B=∅`、`B={i}`、`B=A` 的 `Δw`。

### `QAQMCRenyiWorkEngine::topology_sweep_random_permutation(lambda)`

**種類：** method  
**可見性：** private

**用途**  
在固定 λ 下，對 A 區所有 sites 做一次 random permutation topology sweep。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `lambda` | `double` | Current λ。 |
| `D_sites_` | `vector<int>` | 要 proposal 的 sites（差集 D，不是 A_end）。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `B_mask_` | `uint8_t[N]` | 若 proposal accepted，toggle 對應 site。 |
| backend `A_mask_` | `uint8_t[N]` | 同步更新（透過 `apply_single_bit_toggle`）。 |
| topology counters | `int64` | attempts/accepts。 |

**演算法流程**

1. 複製 `D_sites_`。
2. 用 RNG shuffle 成 random permutation。
3. 依序對每個 site 呼叫 `propose_toggle_site(site, lambda)`。

**邊界情況**

- `λ=0` 時 insert acceptance 應為 0。
- `λ=1` 時 remove acceptance 應為 0。

**不變量**

- 每次 sweep 中每個 D site 被 proposal 一次。
- `B_mask_ ⊆ D_mask_`。

**測試**

- 固定 seed 下 permutation reproducibility。
- 一次 sweep attempts 數等於 `|D|`。

### `QAQMCRenyiWorkEngine::propose_toggle_site(site, lambda)`

**種類：** method  
**可見性：** private

**用途**  
對單一 site proposal insert/remove swap topology。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `site` | `int` | 要 toggle 的 D 區 site（必須在 `D_mask_` 內）。 |
| `lambda` | `double` | Current λ。 |
| `B_mask_` | `uint8_t[N]` | Current dynamic subset。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `B_mask_` | `uint8_t[N]` | 接受時 toggle `B_mask_[site]`。 |
| backend `A_mask_` | `uint8_t[N]` | 接受時透過 `apply_single_bit_toggle(site)` 同步翻 backend mask。 |
| site operators | arrays | 接受時由 `apply_single_bit_toggle` 內部 reproject affected site。 |

**演算法流程**

1. 若 `B_mask_[site] == 0`，proposal insert：`B_new = B ∪ {site}`、`backend_A_new = A_start ∪ B_new`。
2. 若 `B_mask_[site] == 1`，proposal remove：`B_new = B \ {site}`、`backend_A_new = A_start ∪ B_new`。
3. 透過 backend `QAQMCRenyiEngine` 計算 single-bit toggle 的 log weight ratio（backend 看的就是 `backend_A_mask_` 翻 1 bit）：

```text
log_ratio_qaqmc = renyi_engine_.log_weight_ratio_for_toggle(site, backend_A_mask_)
```

   此 primitive **不會** rebuild 全域 offdiag paths（不可呼叫 `set_topology_pair`，那會 trigger 全 rebuild）。回傳 `-inf` 表示 spin/path 不相容。

4. 計算 lambda bias ratio：

```text
insert (B_mask_[site]==0): log_ratio_lambda = log(λ)   - log(1-λ)
remove (B_mask_[site]==1): log_ratio_lambda = log(1-λ) - log(λ)
```

5. 計算 `log_accept = log_ratio_qaqmc + log_ratio_lambda`。
6. 以 `min(1, exp(log_accept))` 接受。
7. 接受時呼叫 `renyi_engine_.apply_single_bit_toggle(site)` 翻 backend `A_mask_` 並 reproject affected site；同時翻 work engine 自己的 `B_mask_[site]` 維持 `backend_A_mask_ = A_start ∪ B_mask_` 不變量。

**邊界情況**

- 若 spin/path 不相容，`log_ratio_qaqmc = -inf`，proposal rejection。
- Endpoint λ：`λ = 0` 時 `log(λ) = -inf`、`λ = 1` 時 `log(1-λ) = -inf`。Spec line 54-55 保證在這兩個端點只有一種合法 topology（empty 或 full），但中間步驟若 schedule 接近端點要小心：以 `λ ∈ {0, 1}` 為輸入時直接回傳 reject。Schedule 內部點不會踩到嚴格 endpoint（lambda_schedule 內部點應 `0 < λ < 1`）。

**不變量**

- 接受後 current boundary condition 必須等於新的 `B_mask_`。

**測試**

- `λ=0.5` 時 lambda bias ratio 為 0。
- Insert/remove ratio 互為反向。

### `QAQMCRenyiWorkEngine::qaqmc_sweep_current_topology()`

**種類：** method  
**可見性：** private

**用途**  
在 current topology mask 下更新兩個 replicas 的 QAQMC configuration。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| current topology mask | `uint8_t[N]` | 由 `B_mask_` 定義。 |
| replicas | operator strings | 目前 two-replica configuration。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| replicas | operator strings | diagonal/cluster update 後的新 configuration。 |

**演算法流程**

1. 確認 backend `renyi_engine_` 已切到 `Mode::Work` 並且 `A_mask_` 等於 work-engine 的 `B_mask_`（不變量；應在每次 toggle 接受後就維持）。
2. 呼叫 `renyi_engine_.mc_step()`，在 Work mode 下即執行 mask-aware diagonal_update + cluster_update，不做 topology toggle / visit count 更新。

**邊界情況**

- 若目前 topology 剛接受 toggle，`apply_single_bit_toggle` 必須已完成 reproject，否則 diagonal/cluster update 會在 stale offdiag-paths cache 下執行。

**不變量**

- QAQMC update 使用 current `B_mask_`（透過 backend 的 `A_mask_`），不是 full target `A_mask_`。

**測試**

- 固定 `B_mask_` 下和現有 `QAQMCRenyiEngine` fixed-mask 更新一致。

### `QAQMCRenyiWorkEngine::run_trajectory()`

**種類：** method  
**可見性：** public

**用途**  
執行一條 nonequilibrium λ trajectory，輸出 work 和 diagnostics。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| current configuration | engine state | 通常應來自 `λ=0` 起點 sector（backend mask = `A_start`）equilibrium-like state。 |
| `lambda_schedule_` | `double[K+1]` | Forward protocol。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| return value | `WorkTrajectoryResult` | Work、`exp(-w)`、`unjoined_at_end_count`、acceptance diagnostics。 |
| engine state | internal | trajectory 結束時位於 final λ/topology。 |

**演算法流程**

1. 檢查 schedule 和 region pair 已設定。
2. 設定 `w_current_ = 0`、`unjoined_at_end_count_ = 0`。
3. 對每段 `λ_m -> λ_{m+1}`：
   - 呼叫 `accumulate_work(λ_m, λ_{m+1})`。
   - 更新 current λ。
   - 做 `n_topology_sweeps_per_lambda` 次 random permutation topology sweep（在 `D_sites` 上）。
   - 做 `n_qaqmc_sweeps_per_lambda` 次 backend `mc_step()`（`Mode::Work`）。
4. 回傳 `w_current_` 和 diagnostics。

**邊界情況**

- 若 final `λ=1` 但 `B_mask_ != D_mask_`，已由 `accumulate_work` 套 prescription 並計入 `unjoined_at_end_count_`。

**不變量**

- Work 不跨 trajectories 累積。

**測試**

- `A_start = A_end` (`D = ∅`) 時 trajectory work 為 0。
- `A_start = ∅, A_end = A` 時退化到 paper 主文行為。
- Very slow protocol 下平均 work 接近 equilibrium free-energy difference。
- Nested non-empty pair (`A_start = A, A_end = AB`) 量測值與「先量 `S(A)` 再量 `S(AB)` 然後相減」一致。

### `QAQMCRenyiWorkEngine::run_trajectories(n, decorrelation_steps)`

**種類：** method  
**可見性：** public

**用途**  
執行多條 nonequilibrium trajectories 並估計 `S2`。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `n` | `int` | Trajectory 數量。 |
| `decorrelation_steps` | `int` | 每條 trajectory 起點間的 no-swap QAQMC updates。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| return value | `WorkRunResult` | `mean(exp(-w))`、`S2`、work diagnostics。 |

**演算法流程**

1. 對 `traj = 1 ... n`：
   - 呼叫 `reset_to_start_sector()`：backend `Mode::Work`、`A_mask_ = A_start_mask_`、`B_mask_ = ∅`、current λ = 0。
   - 在起點 sector 下做 `decorrelation_steps` 次 `renyi_engine_.mc_step()`（對齊 paper：「always staying in the Z_{A_start} ensemble」）。
   - 呼叫 `run_trajectory()`，trajectory 結束時 engine 處於 `λ = 1` 附近、backend mask 可能 ≠ `A_end`。
   - 儲存 `w`、`exp(-w)`、`unjoined_at_end_count`。
2. 計算 `mean_exp_minus_work`（必須用 log-sum-exp）。
3. 計算 `delta_s2 = -log(mean_exp_minus_work) = S2(A_end) - S2(A_start)`。

**邊界情況**

- 若 `exp(-w)` underflow/overflow，必須使用 log-sum-exp aggregator；不可天真累加。
- 若多條 trajectories 都觸發 `unjoined_at_end_count > 0`，應警告 quench 太快、需要更密的 `lambda_schedule`。
- `D = ∅` (`A_start = A_end`) 時所有 trajectory work = 0，`mean(exp(-w)) = 1`，`delta_s2 = 0`。應直接 short-circuit 而不跑 trajectories。

**不變量**

- 每條 trajectory 的 work accumulator 獨立。

**測試**

- 人工 work samples 檢查 Jarzynski aggregation。

## 邊界情況

- `λ=0` 和 `λ=1` 會造成 `log(0)`，必須用 endpoint-safe logic：
  - `λ=0` 起點：保證 `B = ∅`，所有 `b * log(λ_old)` 項因 `b=0` 自動為 0。
  - `λ=1` 終點：若 `b < |D|`，D 中未合併 sites 採 paper 的 "multiply by 1" prescription（`Δ_split = 0`、記 `unjoined_at_end_count`）。
- Jarzynski estimator 對 rare low-work trajectories 敏感；若 `exp(-w)` variance 很大，應輸出 diagnostic。
- 若 topology sweep 在 `λ=1` 前沒有接近 full swap (`|B| = |D|`)，final estimator 可能 variance 很大；`unjoined_at_end_count > 0` 是早期警訊。
- 若 `|D|` 太大，`n_lambda_steps` 和 trajectories 數量需要增加。對 KP ladder workflow，把大 jump 拆成多個小 rungs（每個 rung `|D|` 小）通常比一次 `∅ → A_target` 跳大段更穩定。
- `D = ∅` 是合法 trivial case；`delta_s2 = 0` 由 `run_trajectories` short-circuit。
- 如果 current topology update 後沒有 reproject site operators，後續 diagonal/cluster update 會在錯誤 boundary condition 下執行；`apply_single_bit_toggle` 必須在 accept 時同步呼叫。

## 效能備註

- 第一版 topology sweep 使用 `random permutation sweep`，確保每個 A site 每輪 proposal 一次，同時避免固定 ordering bias。
- 每個 `λ` step 的成本約為：

```text
O(|D| * toggle_ratio_cost + n_qaqmc_sweeps_per_lambda * QAQMC_sweep_cost)
```

其中 `toggle_ratio_cost` 來自 `log_weight_ratio_for_toggle` 的 single-bit ratio 計算，cost 等級與既有 `log_weight_ratio_for_site` 相當（O(local path length)），**不可**走 `set_topology_pair` 的全 rebuild 路徑（會壓垮 sampling）。
- KP workflow 中拆 ladder 成多個小 rungs：每個 rung 的 `|D|` 通常 = 1（單 site），topology sweep 變成 O(1) per λ step，整個 trajectory cost 由 QAQMC sweeps 主導。
- `n_lambda_steps` 越大，work distribution 通常越窄，但成本線性增加。
- `n_qaqmc_sweeps_per_lambda` 越大，configuration 越能適應 current λ，但成本增加。
- Aggregating `exp(-w)` 應用 log-sum-exp，避免 large work 下 underflow。
- Per-trajectory work 累積也可改用「乘 g_new/g_old 比值」的 product accumulator（paper line 751 建議）取代 log-sum，但 double precision 對典型 work 量級已足；v1 採 log-domain 即可。

## 驗收標準

- `D = ∅` (`A_start = A_end`) 時，`delta_s2 ≈ 0`（trajectory short-circuit）。
- `A_start = ∅` 時，`delta_s2 = S2(A_end)` 與 exact diagonalization 或現有 expanded ensemble / ratio estimator 在誤差內一致。
- Nested non-empty pair（如 `A_start = A, A_end = AB`）的 `delta_s2` 與「`S2(AB) - S2(A)` 從 ED 算出來」一致。
- 對同一個 target region，用 ladder of rungs vs 單一大 jump 兩種跑法，累積得到的 `S2(A_end)` 應在誤差內相同。
- `λ=0` 起點必須保持 backend mask = `A_start`。
- 對 `D` 中任一 site，`λ=0.5` 時 topology insert/remove 的 lambda bias ratio 應互為反向且大小為 0。
- 增加 `n_lambda_steps` 時，work variance 應下降或不惡化。
- Forward/reverse protocol 若未來實作，兩者估計應在誤差內一致。

## 測試

- Unit test `log_g` 和 `Δw` endpoint-safe 計算，包含 `|D|=0`、`|D|=1`、`b=|D|` 等 corner cases。
- Unit test `set_region_pair` 拒絕非 nested input。
- Unit test random permutation sweep attempts 數等於 `|D|`。
- Unit test single-site insert/remove acceptance ratio。
- Small-system trajectory smoke test：`A_start = ∅, A_end = {0}`（單 site），檢查 `B_mask` endpoint behavior。
- Regression test 1（`∅ → A` 特例）：和現有 expanded ensemble 在 4x4 m=1 小樣本上比較 central value。
- Regression test 2（nested pair）：`A_start = A, A_end = AB` 的 `delta_s2` 應等於 `S2(AB) - S2(A)` 從 ED 計算。
- Regression test 3（ladder 拆分一致性）：跑 `∅ → A` vs `∅ → 子集 → A` 兩條路徑得到的 `S2(A)` 在誤差內一致。
- Numerical stability test：大量 work samples 用 log-sum-exp aggregation。

## 相關檔案

- `csrc/qaqmc_renyi_core.hpp` — **需新增** `Mode::Work`、`log_weight_ratio_for_toggle(site, B_curr_mask)`、`apply_single_bit_toggle(site)`、`set_mode` 等成員，見「整合 `QAQMCRenyiEngine` 的策略」一節。
- `csrc/qaqmc_renyi_core.cpp` — 對應 implementation。
- `csrc/qaqmc_core.hpp`
- `src/engines/qaqmc_renyi.py` — Python wrapper 需新 entry point 暴露 work engine（推薦命名 `QAQMCRenyiWorkRydberg`）。
- `src/tee/qaqmc_renyi_ratio.py` — 可考慮加 `WorkEstimator` 並列為 `RatioEstimator` 之外的另一個 backend，對 KP composer 透明。
- `src/tee/compose_tee.py` — 不需要動，輸出 `ΔS_k` 累加組 γ 的邏輯共用。
- `paper/Entanglement entropy from nonequilibrium work.md`

## 開放問題

- 第一版是否只支援 forward protocol，或同時支援 reverse protocol 以做 Bennett/Crooks-style diagnostics。（v1 forward only）
- 是否支援非 nested pair（disjoint join+split），需要更通用的 `g(λ, J, S)`。（v1 不支援；KP workflow 用不到）
- `run_trajectory()` 結束後是否要自動 reset 回 `λ=0`，或交給 `run_trajectories()` 控制。（建議由 `run_trajectories()` 統一處理 reset，`run_trajectory()` 保持「從 current state 跑到 final λ」的純函數語義）
- Work estimator 是否要同時輸出 cumulant estimate，方便判斷 Jarzynski variance。（建議加入 `WorkRunResult.work_var` 已涵蓋部分，第二、三 cumulant 可未來擴充）
- Lambda schedule 是否應提供建議的非線性 spacing helper（例如 `λ_m = sin²(π m / 2K)` 之類），減少端點附近的 sweep 浪費。
- KP workflow 是用 ladder of small rungs（每 rung `|D|=1`）還是 medium-jump rungs（每 rung `|D|>1`）效率最好，需要實測選 default。
