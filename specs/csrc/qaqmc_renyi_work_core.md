# 規格：csrc/qaqmc_renyi_work_core.hpp / csrc/qaqmc_renyi_work_core.cpp

## 角色
`QAQMCRenyiWorkEngine` 是預計新增的 C++ 端 two-replica QAQMC nonequilibrium-work Renyi 引擎。它的目標是把 D'Emidio paper 裡的 nonequilibrium work / Jarzynski estimator 接到目前的 QAQMC Renyi topology machinery。

這個引擎不取代目前的 expanded ensemble engine。它提供另一條估計 Renyi entropy 的路徑：

```text
λ = 0 no-swap topology
    -> finite-time λ protocol
    -> λ = 1 full-swap topology
    -> Jarzynski estimator for S2
```

核心想法是：用兩個 physical replicas 表示 Renyi partition function，讓 `λ` 控制 A 區每個 site 是否插入 swap topology，沿 nonequilibrium trajectory 累積 dimensionless work `w`，最後用

```text
S2(A) = -log < exp(-w) >
```

估計第二 Renyi entropy。

## 物件 / 函式

| 名稱 | 種類 | 可見性 | 用途 |
| --- | --- | --- | --- |
| `QAQMCRenyiWorkEngine` | class | public | Two-replica QAQMC nonequilibrium-work Renyi engine。 |
| `WorkTrajectoryResult` | struct | public nested | 單條 trajectory 的 work、acceptance、final topology diagnostics。 |
| `WorkRunResult` | struct | public nested | 多條 trajectories 的 aggregate statistics。 |
| `LambdaSchedule` | struct | public nested | 儲存 `λ_0 ... λ_K` 和 endpoint policy。 |
| `TopologySweepMode` | enum | public nested | 第一版固定使用 `RandomPermutationSweep`。 |
| `set_region_mask(A_mask)` | method | public | 設定目標 entangling region A。 |
| `set_lambda_schedule(lambdas)` | method | public | 設定 nonequilibrium protocol 的 λ schedule。 |
| `thermalize(n_steps)` | method | public | 在 `λ = 0` no-swap sector 熱化。 |
| `run_trajectory()` | method | public | 從目前 configuration 開始執行一條 λ trajectory，回傳 work。 |
| `run_trajectories(n, decorrelation_steps)` | method | public | 執行多條 trajectories 並回傳 aggregate statistics。 |
| `accumulate_work(lambda_old, lambda_new)` | method | private | 依目前 topology subset `B` 累積 work increment。 |
| `topology_sweep_random_permutation(lambda)` | method | private | 對 A 區 sites 做 random permutation topology sweep。 |
| `propose_toggle_site(site, lambda)` | method | private | 對單一 site proposal insert/remove swap。 |
| `qaqmc_sweep_current_topology()` | method | private | 在目前 topology mask 下做 diagonal update + cluster update。 |
| `reset_to_no_swap_sector()` | method | private | 將 current topology 設為 empty B，也就是 `λ = 0` sector。 |
| `force_full_swap_sector()` | method | private | 在 endpoint 需要時將 current topology 設為 full A。 |

## 物理 / 演算法契約

- 引擎維護兩個 physical replicas，不為每個 topology sector 維護獨立 replicas。
- `A_mask` 是目標 entangling region。Trajectory 中的 current swap subset `B` 必須滿足 `B ⊆ A`。
- `λ` 控制 topology mixture 的外場：

```text
g(λ, B) = λ^{|B|} (1 - λ)^{|A|-|B|}
```

- `λ = 0` 時，合法 topology 只有 empty `B = ∅`。
- `λ = 1` 時，合法 topology 只有 full `B = A`。
- Work 只在改變 `λ` 時累積；topology update 和 QAQMC update 本身不直接貢獻 work。
- 單條 trajectory 的 dimensionless work 定義為：

```text
Δw_m = - [log g(λ_{m+1}, B_m) - log g(λ_m, B_m)]
w = Σ_m Δw_m
```

其中 `B_m` 是更新 `λ_m -> λ_{m+1}` 前的 current topology subset。

- 第二 Renyi entropy estimator：

```text
S2 = -log mean(exp(-w))
```

- 每個 `λ` step 內，topology update 第一版使用 `random permutation sweep`：把 A 區 sites 隨機打亂後，每個 site proposal 一次。
- 每次 topology toggle 的 acceptance 必須包含：
  - QAQMC/Renyi topology weight ratio；
  - `log g(λ, B_new) - log g(λ, B_old)`；
  - proposal correction，如第一版 insert/remove proposal 對稱則為 0。
- 在 current topology mask 下，diagonal update 和 cluster update 必須使用和 `QAQMCRenyiEngine` 相同的 channel/replica boundary condition。

## 輸入

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `N`, `Omega`, `delta_min`, `delta_max`, `Rb`, `M`, `epsilon` | scalars | QAQMC physical parameters。 |
| `pos` | `double*`, `(N, pos_dim)` | Site coordinates。 |
| `neighbor_cutoff` | `int` | Rydberg interaction bond cutoff。 |
| `delta_groups` | `int` | Grouped alias table 數量。 |
| `A_mask` | `uint8_t[N]` | 目標 entangling region。 |
| `lambda_schedule` | `double[K+1]` | Nonequilibrium protocol 的 λ path。 |
| `n_topology_sweeps_per_lambda` | `int` | 每個 λ step 做幾次 topology sweep；第一版可先設 1。 |
| `n_qaqmc_sweeps_per_lambda` | `int` | 每個 λ step 做幾次 diagonal+cluster update；第一版可先設 1。 |
| `n_trajectories` | `int` | Jarzynski average 使用的 trajectory 數量。 |
| `decorrelation_steps` | `int` | trajectories 起點之間在 `λ=0` sector 做多少 QAQMC decorrelation steps。 |
| `seed` | `uint64_t` | RNG seed。 |

## 輸出

| 輸出 | Type / Shape | 意義 |
| --- | --- | --- |
| `WorkTrajectoryResult.work` | `double` | 單條 trajectory 的 dimensionless work `w`。 |
| `WorkTrajectoryResult.exp_minus_work` | `double` | `exp(-w)`，供 Jarzynski average 使用。 |
| `WorkTrajectoryResult.final_swap_count` | `int` | Trajectory 結束時 `|B|`。 |
| `WorkTrajectoryResult.topology_accepts/attempts` | `int64` | Topology proposal acceptance diagnostics。 |
| `WorkRunResult.mean_exp_minus_work` | `double` | `mean(exp(-w))`。 |
| `WorkRunResult.s2` | `double` | `-log mean(exp(-w))`。 |
| `WorkRunResult.work_mean/work_var` | `double` | Work distribution diagnostics。 |
| `WorkRunResult.trajectory_count` | `int64` | 實際完成 trajectories 數量。 |

## 資料契約

- `A_mask` 是 length `N` 的 `uint8_t` array。
- Current topology subset `B` 也用 length `N` 的 `uint8_t` mask 表示，且必須滿足：

```text
B_mask[i] == 1  =>  A_mask[i] == 1
```

- `swap_count = |B|` 可由 `B_mask` 的 bit sum 得到。
- `log_g(λ, B)` 應以 log-space 計算：

```text
log_g = |B| * log(λ) + (|A|-|B|) * log(1-λ)
```

- Endpoint handling 不應直接在 `λ=0` 且 `|B|>0` 或 `λ=1` 且 `|B|<|A|` 時計算 log。
- `lambda_schedule` 必須單調非遞減，第一版只支援 forward protocol `0 -> 1`。
- `random permutation sweep` 的 permutation 只包含 `A_mask == 1` 的 sites。

## 狀態 / 不變量

- `B_mask ⊆ A_mask` 必須一直成立。
- `λ = 0` trajectory 起點必須是 no-swap sector：`B_mask = ∅`。
- 若 protocol 結束在 `λ = 1`，理想 final topology 應是 full-swap sector：`B_mask = A_mask`；若未達成，應記錄 diagnostic。
- Work accumulator `w_current_` 在每條 trajectory 開始時歸零。
- Work increment 使用更新 λ 前的 current `B_mask`。
- 每次 topology toggle 接受後，必須 reproject site operators 並更新 current `A_mask_` / topology mask。
- QAQMC diagonal/cluster update 必須在 current `B_mask` 所定義的 boundary condition 下執行。
- `run_trajectory()` 結束後不應把 trajectory work 混入下一條 trajectory。

## 行為

1. 建構子初始化 two-replica QAQMC/Renyi core state、proposal tables、RNG 和 diagnostics。
2. `set_region_mask()` 設定目標 A 區，並建立 A 區 site list。
3. `set_lambda_schedule()` 設定 forward protocol `λ_0 ... λ_K`。
4. `thermalize()` 在 `λ=0` 且 `B=∅` 的 no-swap sector 下執行 QAQMC updates。
5. `run_trajectory()` 從目前 `λ=0` equilibrium-like configuration 開始：
   - `w = 0`。
   - 對每個 `λ_m -> λ_{m+1}`：
     1. 用目前 `B_m` 累積 work increment。
     2. 將 current λ 設為 `λ_{m+1}`。
     3. 做 `n_topology_sweeps_per_lambda` 次 random permutation topology sweep。
     4. 做 `n_qaqmc_sweeps_per_lambda` 次 diagonal update + cluster update。
   - 回傳 `w` 和 diagnostics。
6. `run_trajectories()` 重複執行多條 trajectories，對 `exp(-w)` 做平均並輸出 `S2`。

## 函數規格

### `QAQMCRenyiWorkEngine::set_region_mask(A_mask)`

**種類：** method  
**可見性：** public

**用途**  
設定目標 entangling region A，並建立 topology sweep 使用的 site list。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `A_mask` | `uint8_t[N]` | 目標 region mask。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `A_mask_` | `uint8_t[N]` | 儲存目標 region。 |
| `A_sites_` | `vector<int>` | 所有 `A_mask[i] == 1` 的 site indices。 |

**演算法流程**

1. 檢查 mask 長度等於 `N_`。
2. 複製到 `A_mask_`。
3. 建立 `A_sites_`。
4. 將 current `B_mask_` reset 為 empty。

**邊界情況**

- 空 region 合法但 entropy 應為 0；第一版可以直接 raise 或回傳 zero diagnostic。

**不變量**

- `A_sites_.size() == |A|`。

**測試**

- 設定人工 mask 後檢查 `A_sites_` 和 `B_mask_`。

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
| `B_mask_` | `uint8_t[N]` | 更新 λ 前的 current swap subset。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `w_current_` | `double` | 加上 `Δw`。 |

**演算法流程**

1. 計算 `b = |B|` 和 `a = |A|`。
2. 計算 `log_g_old = b log λ_old + (a-b) log(1-λ_old)`。
3. 計算 `log_g_new = b log λ_new + (a-b) log(1-λ_new)`。
4. 累積 `w_current_ += -(log_g_new - log_g_old)`。

**邊界情況**

- `λ_old = 0` 時必須有 `b = 0`。
- `λ_new = 1` 時若 `b < a`，`log_g_new = -inf`，這代表該 trajectory 權重消失；第一版可記錄 diagnostic 並避免 NaN。

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
| `A_sites_` | `vector<int>` | 要 proposal 的 sites。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `B_mask_` | `uint8_t[N]` | 若 proposal accepted，toggle 對應 site。 |
| topology counters | `int64` | attempts/accepts。 |

**演算法流程**

1. 複製 `A_sites_`。
2. 用 RNG shuffle 成 random permutation。
3. 依序對每個 site 呼叫 `propose_toggle_site(site, lambda)`。

**邊界情況**

- `λ=0` 時 insert acceptance 應為 0。
- `λ=1` 時 remove acceptance 應為 0。

**不變量**

- 每次 sweep 中每個 A site 被 proposal 一次。
- `B_mask_ ⊆ A_mask_`。

**測試**

- 固定 seed 下 permutation reproducibility。
- 一次 sweep attempts 數等於 `|A|`。

### `QAQMCRenyiWorkEngine::propose_toggle_site(site, lambda)`

**種類：** method  
**可見性：** private

**用途**  
對單一 site proposal insert/remove swap topology。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `site` | `int` | 要 toggle 的 A 區 site。 |
| `lambda` | `double` | Current λ。 |
| `B_mask_` | `uint8_t[N]` | Current topology subset。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `B_mask_` | `uint8_t[N]` | 接受時 toggle site。 |
| current topology mask | `uint8_t[N]` | 接受時更新 boundary condition。 |
| site operators | arrays | 接受時 reproject affected site。 |

**演算法流程**

1. 若 `B_mask_[site] == 0`，proposal insert：`B_new = B ∪ {site}`。
2. 若 `B_mask_[site] == 1`，proposal remove：`B_new = B \\ {site}`。
3. 計算 QAQMC topology log weight ratio：

```text
log_ratio_qaqmc = log_weight_ratio_for_site(site, B_old, B_new)
```

4. 計算 lambda bias ratio：

```text
insert: log_ratio_lambda = log(λ) - log(1-λ)
remove: log_ratio_lambda = log(1-λ) - log(λ)
```

5. 計算 `log_accept = log_ratio_qaqmc + log_ratio_lambda`。
6. 以 `min(1, exp(log_accept))` 接受。
7. 接受時更新 `B_mask_`、current topology mask，並 reproject affected site。

**邊界情況**

- 若 spin/path 不相容，`log_ratio_qaqmc = -inf`，proposal rejection。
- Endpoint λ 需要特殊處理避免 `log(0)` NaN。

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

1. 設定 underlying Renyi boundary condition 為 `B_mask_`。
2. 呼叫 mask-aware `diagonal_update()`。
3. 呼叫 mask-aware `cluster_update()`。

**邊界情況**

- 若目前 topology 剛接受 toggle，必須已完成 reproject。

**不變量**

- QAQMC update 使用 current `B_mask_`，不是 full `A_mask_`。

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
| current configuration | engine state | 通常應來自 `λ=0` no-swap equilibrium-like state。 |
| `lambda_schedule_` | `double[K+1]` | Forward protocol。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| return value | `WorkTrajectoryResult` | Work、`exp(-w)`、acceptance diagnostics。 |
| engine state | internal | trajectory 結束時位於 final λ/topology。 |

**演算法流程**

1. 檢查 schedule 和 region 已設定。
2. 設定 `w_current_ = 0`。
3. 對每段 `λ_m -> λ_{m+1}`：
   - 呼叫 `accumulate_work(λ_m, λ_{m+1})`。
   - 更新 current λ。
   - 做 random permutation topology sweep。
   - 做 QAQMC diagonal+cluster sweep。
4. 回傳 `w_current_` 和 diagnostics。

**邊界情況**

- 若 final `λ=1` 但 `B_mask_ != A_mask_`，應記錄 warning/diagnostic。

**不變量**

- Work 不跨 trajectories 累積。

**測試**

- `A` 為空時 trajectory work 為 0。
- Very slow protocol 下平均 work 接近 equilibrium free-energy difference。

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
   - Reset 或回到 `λ=0` no-swap sector。
   - 做 `decorrelation_steps`。
   - 呼叫 `run_trajectory()`。
   - 儲存 `w` 和 `exp(-w)`。
2. 計算 `mean_exp_minus_work`。
3. 計算 `S2 = -log(mean_exp_minus_work)`。

**邊界情況**

- 若 `exp(-w)` underflow/overflow，應使用 log-sum-exp。

**不變量**

- 每條 trajectory 的 work accumulator 獨立。

**測試**

- 人工 work samples 檢查 Jarzynski aggregation。

## 邊界情況

- `λ=0` 和 `λ=1` 會造成 `log(0)`，必須用 endpoint-safe logic。
- Jarzynski estimator 對 rare low-work trajectories 敏感；若 `exp(-w)` variance 很大，應輸出 diagnostic。
- 若 topology sweep 在 `λ=1` 前沒有接近 full swap，final estimator 可能 variance 很大。
- 若 `A_mask` 太大，`n_lambda_steps` 和 trajectories 數量需要增加。
- 如果 current topology update 後沒有 reproject site operators，後續 diagonal/cluster update 會在錯誤 boundary condition 下執行。

## 效能備註

- 第一版 topology sweep 使用 `random permutation sweep`，確保每個 A site 每輪 proposal 一次，同時避免固定 ordering bias。
- 每個 `λ` step 的成本約為：

```text
O(|A| * topology_toggle_cost + n_qaqmc_sweeps_per_lambda * QAQMC_sweep_cost)
```

- `n_lambda_steps` 越大，work distribution 通常越窄，但成本線性增加。
- `n_qaqmc_sweeps_per_lambda` 越大，configuration 越能適應 current λ，但成本增加。
- Aggregating `exp(-w)` 應用 log-sum-exp，避免 large work 下 underflow。

## 驗收標準

- `A` 為空時，`S2 ≈ 0`。
- 小系統結果應與 exact diagonalization 或現有 expanded ensemble estimator 相容。
- `λ=0` 起點必須保持 no-swap topology。
- `λ=0.5` 時 topology insert/remove 的 lambda bias ratio 應互為反向且大小為 0。
- 增加 `n_lambda_steps` 時，work variance 應下降或不惡化。
- Forward/reverse protocol 若未來實作，兩者估計應在誤差內一致。

## 測試

- Unit test `log_g` 和 `Δw` endpoint-safe 計算。
- Unit test random permutation sweep attempts 數等於 `|A|`。
- Unit test single-site insert/remove acceptance ratio。
- Small-system trajectory smoke test：`A` 含一個 site，檢查 `B_mask` endpoint behavior。
- Regression test：和現有 expanded ensemble 在 4x4 m=1 小樣本上比較 central value。
- Numerical stability test：大量 work samples 用 log-sum-exp aggregation。

## 相關檔案

- `csrc/qaqmc_renyi_core.hpp`
- `csrc/qaqmc_renyi_core.cpp`
- `csrc/qaqmc_core.hpp`
- `src/engines/qaqmc_renyi.py`
- `src/tee/reweighting.py`
- `src/tee/qaqmc_renyi_ratio.py`
- `paper/Entanglement entropy from nonequilibrium work.md`

## 開放問題

- 第一版是否只支援 forward protocol，或同時支援 reverse protocol 以做 Bennett/Crooks-style diagnostics。
- Endpoint `λ=0/1` 要採 deterministic endpoint 邏輯，還是使用避開端點的 half-step schedule。
- `run_trajectory()` 結束後是否要自動 reset 回 `λ=0`，或交給 `run_trajectories()` 控制。
- `n_qaqmc_sweeps_per_lambda` 第一版是否固定為 1，還是 exposed 給 Python driver。
- Work estimator 是否要同時輸出 cumulant estimate，方便判斷 Jarzynski variance。
