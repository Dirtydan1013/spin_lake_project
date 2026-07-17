# 規格：csrc/cpu/include/qaqmc_renyi_work_core.hpp / csrc/cpu/detail/qaqmc_renyi_work_core.cpp

最後更新：2026-07-17（重寫；引擎已實作完成並在 production 使用 — 舊 spec
的「預計新增」狀態已過時。含 set_cut、warm-start、per-trajectory 診斷）。

## 角色

`QAQMCRenyiWorkEngine` 是 two-replica 非平衡功（Jarzynski）Renyi 引擎：
驅動一個 `Mode::Work` 的 `QAQMCRenyiEngine` backend，沿有限時間 λ 協議在
nested 區域對 `(A_start ⊆ A_end)` 之間內插，回傳

    ΔS₂ = S₂(A_end) − S₂(A_start) = −log ⟨exp(−w)⟩

- `A_start = ∅` 時退化為 D'Emidio（arXiv:2402.05439）的標準 S₂(A_end)。
- 非空 nested 對（如 `(A, A∪B)`）直接量 KP ratio-ladder 的一個 rung。

Python wrapper：`src/engines/qaqmc_renyi_work.py`；production driver：
`src/mpi/qaqmc_renyi_work_mpi.py`（單區域與 KP-regions 模式，含 chunked
checkpoint / warm start / `--backend cuda`）。CUDA 對應：
`src/engines/qaqmc_renyi_work_cuda.py`（協議在 Python 層重實作，backend 為
device-resident `RenyiEngine`）。

## 物件 / 函式

| 名稱 | 種類 | 可見性 | 用途 |
| --- | --- | --- | --- |
| `QAQMCRenyiWorkEngine` | class | public | 外層協議引擎 |
| `WorkTrajectoryResult` | struct | public | 單軌跡：w、exp(−w)、final |B|、unjoined_at_end、topology attempts/accepts |
| `WorkRunResult` | struct | public | 聚合：ΔS₂（log-sum-exp）、work 均值/變異、**per-trajectory 診斷陣列**（work_samples、final_swap_counts、unjoined_counts、attempts/accepts） |
| `set_region_pair(A_start, A_end)` | method | public | 設 nested 對；建 `D = A_end \ A_start`、backend mask=A_start、B=∅ |
| `set_region(A)` | method | public | = `set_region_pair(∅, A)` |
| `set_lambda_schedule(λs)` | method | public | 單調不減、0 起 1 終 |
| `set_cut(m_star)` | method | public | 移 swap 邊界；**vacuum-reset op strings + 作廢 checkpoint**，必須在 thermalize/import 前呼叫 |
| `set_sweeps_per_lambda(n_topo, n_qaqmc)` | method | public | 每 λ 步的 sweep 數（預設各 1，對齊 paper） |
| `export_start_config` / `import_start_config` | method | public | warm start：匯出/還原 A_start-sector 兩 replica op strings（import 前必須先以**相同區域對**呼叫 set_region_pair；import 會 recompute midpoint + 種 checkpoint chain，可跳過 thermalize） |
| `thermalize(n)` | method | public | 在 start sector（mask=A_start、B=∅、λ=0）熱化 |
| `run_trajectory()` | method | public | 跑一條軌跡到 λ=1（caller 保證起始 sector 正確） |
| `run_trajectories(n, decorr)` | method | public | n 條軌跡，軌跡間在 start sector 做 decorr 步 mc_step；log-sum-exp 聚合 |
| `backend()` | method | public | 直接存取內部 `QAQMCRenyiEngine` |

## 物理 / 演算法契約

- 內插配分函數：λ 下每個 D 中的站獨立處於 joined（∈B）或 unjoined 態，
  單站 toggle 的 Metropolis 判準含拓撲比 `(λ/(1−λ))^{±1}` × 物理權重比
  （backend 的 `log_weight_ratio_for_toggle`）。
- 功累積在 λ 步進時發生（`accumulate_work(λ_old, λ_new)`）：joined 站貢獻
  `log(λ_new/λ_old)`、unjoined 站貢獻 `log((1−λ_new)/(1−λ_old))`；
  `λ_new = 1` 時 unjoined 站走 **"multiply by 1"** 處方 —
  貢獻記 0 並累計 `unjoined_at_end_count`。
- **Bias 方向性**：Jarzynski 有限 K 的 dissipation bias 與 multiply-by-1
  bias 都是單邊（ΔS₂ 偏高）；`unjoined_at_end` 與 K-scan 是 production 的
  收斂診斷（E: +0.27@K4 → +0.04@K50 → +0.02@K200 on 2-site gate）。
- 拓撲 sweep 是對 D 站的隨機排列逐站一次 toggle 嘗試
  （`topology_sweep_random_permutation`）；外層自己的 `rng_` 與 backend
  RNG 分離。
- **驗證方法論（E14 教訓）**：單點 end-to-end ED match 可能是
  compensating-bias 假象；正確性的判準是 (a) fixed-λ 平衡 sector 佔據 vs
  ED、(b) ΔS₂ 對 K 無系統趨勢（unbiased kernel 在任何 K 皆無偏）、
  (c) cache==recompute 不變量。

## 模組輸入 / 輸出

建構參數同 backend（N/Ω/δ/Rb/M/ε/seed/pos/cutoff/delta_groups=600/box）。

| 輸出 | Type / Shape | 意義 |
| --- | --- | --- |
| `WorkRunResult.delta_s2` | double | −log-sum-exp 聚合的 ΔS₂ |
| `.work_samples` | `double[n_traj]` | 原始 w（driver 以 chunk 寫入 HDF5；Jarzynski 需要 raw，不可預先平均） |
| `.final_swap_counts` / `.unjoined_counts_per_traj` | `int32[n_traj]` | 收斂診斷 |
| `.topology_attempts/accepts_per_traj` | `int64[n_traj]` | 接受率診斷 |

## 資料契約

- masks 為 `uint8[N]`（0/1）；`A_start & ~A_end` 必須為空（nested）。
- warm-start config = 兩 replica 的 int32 op strings（K-independent、
  region-pair-dependent）；MPI driver 存於 `configs/rank{r}.h5`。
- driver 的 chunk schema（`_RENYI_CHUNK_DATASETS`）：work_samples、
  final_swap_counts、unjoined_counts_per_traj、topology_attempts/accepts。

## 狀態 / 不變量

- 軌跡起點恆為 (λ=0, B=∅, backend mask=A_start)；`reset_to_start_sector`
  由 restore checkpoint（若有效）或 vacuum 達成。
- checkpoint chain：`run_trajectories` 每條軌跡前 decorr + save，故
  checkpoint 永遠是「乾淨的 A_start sector 配置」。
- `set_cut` 後 checkpoint 一定無效（channel 映射變了）。

## 整體流程（run_trajectories）

1. 每條軌跡：restore checkpoint → decorr 步 → save checkpoint。
2. `run_trajectory()`：沿 λ schedule 逐步 — 累積功 → （0<λ<1 時）
   n_topo 次 topology sweep → n_qaqmc 次 backend mc_step。
3. 聚合：`x = −w`，log-sum-exp 求 ⟨e^{−w}⟩，`ΔS₂ = −log⟨e^{−w}⟩`；
   `work_var` 用 ddof=1（n=1 時為 0）。

## 邊界情況

- `D = ∅`（A_start == A_end）：ΔS₂ = 0、軌跡是 no-op（wrapper 直接回零陣列）。
- λ schedule 端點：λ=0/1 不做 toggle 嘗試（`attempt_string_toggle` 要求
  0<λ<1）；CPU 參考在端點仍「訪問」每個提案並確定性拒絕（計入 attempts）—
  CUDA wrapper 對齊此計數慣例。
- 軌跡結束仍 unjoined 的站：見 multiply-by-1 契約；count 必須回報。

## 效能備註

- backend 的 coloring-OpenMP cluster 與 grouped alias 同前；本層開銷
  主要在 per-λ 的 topology sweep（O(|D|) 次 log_weight_ratio，每次
  O(該站 offdiag path 長)）。
- production K=400、decorr=200 的成本結構見
  `docs/design/gpu_acceleration_proposal.md`（GPU 化動機）。

## 驗收標準 / 測試

- `tests/engines/integration/test_qaqmc_renyi_work_vs_ed.py`：
  1D 小系統 ΔS₂ vs ED（K=50 的 nested-pair 子測試 — K=4 會觸發
  multiply-by-1 bias +0.27，是特性不是 bug）。
- K-independence：ΔS₂ 對 K 的系統性趨勢 = biased kernel 的指紋。
- fixed-λ sector 佔據 vs ED（sharp check）。
- MPI driver 的 exact-resume / collective-resume / fingerprint 測試
  （`tests/mpi/`，CUDA backend 路徑）。

## 相關檔案

- `csrc/cpu/include/qaqmc_renyi_core.hpp` — backend（Mode::Work 原語）
- `csrc/cpu/bindings/bindings_renyi_work.cpp`
- `src/engines/qaqmc_renyi_work.py`、`src/engines/qaqmc_renyi_work_cuda.py`
- `src/mpi/qaqmc_renyi_work_mpi.py`（KP 模式 = `src/kp/` 的 region masks）
- `docs/specs/reviews/qaqmc_renyi_work_code_review.md`（2026-05 歷史 review）

## 開放問題

- KP TEE production（δ≈4.25）的 K / n_traj / decorr 最佳配置仍待首跑校準；
  `unjoined_at_end` 分佈是首要監控量。
