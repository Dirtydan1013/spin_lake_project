# QAQMC CPU Memory Optimization：均衡版實作與後續計畫

最後更新：2026-07-16
分支：`cpu_memory_optimization`  
基準分支：`z2_spin_lake` (`31d6c5c`)  
第一階段範圍：standard single-replica `QAQMCEngine`，`N=216`、full bonds、超長 operator string 與 64-chain CPU production。
當前狀態：**全部 production gates 完成。** 均衡版、process-local shared model、
多 chain threaded runner 與三種 event layouts 均已實作；單 socket 與 `M=100M`
fit gate 已完成。`cpunode02` 的 64-rank A/B 與 4-socket NUMA gate 已於
2026-07-16 通過（jobs `26784`/`26785`，M=2.76M；production-M 確認
`26786`/`26787`，M=27.6M）——結果見 §1.4。原 jobs `26731`/`26732` 因
排隊期間共用 working tree 被切回 `z2_spin_lake` 而失敗（probe scripts
不在該分支）；重跑改從 dedicated worktree 提交，job 引用的 checkout 不可變。

CPU native source 已集中到 `csrc/cpu/{include,detail,bindings}`（2026-07-16
目錄整併後的佈局；`qaqmc_cpp` public API 與 Python/MPI imports 不變）。

## 1. 目標與原則

本工作的主要目標不是用速度換記憶體，而是先移除冗餘資料、縮小 lossless
representation，並讓相同 Hamiltonian 的 chains 共用 immutable tables。

優先順序如下：

1. 保持 Markov transition、RNG draw count、checkpoint schema 與 observables 正確。
2. 均衡版本在大 `M` 將 steady-state engine memory 降低約 40%。
3. 單 rank `mc_step()` 不得慢超過 3%，整個 CPU node throughput 不得慢超過 5%。
4. 進一步讓 node-local chains 共用 Hamiltonian tables，降低 64-rank duplication。
5. 將可能影響 cluster 速度的壓縮隔離為 optional aggressive mode。

第一階段不改變 QAQMC 物理、更新順序、acceptance rule、site-label permutation
或 sample/checkpoint cadence。GPU backend、SSE、Rényi/work engine 不在第一個
implementation gate 內；等 standard engine 驗證完成後再共用安全的資料型別與
model-data abstraction。

### 1.1 已交付的 40% 均衡版

- standard engine 的 operator type 改為 `int8`，operator location 與 alias index
  依可表示範圍自動選 `uint16`/`uint32`；`N=384` full-bond fallback 已測。
- grouped alias table 改為 compact SoA，移除無 hot-path reader 的
  `bond_W_max_all` 與可由 sampled index 推導的 location kind。
- 移除常駐 `2M` double delta schedule；C++ 按 slice 計算，public API 仍可按需匯出。
- 限制 event scratch vector 的 retained headroom，避免波動後長期保留近 2× capacity。
- Python wrapper 不再常駐第二份 full int32 operator string；checkpoint/API
  仍匯出原有 int32 schema。MPI profile runtime 只建立實際量測點；
  HDF5 保留舊的 full-ramp dataset，但改為有界 chunk 串流寫入。
- 新增 fresh-process RSS/capacity/timing probe 及 compact layout、overflow fallback、
  checkpoint replay、profile-grid tests。

### 1.2 Compute-node A/B 結果

測試於空閒的 `cpunode01` 單核進行，baseline/optimized 使用相同
portable `x86-64` Release build、`6x6 kagome_bond`、periodic、`N=216`、full
bonds、`G=600`、seed 42。RSS 在 warmup 後、operator export 前量測。

| `M` | baseline RSS | optimized RSS | process RSS 節省 | baseline s/step | optimized s/step | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.76M | 637.61 MiB | 385.04 MiB | **39.61%** | 0.39360 | 0.33298 | **1.182×** |
| 27.6M | 2201.52 MiB | 1332.60 MiB | **39.47%** | 3.68268 | 3.31169 | **1.112×** |

扣除 Python/MPI process baseline 後，engine incremental RSS 分別降低 42.14% 與
40.13%。`M=27.6M` optimized logical/capacity core bytes 為 1,353,982,955 /
1,368,392,311 bytes，顯示實際 RSS 與核心 allocation 計算一致。這個均衡版
不但沒有速度回歸，在這兩個大 `M` 上還快 11–18%。

`cpunode02` 當時正在執行本帳號的 64-core production job，因此沒有搶佔
該 node 做 64-rank A/B。該 gate 已於 2026-07-16 補齊，結果見 §1.4——
64-rank 下均衡版不但沒有 regression，反而比 baseline 快 21%，RSS 省 39%。

### 1.4 cpunode02 64-rank / 4-socket NUMA gate（2026-07-16）

執行環境：`cpunode02`（4 sockets、64 physical cores / 128 HT、256 GB）、
exclusive node、portable x86-64 Release build（worktree `build_gate`，branch
head `eea6e75`，28 個 focused tests 先通過）。baseline = `31d6c5c` 舊引擎。

**64 獨立 MPI ranks A/B（job `26784`，`M=2.76M`，`--bind-to core`）：**

| variant | median s/step | slowest s/step | node steps/s | rank RSS (max) | node RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline packed64 | 1.0296 | 1.2113 | 62.16 | 650.2 MiB | 40.61 GiB |
| optimized packed64 | 0.8496 | 0.9825 | 75.33 (**+21.2%**) | 396.4 MiB (**−39.0%**) | 24.71 GiB (**−39.2%**) |
| optimized `p_bond16` | 0.8457 | 0.9568 | 75.68 (**+21.8%**) | 374.8 MiB (**−42.4%**) | 23.39 GiB |
| optimized `p_only32` | 1.4252 | 1.6426 | 44.91 (−27.7%) | 354.7 MiB | 22.08 GiB |

- balanced acceptance「64-rank chain-steps/s regression ≤5%」**通過**（實際 +21%）；
  RSS/rank 節省 39–42%，與單核 A/B 一致。
- `p_only32` 在 64-rank 競爭下 slowdown 放大到 −28%（單核只 −10%）——
  dependent random load 在記憶體頻寬飽和時代價加倍，**確認只作 RAM 硬上限
  的最後手段**，不進 production 預設。

**4 ranks × socket、shared-model threaded batch（job `26785`，`M=2.76M`，
`--map-by ppr:1:socket:PE=16`）：**

| storage | chains/rank | total | node chain-steps/s | node RSS |
| --- | ---: | ---: | ---: | ---: |
| packed64 | 1 | 4 | 8.19 | 1.88 GiB |
| packed64 | 4 | 16 | 34.82 | 3.14 GiB |
| packed64 | 8 | 32 | 54.15 | 4.80 GiB |
| packed64 | 16 | 64 | 74.23 | 8.15 GiB |
| `p_bond16` | 16 | 64 | **76.61** | **6.80 GiB** |
| `p_only32` | 16 | 64 | 42.19 | 5.48 GiB |

- 等量 64 chains 比較：shared-model `p_bond16` 76.61 steps/s @ 6.80 GiB，
  vs 64 獨立 optimized ranks 75.33 @ 24.71 GiB，vs 64 獨立 baseline ranks
  62.16 @ 40.61 GiB —— **throughput 持平偏快（+1.7%），node RSS 再省 72%**；
  對 baseline 是 +23% throughput、**RSS 1/6**。shared-model acceptance
  「throughput 至少持平」通過。
- marginal RSS ≈ 85–107 MiB/chain（`p_bond16`/packed64），確認 immutable
  model pages 沒有隨 chain 數線性複製。
**Production-M 確認（`M=27.6M`，jobs `26786`/`26787`）：**

64 獨立 MPI ranks A/B（job `26786`）：

| variant | median s/step | node steps/s | rank RSS (max) | node RSS |
| --- | ---: | ---: | ---: | ---: |
| baseline packed64 | 9.4752 | 6.755 | 2213.9 MiB | 138.26 GiB |
| optimized packed64 | 8.2026 | 7.802 (**+15.5%**) | 1344.7 MiB (**−39.3%**) | 83.94 GiB |
| optimized `p_bond16` | 8.0911 | 7.910 (**+17.1%**) | 1135.0 MiB (**−48.7%**) | 70.87 GiB |
| optimized `p_only32` | 13.5229 | 4.733 (−29.9%) | 923.9 MiB | 57.65 GiB |

→ balanced acceptance「`M>=27.6M` memory/rank ≥38% 節省」與「64-rank steps/s
regression ≤5%」皆通過（實際 −39.3% RSS、**+15.5%** steps/s）。

64-chain shared-model（job `26787`，4 ranks × 16 chains，warmup 1 / steps 3）：

| storage | node chain-steps/s | rank RSS (max) | node RSS | model/socket |
| --- | ---: | ---: | ---: | ---: |
| packed64 | 6.884 | 17.26 GiB | **67.40 GiB** | 240.93 MiB |
| `p_bond16` | 7.076 | 13.92 GiB | **54.34 GiB** | 240.93 MiB |

→ shared-model acceptance「64 chains、`M=27.6M` engine core < 70 GiB/node」
**通過**（packed64 全 process RSS 67.4 GiB 已含 4 份 Python runtime；
`p_bond16` 54.3 GiB 且餘裕充足）。

**誠實註記（throughput nuance）**：在 `M=2.76M` 時 shared-model 與 64 獨立
ranks throughput 持平（76.6 vs 75.7 steps/s）；但在 production `M=27.6M`
shared-model 比 64 獨立 optimized ranks 慢 ~10–12%（7.08 vs 7.91 steps/s，
`p_bond16`），只比 64 獨立 **baseline** ranks 快 +4.7%。大 M 下 working set
遠超 LLC，per-socket 16 threads 的頻寬競爭抵銷了 shared-table reuse。
結論：**`M≈27.6M` 在 256 GB node 上 production 首選仍是 64 獨立 optimized
`p_bond16` ranks（70.9 GiB、最快）**；shared-model 的定位是 RAM-bound regime
（`M≳100M`，64 獨立 ranks ≈213 GiB 放不下，shared-model 可以），
用 ~10% throughput 換 chains 能不能開得起來。

### 1.3 Shared-model 與大 M 實測結論

Phase 3 採用「每個 NUMA socket 一個 MPI process、process 內以 threads 跑多條
chains」；所有 chains 共用一份 240.93 MiB immutable model，operator string、
RNG、event scratch 與 observables 仍完全獨立。C++ transition/profile path 會釋放
Python GIL，因此 chain kernels 確實可同時執行。

在 `cpunode01`（8 physical cores / 16 hardware threads）、`M=2.76M`、`packed64`
下的結果：

| chains/process | chain-steps/s | dominant resident | process RSS |
| ---: | ---: | ---: | ---: |
| 1 | 2.784 | 347.40 MiB | 466.27 MiB |
| 2 | 5.882 | 453.87 MiB | 573.94 MiB |
| 4 | 10.370 | 666.80 MiB | 787.69 MiB |
| 8 | 15.484 | 1092.67 MiB | 1212.43 MiB |
| 16 | 17.716 | 1944.41 MiB | 2059.90 MiB |

16 chains 相對單 chain throughput 為 **6.36×**；超過 8 physical cores 後因 SMT
而趨於飽和。若用 16 個獨立 B=1 processes 外推，RSS 約 7.29 GiB；shared-model
B=16 實測 2.01 GiB，node process RSS 約省 **72%**。這項節省主要來自只保留
一份 model 與一份 Python runtime，不是改變 Markov chain。

同一個 B=16 gate 使用 `p_bond16` 時為 18.27 chain-steps/s、1720.38 MiB RSS；
相對 `packed64` 再省 16.5% RSS，throughput 沒有下降。這支持目前 N=216
production 採 `p_bond16`，但 64-core node 結論仍以 pending NUMA gate 為準。

單 chain 大 M fit 結果如下；數值為 warmup 後 RSS：

| `M` | layout | RSS | 相對 packed 節省 | chain-step/s | 相對 packed |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2.76M | `packed64` | 466.27 MiB | — | 2.784 | — |
| 2.76M | `p_bond16` | 446.00 MiB | 4.3% | 2.828 | +1.6% |
| 2.76M | `p_only32` | 423.59 MiB | 9.2% | 2.490 | -10.5% |
| 27.6M | `packed64` | 1416.59 MiB | — | 0.2912 | — |
| 27.6M | `p_bond16` | 1205.59 MiB | 14.9% | 0.2970 | +2.0% |
| 27.6M | `p_only32` | 995.54 MiB | 29.7% | 0.1984 | -31.9% |
| 100M | `packed64` | 4174.41 MiB | — | 0.07938 | — |
| 100M | `p_bond16` | 3415.61 MiB | **18.2%** | 0.07846 | -1.2% |
| 100M | `p_only32` | 2652.67 MiB | **36.5%** | 0.05595 | -29.5% |

raw C++/Python API 為 backward compatibility 仍以 `packed64` 為預設。對目前
`N=216` production，**建議使用 `p_bond16`**：大 M 省 15–18% RSS，未觀察到
顯著 throughput regression。`p_only32` 只在 RAM 是 hard limit 時使用。

## 2. 現況記憶體模型

令：

```text
L   = M_total = 2M
f_b = bond operator fraction
f_s = 1 - f_b = site operator fraction
```

對 `6x6 kagome_bond`、`N=216`、full bonds、`G=600`：

```text
n_bonds  = 23,220
max_alias = N + n_bonds = 23,436
```

使用 production geometry、`M=276,000` 的 12-sweep 短 probe 得到：

```text
f_b ≈ 0.996688
f_s ≈ 0.003312
```

這不是 observable equilibration 結果，只用來估算 event-list allocation；它顯示
目前 full-bond setup 幾乎達到 bond-event memory 的最壞情況。正式實作必須在
target production run 重新記錄 `f_b` 與 vector capacity。

### 2.1 每個 rank 重複的 immutable grouped-alias tables

| Array | 現況大小 |
| --- | ---: |
| `entries[G,max_alias]`，16 bytes/entry | 214.56 MiB |
| `bond_W_rmax_all[G,n_bonds]`，double | 106.29 MiB |
| `bond_W_max_all[G,n_bonds]`，double | 106.29 MiB |
| 合計 | **427.15 MiB/rank** |

`bond_W_max_all` 在 grouped table 建好後沒有 standard engine hot-path reader；
更新只使用 reciprocal `bond_W_rmax_all`。因此其中 106.29 MiB/rank 是第一個
可以無條件移除的目標。

`AliasEntry` 目前包含：

```cpp
double  prob;
int32_t alias;
int32_t loc_kind;
```

但 item ordering 固定為前 `N` 個 site operators，後面為 bond operators，所以
`loc_kind` 可以由 sampled alias index 精確推導，不必逐 entry 儲存。

### 2.2 每條 chain 的 `O(L)` mutable arrays

| Array | Bytes |
| --- | ---: |
| `delta_sched_` (`double`) | `8L` |
| `op_types_` (`int32`) | `4L` |
| `op_sites_` (`int32`) | `4L` |
| `bond_spin_` (`int8`) | `1L` |
| `site_op_list_` (`int32`) | `4 f_s L` |
| `site_bond_list_`（兩個 8-byte endpoint events/bond op） | `16 f_b L` |

合計：

```text
current mutable bytes
  = (17 + 4 f_s + 16 f_b)L
  = (21 + 12 f_b)L
  ≈ 32.96L bytes  when f_b = 0.996688
```

這裡尚未包含 vector over-capacity、allocator metadata、Python/MPI runtime、
profile observables、HDF5 buffers 與 constructor temporary allocations。正式
memory probe 必須同時報告 logical size、capacity、RSS 與 peak RSS。

### 2.3 目前 standard engine 的估計 steady-state memory

| `M` (half-ramp) | `L=2M` | 目前 core memory/rank |
| ---: | ---: | ---: |
| 2.76M | 5.52M | 約 0.59 GiB |
| 27.6M | 55.2M | 約 2.11 GiB |
| 100M | 200M | 約 6.56 GiB |

因此 `M=100M` 若開 64 個完全獨立 ranks，僅 engine core 就約 420 GiB；現有
240 GB node 無法容納。repo 的 100M probe 使用 24 ranks，與這個估計一致。

## 3. 均衡版本設計

均衡版本保留目前 64-bit packed bond event，避免把 `op_sites_[p]` 的 dependent
random load 放回 cluster hot loop。目標 changes 如下。

### 3.1 移除 unused grouped envelope maxima

- 從 standard engine 的 `GroupedAlias` 移除 `bond_W_max_all` ownership。
- 建表時只保留 `bond_W_rmax_all`。
- 不先刪除 legacy/public `AliasTable::bond_W_max_all`，因 Python fallback 與其他
  engines 仍可能使用；待個別 call-site audit 後再處理。
- 預期節省：106.29 MiB/rank，steady-state update 速度不變。

### 3.2 Compact alias representation

- 將 `AliasEntry` AoS 拆成：
  - `alias_prob`：`double[]`；
  - `alias_index`：adaptive `uint16[]` 或 `uint32[]`。
- 刪除 `loc_kind`，由 final sampled index 推導：

  ```text
  idx < N  -> site operator at site idx
  idx >= N -> bond operator at bond (idx - N)
  ```

- `N=216` 時 `max_alias=23,436 < 65,536`，alias index 可用 `uint16`。
- `N=384` full bonds 時 `max_alias=73,920`，必須自動 fallback 到 `uint32`。
- 這項改動不改變 alias probabilities、RNG calls 或 sampled item。

對目前 `N=216`，grouped tables 可由 427.15 MiB 降為約：

```text
alias_prob  double : 107.28 MiB
alias_index uint16 :  26.82 MiB
bond rmax   double : 106.29 MiB
total              : 240.39 MiB
```

### 3.3 Compact operator string

- `op_types_` 由 `int32` 改為 `int8`；合法值只有 `-1, 1, 2`。
- `N=216` full bonds 的 `op_sites_` 使用 `uint16`。
- 當 site/bond index 超過 65,535 時自動 fallback 到 `uint32`。
- pybind/API 邊界維持既有 NumPy/checkpoint contract；輸入時驗證並壓縮，輸出時
  可轉回相容 dtype。
- off-diagonal seam、checkpoint restore、measurement 與 tests 的所有 accessors
  必須經 typed helper，避免散落 implicit narrowing conversions。

### 3.4 移除 per-chain `delta_sched_[L]`

- 以既有 `delta_at(p)` 公式產生 slice delta。
- constructor 的 group boundary/midpoint sampling 改呼叫 `delta_at(p)`。
- dimer delta lookup、off-diagonal half-line 與 exported schedule 改為按需產生。
- cluster 目前本來就使用 `delta_at(p)`，主要 benchmark 風險在 diagonal 的
  sequential schedule read 改成 arithmetic。
- 加入 all-slices sampled/bitwise comparison，確認同 architecture/build 下與目前
  schedule 表達式一致；若 compiler rounding 無法維持 bit identity，至少必須通過
  acceptance-boundary 與 statistical gates。

### 3.5 均衡版本預期記憶體

保留 8-byte bond events 時：

```text
balanced mutable bytes
  = op_type(1L) + op_site(2L) + bond_spin(1L)
    + site_events(4 f_s L) + bond_events(16 f_b L)
  = (8 + 12 f_b)L
  ≈ 19.96L bytes
```

| `M` | 現況/rank | 均衡版本/rank | 節省 |
| ---: | ---: | ---: | ---: |
| 2.76M | 0.59 GiB | 約 0.34 GiB | 約 42% |
| 27.6M | 2.11 GiB | 約 1.26 GiB | 約 40% |
| 100M | 6.56 GiB | 約 3.95 GiB | 約 40% |

`N=384` 因 alias/operator indices 必須使用 32-bit，收益會比表中低；memory
probe 必須另外建立 8x8/full-bond baseline，不能套用 N=216 數字。

## 4. Node-shared immutable model data

每個 MPI rank 現在都建立完全相同的 geometry、bond data 與 grouped-alias tables。
即使均衡版本降到 240.39 MiB，64 ranks 仍會重複約 15 GiB 相同資料，也讓
LLC/NUMA traffic 無法有效共享。

### 4.1 內部 ownership refactor

先將 engine 拆成兩層：

```text
QAQMCModelData (immutable, shareable)
  geometry / bonds / inv_coord / grouped alias / ramp parameters

QAQMCChainState (mutable, one per chain)
  RNG / operator string / counts / event lists / bond_spin / observables
```

單 engine API 可持有 `shared_ptr<const QAQMCModelData>`，確保 non-MPI usage 不變。

### 4.2 已選擇的 sharing architecture

實作選擇一個 process 管多條 chains，而非 cross-process `mmap`：

1. 第一條 chain 建立 `QAQMCModelData`；
2. 後續 `QAQMCEngine(model_data, seed)` 只配置自己的 mutable state；
3. `QAQMCSharedModelBatch` 使用 persistent thread pool，一個 worker 對應一條 chain；
4. 每個 NUMA socket 啟動一個 MPI rank，使 model pages 與 chain scratch first-touch
   留在 socket local memory；
5. 每條 lane 使用 deterministic `seed + lane * seed_stride`，profile result schema
   與單 chain API 不變。

這條路徑不需要 shared-file identity、configuration hash 或 stale-mapping cleanup；
ownership 由 `shared_ptr` 管理，最後一條 chain 結束時才釋放 model。代價是 production
launcher 要由 64 ranks 改成 4 MPI ranks × 16 threaded chains。既有 one-rank-per-chain
CLI 完全保留，可逐批遷移。

### 4.3 Shared-table node memory

假設每個 NUMA socket 一份 240.39 MiB table：

| `M` | 現況 64 ranks | 均衡 chains + shared tables |
| ---: | ---: | ---: |
| 2.76M | 約 37.5 GiB | 約 7.0 GiB |
| 27.6M | 約 135 GiB | 約 66 GiB |
| 100M | 約 420 GiB | 約 238 GiB |

100M 的 238 GiB 尚未包含 runtime/observable overhead，因此仍不足以安全使用
240 GB node；需要 aggressive event compression 或減少 ranks。

上述表格是設計期保守估值；實作後應以 probe 的
`shared_model_bytes + sum(per_chain_capacity_bytes)` 與 process RSS 為準。
`cpunode01` 的 1/2/4/8/16-chain scaling 已完成；`cpunode02` 的 4 ranks ×
1/2/4/8/16 chains 與獨立 64-rank A/B 已由 jobs `26785`/`26784` 完成
（原 `26731`/`26732` 因 shared-tree branch switch 失敗後重跑），
結果見 §1.4。

## 5. Optional aggressive memory mode

目前每個 bond endpoint event 儲存 packed `(p,bond,endpoint)` 共 8 bytes。一個
bond operator 產生兩個 endpoint events，所以幾乎占 `16L` bytes。

最小的 aggressive mode 改成每個 endpoint 只存 `uint32 p`：

- `bond = op_sites_[p]`；
- endpoint 由 owning site 與 `bond_sites[bond]` 比較得出；
- 每個 bond operator 的 event memory 由 16 bytes 降到 8 bytes。

預期 mutable memory：

```text
aggressive mutable bytes
  = (8 + 4 f_b)L
  ≈ 11.99L bytes
```

| `M` | 現況/rank | aggressive/rank | 節省 |
| ---: | ---: | ---: | ---: |
| 2.76M | 0.59 GiB | 約 0.30 GiB | 約 50% |
| 27.6M | 2.11 GiB | 約 0.85 GiB | 約 60% |
| 100M | 6.56 GiB | 約 2.47 GiB | 約 62% |

配合兩份 NUMA-shared tables，64 chains、M=100M 的 engine core 約 143 GiB，
才有足夠空間容納 Python/MPI/profile/HDF5 overhead。

此模式會把 dependent `op_sites_[p]` random load 放回 segment Metropolis 與 commit
路徑，可能降低 cluster throughput。因此它必須：

- 為獨立 compile/runtime option；
- 與 balanced 8-byte packed event 做相同 allocation 下的 A/B benchmark；
- 不作為預設值，除非大-M node throughput 沒有顯著退步。

另有 `p_bond16` 折衷：event 使用 `p:uint32[] + bond:uint16[]` SoA；當
`n_bonds <= 65535` 時每 endpoint 6 bytes，避免 `op_sites_[p]` dependent lookup，
endpoint 仍由 owning site 推導。超出 16-bit bond range 時 constructor 會拒絕，
不做 silent narrowing。三種 layouts 都保持 event order、RNG state 與逐步
trajectory exact。`p_bond16` 在 `M=2.76M/27.6M/100M` 分別省
4.3%/14.9%/18.2% process RSS，throughput 差異為 +1.6%/+2.0%/-1.2%，
通過 3% single-chain gate。

## 6. 實作順序

### Phase 0：可重現 memory/performance baseline

- [x] 新增 standard QAQMC memory probe script。
- [x] 報告每個核心 vector 的 `size × sizeof(T)` 與 `capacity × sizeof(T)`。
- [x] 報告 process `VmRSS`、`VmHWM`／`ru_maxrss`。
- [x] 報告 `f_b`、`f_s` 與 event counts；不在 production hot loop 增加 alias-attempt counter。
- [x] 記錄 init、diagonal、cluster 與 full-step wall time。
- [x] 完成 `M=2.76M / 27.6M` 大尺度 A/B；100M 保留為後續 fit probe。
- [x] 保存相同 seed 的 operator string、RNG state 與 phase timing reference。

### Phase 1：零風險 table cleanup

- [x] 移除 standard `GroupedAlias::bond_W_max_all`。
- [x] 由 alias index 推導 site/bond location。
- [x] compact alias arrays，同時支援 safe uint16 與 general uint32 fallback。
- [x] 驗證 RNG draw ordering 與 20-step same-seed exact trajectory。
- [x] 驗證 init RSS 與 steady-state RSS。

### Phase 2：均衡 per-chain compression

- [x] 導入 `OpType=int8` internal representation。
- [x] 導入 adaptive `OpIndex16/32` representation。
- [x] 維持 Python/HDF5/checkpoint backward compatibility。
- [x] 移除 materialized `delta_sched_`，完成所有 standard-engine call-site migration。
- [x] 通過 standard、profile、off-diagonal seam/string 與 ED regression tests。
- [x] 在 `cpunode01` 完成單核 production-scale memory/performance A/B。
- [x] `cpunode02` 64-rank node-throughput A/B（job `26784`，§1.4：+21% steps/s、−39% RSS）。

### Phase 3：immutable model sharing

- [x] 拆出 immutable `QAQMCModelData`；mutable chain state 保留在 `QAQMCEngine`。
- [x] 單 process 多 engine 共用 model-data unit test。
- [x] 選擇並實作 threaded multi-chain runner。
- [x] 以 `shared_ptr` 實作 ownership/lifetime 與 exception cleanup。
- [x] 完成單 socket 1/2/4/8/16 chains scaling benchmark。
- [ ] 在 `cpunode02` 驗證每 NUMA socket 一份 model、4×16 core binding。

### Phase 4：optional aggressive events

- [x] p-only endpoint event prototype。
- [x] 6-byte `p_bond16` SoA event prototype與 16-bit range guard。
- [x] same-seed operator string、RNG、逐 step trajectory exact comparison。
- [x] `p_only32` 完成 `M=2.76M/27.6M/100M` memory與 throughput A/B。
- [x] `p_bond16` 完成 `M=2.76M/27.6M/100M` A/B；列為 N=216 建議模式。
- [x] `p_only32` slowdown 超過 10%，保留 production option但不設為預設。

### Phase 5：擴展至其他 engines

- [ ] 將已驗證的 compact immutable model data套用到 `QAQMCRenyiEngine`。
- [ ] 分開計算 two-replica/channel event memory，不直接沿用 standard 公式。
- [ ] 評估 work engine 與 SSE 是否能共用 geometry/bond tables。
- [ ] 各 engine 必須通過自己的 ED、detailed-balance 與 checkpoint gates。

## 7. Correctness tests

當前結果：

- portable Release build 的 `tests/engines tests/mpi` 完整 suite：**106 passed**。
- baseline/optimized 在同 build flags、seed、`M=4096` 下連續 20 步的
  `op_types`、`op_sites` 與 serialized RNG state 每步完全一致。
- exported delta schedule bitwise exact；small profile 的 density/Z/C/p-index arrays exact。
- modular API、CPU→CUDA bridge、compact/shared-model/event-layout、四個 engines
  與 profile-grid 的 focused gate：portable Release 與 `_GLIBCXX_ASSERTIONS`
  builds 都是 **32 passed**。
- seam、half-line/ED sector residence、delta-groups-vs-ED、two-site-vs-ED、
  string ED/Jarzynski 與 fidelity-susceptibility calibration regression 通過。
- 4×4 periodic profile CLI 已完成單 rank 取樣、HDF5 與 final-config 輸出。
- 環境缺少 `libasan` runtime，因此沒有宣稱 ASAN gate 通過；改用
  `_GLIBCXX_ASSERTIONS` 進行 bounds/invariant 建置驗證。

### 7.1 Bit-exact structural gates

在相同 compiler、architecture、seed 與初始 operator string 下比較 baseline：

- 每一步 `op_types` 與 `op_sites`。
- serialized RNG state。
- midpoint state、profile states 與 worldline closure。
- per-site site/bond event lists。
- `bond_spin[p]`。
- segment boundaries、proposal/accept counts。
- checkpoint save/load 後的下一步 trajectory。

若移除 delta table 因 floating-point code generation 無法 bit-identical，必須先定位
差異來源；不可只因 final observable 接近就忽略 acceptance-boundary changes。

### 7.2 Existing regression gates

至少包含：

- `tests/engines/unit/test_ed_core_qaqmc_midpoint.py`
- `tests/engines/unit/test_qaqmc_string_seam.py`
- `tests/engines/unit/test_qaqmc_string_halfline.py`
- standard/profile integration tests。
- off-diagonal string vs ED tests（compact op storage 會影響此路徑）。
- MPI checkpoint/warm-start integration tests。

### 7.3 Statistical gates

- small-N diagonal conditional distribution vs exact weights。
- cluster segment Metropolis frequency vs exact ratio。
- density、loop/string、VBS/SS/SF vs baseline within statistical uncertainty。
- site-order permutation、多 rank independent-seed reproducibility。

## 8. Performance與驗收門檻

所有比較必須在同一 node、相同 affinity、相同 rank count、相同 initial state 下進行。
同時報告 median、slowest rank 與 node aggregate throughput。

### Balanced version acceptance

- `N=216, M>=27.6M` steady-state engine core memory/rank至少降低 38%。
- 單 rank full `mc_step` regression不超過 3%。
- 64-rank chain-steps/s regression不超過 5%。
- same-build same-seed trajectory維持 bit-identical，或任何例外有逐項證明與測試。
- 無 checkpoint/HDF5 schema regression。

### Shared-model acceptance

- 同 Hamiltonian 的 node-local table physical pages不得隨 chain count線性增加。
  ✅（§1.4：marginal ≈85–107 MiB/chain @2.76M，model 每 socket 一份 240.9 MiB）
- 64 chains、M=27.6M engine core目標低於 70 GiB/node。
  ✅（§1.4 job `26787`：packed64 67.4 GiB、`p_bond16` 54.3 GiB 全 process RSS）
- shared-table synchronization與mapping不能成為每-step overhead。✅
- node throughput至少持平，理想目標提高 10–30%（來自 LLC/NUMA reuse）。
  ⚠️ 部分達成：`M=2.76M` 持平（+1.7% vs 獨立 optimized ranks）；`M=27.6M`
  −10–12% vs 獨立 optimized ranks（+4.7% vs 獨立 baseline）。LLC reuse 紅利
  在大 M 被 socket 頻寬競爭抵銷——見 §1.4 誠實註記與 production 建議。

### Aggressive-mode acceptance

- 64 chains、M=100M engine core目標低於 160 GiB/node。
- cluster update slowdown若超過 10%，預設 production仍使用 balanced mode。
- 最終選擇依「可完成的 chains × steps/s」而非單 rank latency。

## 9. 主要風險與對策

| 風險 | 對策 |
| --- | --- |
| compact type 造成 narrowing/overflow | constructor hard bounds、adaptive 16/32、boundary tests |
| alias layout 改變 sampled trajectory | 保持 probability/RNG call ordering，逐 step exact comparison |
| delta arithmetic 改變 rounding | all-slice comparison、acceptance-boundary tests、必要時保留 speed mode schedule |
| shared mmap lifetime或stale data | config hash、read-only mapping、atomic publish、job cleanup |
| NUMA shared pages變成 remote traffic | 每 socket一份、first-touch、core binding benchmark |
| vector capacity與RSS不一致 | 同時報告 logical bytes、capacity bytes、RSS、peak RSS |
| aggressive event壓縮拖慢 cluster | 獨立 option、phase timing、go/no-go threshold |
| standard refactor破壞 seam/Rényi code | typed accessors、分階段 migration、完整 ED regression |

## 10. 交付項目

- [x] 可重現的 CPU memory/runtime probe。
- [x] per-array logical/capacity memory report。
- [x] balanced compact standard engine。
- [x] backward-compatible Python/checkpoint interface。
- [x] correctness、RSS 與單-chain performance regression tests。
- [x] README usage 說明。
- [x] node-local shared immutable model-data path（Phase 3）。
- [x] optional aggressive event representations（Phase 4）。
- [x] CPU public/detail/pybind source tree 模組化與 legacy header compatibility gate。
- [x] 1/2/4/8/16-chain single-socket scaling benchmark。
- [x] `cpunode02` 64-rank與64-chain NUMA/core-binding benchmark
  （jobs `26784`/`26785` @ M=2.76M、`26786`/`26787` @ M=27.6M；§1.4）。

balanced compact representation維持 default；shared model是 opt-in batch API，
aggressive events是 runtime option，三者都有獨立 rollback surface。
