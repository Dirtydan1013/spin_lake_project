# QAQMC CPU Memory Optimization：均衡版實作與後續計畫

最後更新：2026-07-14  
分支：`cpu_memory_optimization`  
基準分支：`z2_spin_lake` (`31d6c5c`)  
第一階段範圍：standard single-replica `QAQMCEngine`，`N=216`、full bonds、超長 operator string 與 64-chain CPU production。
當前狀態：**40% 均衡版已實作完成，等待 `cpunode02` 空出後補 64-rank native production gate。**

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
該 node 做 64-rank A/B。這是目前唯一尚未執行的 balanced production gate，
不影響已完成的單-chain 實作與 correctness gates。

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

### 4.2 Cross-process sharing 候選方案

優先評估保留 one-MPI-rank-per-chain 的 read-only mapped table：

1. node-local rank 0 建立 packed model-data blob；
2. 以 Hamiltonian/config hash 作 identity；
3. 其他 local ranks 在 barrier 後 read-only `mmap` 同一份 pages；
4. 每個 NUMA socket可各 first-touch 一份，避免跨 socket remote reads；
5. C++ core 使用 owning storage 或 immutable `TableView`，不強迫依賴 MPI library。

若 mapping/lifetime 過於複雜，第二方案是一個 process 管多條 chains，以 OpenMP
在 chain dimension 平行，所有 chain objects 共用 `QAQMCModelData`。這會影響
現有 Python/MPI output architecture，應在 per-rank balanced version 穩定後再做。

### 4.3 Shared-table node memory 預估

假設每個 NUMA socket 一份 240.39 MiB table：

| `M` | 現況 64 ranks | 均衡 chains + shared tables |
| ---: | ---: | ---: |
| 2.76M | 約 37.5 GiB | 約 7.0 GiB |
| 27.6M | 約 135 GiB | 約 66 GiB |
| 100M | 約 420 GiB | 約 238 GiB |

100M 的 238 GiB 尚未包含 runtime/observable overhead，因此仍不足以安全使用
240 GB node；需要 aggressive event compression 或減少 ranks。

## 5. Optional aggressive memory mode

目前每個 bond endpoint event 儲存 packed `(p,bond,endpoint)` 共 8 bytes。一個
bond operator 產生兩個 endpoint events，所以幾乎占 `16L` bytes。

aggressive mode 改成每個 endpoint 只存 `uint32 p`：

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

另一個折衷是 event `p:uint32[] + bond:uint16[]` SoA；N=216 時每 endpoint
6 bytes，節省較少但能避免 dependent lookup，也應納入 benchmark。

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
- [ ] `cpunode02` 空出後補 64-rank node-throughput A/B。

### Phase 3：immutable model sharing

- [ ] 拆出 `QAQMCModelData` 與 `QAQMCChainState`。
- [ ] 單 process 多 engine 共用 model-data unit test。
- [ ] 選擇 mmap table view 或 threaded multi-chain runner。
- [ ] 實作 configuration hash、ownership/lifetime 與 failure cleanup。
- [ ] 每 NUMA socket一份 model pages，驗證 first-touch/core binding。
- [ ] 1/8/32/64 chains node scaling benchmark。

### Phase 4：optional aggressive events

- [ ] p-only endpoint event prototype。
- [ ] 6-byte logical SoA event prototype。
- [ ] fixed operator string 的 event multiset、segment boundaries exact comparison。
- [ ] `M` ladder memory與 cluster throughput A/B。
- [ ] 只有在 node-level throughput/fit 明確受益時保留 production option。

### Phase 5：擴展至其他 engines

- [ ] 將已驗證的 compact immutable model data套用到 `QAQMCRenyiEngine`。
- [ ] 分開計算 two-replica/channel event memory，不直接沿用 standard 公式。
- [ ] 評估 work engine 與 SSE 是否能共用 geometry/bond tables。
- [ ] 各 engine 必須通過自己的 ED、detailed-balance 與 checkpoint gates。

## 7. Correctness tests

當前結果：

- baseline/optimized 在同 build flags、seed、`M=4096` 下連續 20 步的
  `op_types`、`op_sites` 與 serialized RNG state 每步完全一致。
- exported delta schedule bitwise exact；small profile 的 density/Z/C/p-index arrays exact。
- compact-layout/profile-grid 11 tests 在 Release 與 `_GLIBCXX_ASSERTIONS` build 都通過。
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
- 64 chains、M=27.6M engine core目標低於 70 GiB/node。
- shared-table synchronization與mapping不能成為每-step overhead。
- node throughput至少持平，理想目標提高 10–30%（來自 LLC/NUMA reuse）。

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
- [ ] node-local shared immutable model-data path（Phase 3）。
- [ ] optional aggressive event representation 與 A/B 結果（Phase 4）。
- [ ] 1/8/32/64-chain node scaling benchmark 與 NUMA/core-binding 說明。

只有 balanced gates 全部通過後，才把 compact representation設成 default；
shared model與 aggressive events 仍各自保留獨立 rollback point。
