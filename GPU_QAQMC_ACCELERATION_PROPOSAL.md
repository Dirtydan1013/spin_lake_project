# QAQMC GPU 加速提案：從 prefix-XOR diagonal update 到 device-resident cluster update

狀態：討論稿（`gpu_version` branch）  
範圍：`QAQMCEngine` single-replica diagonal/profile engine，先不涵蓋 Renyi、work protocol 與 off-diagonal string seam  
基準版本：`z2_spin_lake` commit `31d6c5c`

## 1. 結論先行

這個 engine 可以 GPU 化，而且不應只採用「很多獨立 chains 模擬很多 CPU MPI ranks」的做法。
production 使用的 operator string 很長（已見 `M=2.76e6`、`2.76e7`、`1e8`），單一 chain 本身就提供了大量 time-slice parallelism。

最可能成功的架構是：

1. 保留 CPU `QAQMCEngine` 作為 reference backend。
2. 新增獨立的 `qaqmc_cuda` backend，operator string 在整段 run 中常駐 GPU。
3. 把 diagonal state propagation 改寫為 packed-bitset prefix XOR。
4. 在取得每個 slice 的 prefix state 後，同一條 chain 的所有 diagonal slots 可獨立 resample。
5. cluster update 使用 GPU 建立按 site、按 `p` 排序的 event lists；site 之間維持必要的序列順序，同一 site 內的 segments 平行做 weight reduction、accept/reject 與 spin-cache 更新。
6. profile observables 與 bin accumulation 留在 GPU，只把 checkpoint/bin 結果搬回 host。
7. 小 `M` 或 work-trajectory workload 再加 batch-of-chains；它是第二層平行，不是唯一平行來源。

不建議只移植 diagonal update 後在每個 MC step 把 operator string 搬回 CPU。PCIe transfer 會吃掉收益，而且現有大 `M` 計時顯示 cluster 已占 67–76%；只加速 diagonal 的端到端理論上限約為 1.3–1.5 倍。

## 2. 現有 CPU engine 的實際資料流

一次 `mc_step()` 是：

```text
operator string at sweep t
          │
          ▼
diagonal_update_profiled
  ├─ 從 |0...0> propagate state
  ├─ 保留 type=-1 slots
  ├─ 對 type=1/2 slots做 grouped-alias rejection resampling
  ├─ 同時累積 per-site event counts
  └─ 可選：沿 ramp 量 profile
          │
          ▼
build_vertex_lists
  ├─ counting-sort fill per-site site-op lists
  ├─ fill per-site bond-event lists
  └─ propagate state 並建立 bond_spin[p]
          │
          ▼
cluster_update
  ├─ site 依序處理
  ├─ 每個 site 的 internal segments 做 Metropolis
  ├─ accepted segment 更新 bond_spin[p]
  └─ 依相鄰 segment flip parity 切換 type 1 ↔ -1
          │
          ▼
operator string at sweep t+1
```

對應程式位置：

- diagonal sweep：`csrc/qaqmc_core.cpp:1022`
- grouped alias rejection：`csrc/qaqmc_core.cpp:1071`
- vertex-list construction：`csrc/qaqmc_core.cpp:1138`
- segment Metropolis：`csrc/qaqmc_core.cpp:1207`
- `mc_step()` phase timing：`csrc/qaqmc_core.cpp:1333`

### 2.1 關鍵診斷：diagonal loop 是「實作串行」，不是「數學上全串行」

令 `F_p` 是 slice `p` 的 off-diagonal flip mask：

```text
F_p = one_hot(op_sites[p])   if op_types[p] == -1
      0                      otherwise
```

不含 seam 時，slice `p` 執行前的 state 是：

```text
S_before[p] = F_0 XOR F_1 XOR ... XOR F_{p-1}
```

本次 diagonal update 只會把原本的 type `1/2` slot 改成另一個 type `1/2` slot；它不會新增、刪除或移動 type `-1` operators。因此 diagonal resampling 前後的 `F_p` 完全不變。

所以在固定輸入 operator string 下：

- 所有 `S_before[p]` 可以先由 associative XOR scan 計算。
- 給定 `S_before[p]` 後，不同 diagonal slots 的 conditional resampling 互相獨立。
- `state_at_M` 就是 prefix scan 在 `p=M` 的輸出。
- profile point 的 state 是 `S_before[p] XOR F_p`。

目前 source comment 已經間接記錄這個事實：diagonal update 不改變 off-diagonal operators，因此它走過的 state trajectory 與獨立的 `measure_profile()` 相同（`csrc/qaqmc_core.cpp:1013-1016`）。GPU 版本應把這個不變量提升成演算法主體。

這比「一條 chain 配一個 CUDA thread，沿 `p` 串行」更有希望，尤其適合超長 `M`。

### 2.2 真正不能任意重排的是 cluster 的 site update

當 site `i` 的 segment 被接受，`bond_spin[p]` 會更新；後續相鄰 site 的 acceptance ratio 必須看到更新後的值。因此共享 active bond 的 sites 不能任意同時更新。

- 若 `neighbor_cutoff=-1`，interaction graph 是 complete graph，site coloring 退化成 `N` 個 colors，site-level 平行度幾乎為零。
- 但同一個 site 的不同 internal segments 對應互不重疊的 `p` intervals；在該 site 開始時讀取固定的 bond-spin state，它們可以平行計算 ratio 和 acceptance，之後再平行 commit。
- 因此 GPU cluster mapping 應是「sites 按必要順序、site 內 segments 大量平行」，而不是強行讓所有 sites 同時 flip。

## 3. 現有 production 計時證據

repo 內的 runtime probes 顯示：

| `M` | `M_total` | CPU 每 step（slowest rank） | diagonal | cluster |
| ---: | ---: | ---: | ---: | ---: |
| 2,760,000 | 5,520,000 | 約 0.85–0.92 s | 61.5% | 38.5% |
| 27,600,000 | 55,200,000 | 約 8.65–9.37 s | 32.8–33.3% | 66.7–67.2% |
| 100,000,000 | 200,000,000 | 約 48–54 s | 23.7% | 76.3% |

來源：`logs/probe_otf_26189.out`、`logs/probe_otf_26173.out`、`logs/probe_otf_26278.out`、`logs/probe_otf_26174.out`。

注意：這些 probes 來自不同 job/rank 配置，不能拿來做嚴格 scaling fit；但足以判斷大 `M` 時 cluster 是主要瓶頸，而且 working set 超出 CPU cache 後成本惡化。

若只把 diagonal 加速到無限快，Amdahl 上限為：

```text
M=27.6M: 1 / 0.67 ≈ 1.49x
M=100M : 1 / 0.76 ≈ 1.31x
```

因此 diagonal-only CUDA kernel 只適合做可行性驗證，不能作為 production 終點。

## 4. GPU memory 診斷

以下以 6×6 `kagome_bond`、`N=216`、full van der Waals bonds、`G=600` 為例：

```text
n_bonds  = N(N-1)/2 = 23,220
max_alias = N + n_bonds = 23,436
```

### 4.1 所有 chains 共用的資料

目前 `GroupedAlias`：

| 陣列 | 估計大小 |
| --- | ---: |
| `entries[G, max_alias]`，16 bytes/entry | 214.56 MiB |
| `bond_W_rmax_all[G, n_bonds]`，double | 106.29 MiB |
| `bond_W_max_all[G, n_bonds]`，double | 106.29 MiB |
| 合計 | 427.15 MiB |

hot path 只讀 `entries` 和 `bond_W_rmax_all`。`bond_W_max_all` 建表後沒有被 standard engine update 使用；GPU device copy 可省略它，將常駐 hot table 降為約 320.86 MiB。

這些表對相同 Hamiltonian/schedule 的所有 chains 完全共享，不應 per-chain 複製。

每個 group 的 hot working set 約為：

```text
entries: 23,436 × 16 bytes ≈ 366 KiB
rmax  : 23,220 ×  8 bytes ≈ 181 KiB
total : 約 547 KiB/group
```

它放不進一般 shared memory，但可藉由讓相鄰 blocks 處理同一 group 的 slices，提高 L2 reuse。GPU grid 不應隨機交錯不同 delta groups。

### 4.2 每條 chain 的主要資料

令 `L=M_total=2M`：

- `op_types[L] + op_sites[L]`：`8L` bytes。
- CPU 目前另存 `delta_sched[L]`：`8L` bytes；GPU backend 不應 per-chain 複製，最好直接用 `delta_at(p)` 的算式產生，否則至少只能存一份 shared schedule。
- `bond_spin[L]`：`L` byte。
- `site_op_list`：最多 `4L` bytes。
- `site_bond_list`：最多 `16L` bytes（每個 bond slot 產生兩個 8-byte endpoint events）。
- segment flags、event-building offsets、scan/sort temporary storage：依實作增加。

若 GPU 不 materialize `delta_sched`，保守 core/workspace 範圍可抓約 `19L–30L` bytes，不含 radix-sort double buffers。這不是最終 allocation contract，但足以做硬體選型：

| `M` | operator arrays only | 約略 core/workspace |
| ---: | ---: | ---: |
| 2.76M | 42 MiB | 0.10–0.15 GiB/chain |
| 27.6M | 421 MiB | 0.98–1.54 GiB/chain |
| 100M | 1.49 GiB | 3.54–5.59 GiB/chain |

排序或 stable-bucketing temporary storage 會再增加 peak memory。這表示：

- 24 GB GPU 可以處理單一超長 chain，但不一定能同時容納很多 `M=1e8` chains。
- 80 GB GPU 才適合對超長 `M` 做多-chain batch。
- 顯存限制的是 concurrent chains/batch size，不限制總 samples；可分批跑。
- 因為單鏈可以沿 `p` prefix-scan 平行，batch size 小不代表 GPU 必然閒置。

### 4.3 已確認的 `gpunode02` 硬體

2026-07-14 透過 Slurm allocation 在 node 上實查 `nvidia-smi`：

| GPU index | 型號 | VRAM | Compute capability |
| ---: | --- | ---: | ---: |
| 0 | NVIDIA A100-PCIE-40GB | 40,960 MiB | 8.0 (`sm_80`) |
| 1 | Tesla V100-PCIE-32GB | 32,768 MiB | 7.0 (`sm_70`) |
| 2 | Tesla V100-PCIE-32GB | 32,768 MiB | 7.0 (`sm_70`) |

driver 為 `570.172.08`。查詢當下 A100 使用約 30,785 MiB，兩張 V100 為 0 MiB；這只是當下排程狀態，不是固定可用量。

因此第一版 CUDA target 應同時產生 `sm_70` 和 `sm_80` code，並把 V100、A100 當成兩個獨立 benchmark target。三張卡異質，不能只報三卡平均：

- correctness/迭代優先用當時空閒的卡，不硬編 GPU index。
- V100 32 GB 是 memory-fit 的最低 production gate。
- A100 40 GB 是主要 throughput benchmark，但仍不足以假設可對 `M=1e8` 做大型 multi-chain batch。
- multi-GPU driver 採 one MPI rank per GPU，各 rank 依自己的 free VRAM 決定 batch size；不要把一條 chain 跨 A100/V100 切分。

## 5. 建議的 CUDA backend 架構

### 5.1 Backend 邊界

不要直接把 CUDA code 混進目前唯一的 `qaqmc_cpp` module。建議新增：

```text
csrc/cuda/
  qaqmc_cuda_core.cuh
  qaqmc_cuda_core.cu
  qaqmc_cuda_scan.cu
  qaqmc_cuda_events.cu
  qaqmc_cuda_cluster.cu
  qaqmc_cuda_observables.cu
  bindings_cuda.cpp

src/engines/qaqmc_cuda.py
tests/gpu/
```

build 為 optional module：

```text
qaqmc_cpp   # 現有 CPU reference，永遠可建
qaqmc_cuda  # 找到 CUDA toolkit 且 QAQMC_ENABLE_CUDA=ON 才建
```

Python API 先保持窄而清楚：

```python
engine = BatchedQAQMCEngineCUDA(..., n_chains=B)
engine.run_profile(...)
engine.get_checkpoint(chain_id)
engine.set_checkpoint(chain_id, ...)
```

CPU/MPI driver 最終映射：

```text
one MPI rank per GPU
  └─ one CUDA engine
       ├─ chain 0
       ├─ chain 1
       └─ ... chain B-1
```

不同 GPU 之間仍以 independent chains 分工，不建議第一版把單條 operator string 切跨多張 GPU。

### 5.2 Device-resident 原則

以下資料在 equilibration/sample loop 中不離開 GPU：

- operator strings
- RNG counters/states
- event lists 與 `bond_spin`
- profile/bin accumulators
- alias/bond/geometry tables

host 只在以下時機收資料：

- progress counters
- completed bin/checkpoint
- final operator string/RNG checkpoint
- fatal invariant diagnostics

禁止每個 `mc_step()` 做 device→host→device round trip。

## 6. Diagonal GPU 演算法

### 6.1 State representation

把 `N` 個 occupation bits pack 成：

```cpp
constexpr int WORDS = (N + 63) / 64;
uint64_t state[WORDS];
```

`N=216` 時 `WORDS=4`；`N=384` 時 `WORDS=6`。XOR 是 associative operation，適合 hierarchical scan。

第一版可以針對常用 `N<=384` template specialization，超過上限 fallback CPU。這比一開始支援任意動態 `N` 更容易讓 registers/shared-memory 使用可控。

### 6.2 不存整條 per-slice state

若直接存 `S_before[p]`，`N=216` 需要 `32L` bytes；`L=55.2M` 就是約 1.65 GiB。沒有必要。

建議 two-level tiled scan：

1. `tile_parity_kernel`
   - 每個 block 處理連續 `T` 個 slices。
   - 將該 tile 中所有 type `-1` one-hot masks XOR 成 `tile_parity[tile]`。
2. `exclusive_scan_tiles`
   - 對 `tile_parity` 做 exclusive XOR scan，得到每個 tile 的 starting state。
3. `diagonal_resample_kernel`
   - 每個 block 再處理一個連續 tile。
   - block 內對每個 slice 的 one-hot mask做 exclusive scan。
   - `S_before[p] = tile_start XOR local_prefix[p]`。
   - type `-1` slot保持不變；type `1/2` slot獨立做 alias rejection。

每個 thread 對應一個 `p` 時，operator arrays 是 coalesced access。建議初始測試 `T=128/256`，再由 Nsight 決定。

### 6.3 RNG

CPU 使用單一 `mt19937_64`，GPU 不應企圖維持 CPU bitwise trajectory。建議使用 counter-based Philox：

```text
key     = global_seed, chain_id
counter = sweep_id, p, rejection_attempt, draw_lane
```

優點：

- 不需要龐大的 per-thread mutable RNG state。
- thread scheduling 改變不會改變同一 GPU build 的 replay。
- 每個 slice 的 rejection loop 有獨立 stream。
- checkpoint 只需保存 seed/chain/sweep counters 加 operator string。

驗證目標是相同 transition distribution 和 observables，不是 CPU/GPU 同 seed 同軌跡。

### 6.4 Alias-table locality

group index仍由 `floor(p*G/L)` 得到。讓 tiles 不跨 group boundary，或將 group-boundary tile拆開，使同一批 blocks集中讀一個約 547 KiB group table。

第一版維持：

- double `prob`
- double `rmax`
- 現有 `compute_bond_W_inline` 算術順序

確認統計正確後，才能分別 benchmark：

- alias probability改 float
- `loc_kind` packing
- forward/backward delta groups 去重
- constant/read-only cache placement

不能一次混入，否則數值偏差難定位。

### 6.5 Fused outputs

同一個 prefix state 可同時產生：

- `state_at_M`
- diagonal slot resampling
- final per-slot operator kind/location
- type-2 slot 的初始 `bond_spin[p]`
- selected profile point 的 packed state

CPU 目前把 counting 與 profile 融進 diagonal sweep；GPU 也應融合不增加 contention 的工作，但不要為了維持「單 kernel」而使用大量 global atomics。event counts/lists 可由專門的 parallel primitive 建立。

## 7. Vertex/event list 建立方案

cluster 需要每個 site 按 `p` 排序的事件。GPU 第一版最穩妥的作法是使用成熟的 radix-sort/scan primitives，而不是手寫 lock-free scatter。

### 7.1 MVP：統一 event stream + radix sort

對每個 slot輸出：

- type `±1`：一個 site event。
- type `2`：兩個 bond-endpoint events。

key/value 可設計為：

```text
key   = (chain_id, site, p)
value = (kind, bond_id, endpoint, op_type)
```

以 `(chain, site, p)` radix sort 後，每個 site 天然得到有序 stream。再用 run-length encode / scan 找出 site ranges。

優點：實作風險低、有成熟 CUB primitive、容易驗證。缺點：double-buffer temporary memory 大，而且每個 sweep 都需排序。

### 7.2 優化版：stable site bucketing

如果 Nsight 顯示 radix sort 成為主瓶頸，再改成：

1. 每個連續 `p` tile產生 per-site counts。
2. 對每個 site 沿 tile index 做 exclusive scan。
3. tile 內做 stable local sort/prefix。
4. scatter 到已知 site offsets。

因為 tiles 本身按 `p` 排列，只要 tile 內 stable，輸出自然按 `p` 排序，不需全域 radix sort。

這個版本記憶體較省、理論 bandwidth 較低，但程式複雜度與 bug 風險高，應排在正確的 MVP 之後。

## 8. Cluster GPU 演算法

### 8.1 Site 順序

full interaction (`neighbor_cutoff=-1`) 下，site updates 保留 `site=0..N-1` 的序列順序，維持和 CPU 相同的 valid Metropolis kernel。

有限 cutoff 時可以沿用 Renyi engine 已有的 graph-coloring 思路：同 color sites 無共享 bond，可平行處理；但它是第二階段優化。

### 8.2 Site 內 segment parallelism

對目前 site `i`：

1. 從有序 event stream 找出 single-site events，它們定義 internal segments。
2. 每個 bond endpoint event依前一個 single-site event取得 segment id。
3. 每個 event平行計算 `w_new/w_old`、zero/inf flags。
4. 對 segment id 做 segmented reduction，重現 CPU 的 ratio/shift/zero/inf acceptance語意。
5. 每個 segment用 Philox 產生一個 acceptance draw。
6. accepted segments 的 bond events平行 XOR `bond_spin[p]`。
7. 每個 single-site event依左右 segment flags平行切換 type `1 ↔ -1`。

同一 site 的 segments 對應不重疊的 `p` ranges，因此 commit 不會對同一 `bond_spin[p]` 發生競爭。

### 8.3 Floating-point reduction

CPU 在 segment內依 `p` 順序逐項乘 ratio，GPU tree reduction會改變 rounding。這在統計上可以合法，但 acceptance threshold 附近可能讓軌跡分叉。

第一版有兩個選項：

- correctness-first：每個 segment一個 thread依序乘；跨 segments 平行。通常 segment數量很大，可能已足以填滿 GPU。
- throughput-first：每個長 segment用 deterministic tree reduction，接受 CPU/GPU非 bitwise identity。

建議先做 correctness-first。profile 確認長 segments 確實成為瓶頸後，再只對超過 threshold 的 segment啟用 deterministic parallel reduction。

### 8.4 Kernel launch overhead

`N=216` 意味每 step 最多有 216 個按 site 排序的 cluster stages。將整個固定拓撲的 mc-step pipeline capture 成 CUDA Graph，避免每步數百次 host launch。

若每 site 的平均 event數是 `O(L/N)`，在 production 超長 `L` 下單一 stage仍有足夠 GPU work；小 `M` 時則依賴 batch-of-chains 提升 occupancy。

## 9. Profile observables

profile workload其實非常適合 GPU：prefix scan已提供 selected slices 的 packed state，後續 density、loop/string parity、VBS/SS、occupation SF 都是相同 state 上的獨立 reductions。

建議：

1. diagonal scan只把每個 requested profile point 的 packed state寫到小型 buffer。
2. observables用獨立 kernels處理，避免把複雜量測塞進 rejection kernel造成 register pressure。
3. 在 GPU 上累積成現有 `batch_size/checkpoint` 定義的 bin means。
4. occ-SF outer products也留在 device累積，只傳 final super-bins。

`profile_step=100`、`L=55.2M` 時，selected packed states約為：

```text
(L / 100) × 32 bytes ≈ 16.8 MiB   # N=216
```

比儲存所有 slices 的 state 小兩個數量級，且足以支援完整 profile。

## 10. 開發階段與 go/no-go gates

### Phase 0：CPU baseline補強

先把現有 coarse timers細分為：

- diagonal state propagation
- alias proposals / rejection attempts
- profile measurement
- event counting/fill
- bond-spin build
- segment ratio evaluation
- segment commit/type reassignment

並記錄：

- proposals per accepted diagonal slot
- site/bond operator fractions
- segment length histogram
- per-site event count distribution
- peak RSS

這一步能避免把 GPU 工程押在錯誤瓶頸上。

### Phase 1：diagonal scan prototype

範圍：single replica、無 seam、無 profile、固定 `N<=384`、device-resident operator string。

通過條件：

- fixed input string 的 `S_before[p]` 與 CPU 完全一致。
- sampled diagonal operator histogram符合 exact per-slice distribution。
- alias envelope invariant全通過。
- `M` ladder上 diagonal kernel相對單 CPU core至少 5×；否則先停止 full port並檢討 memory/table locality。

### Phase 2：standard cluster update

先 radix-sort MVP，再視 profile 改 stable bucket。

通過條件：

- operator type/site範圍、open-boundary closure、`state_at_M` invariants 全通過。
- small-N ED observables與 CPU 均在統計誤差內。
- fixed topology 的 segment acceptance test逐例對照 CPU reference。
- full `mc_step` 相對單 CPU core至少 3×。

### Phase 3：profile + checkpoint

通過條件：

- density、loop/string、VBS/SS、SF 的 per-configuration result可對固定 operator string逐項對照 CPU。
- device binning與 host reference一致到設定 tolerance。
- sampling loop沒有 per-step PCIe operator-string transfer。

### Phase 4：batch chains + multi-GPU MPI

通過條件應比較整體資源，而不是只和一個 CPU core比：

```text
GPU chains×steps/s  vs  目前 24/32-rank CPU node chains×steps/s
```

同時報告：

- samples/s
- effective samples/s（含 integrated autocorrelation）
- peak VRAM
- energy/sample（若 cluster可提供）
- 初始化與 checkpoint overhead

### Phase 5：擴充功能

依序建議：

1. off-diagonal seam（把 fixed seam mask視為 prefix scan中的額外 XOR event）。
2. string-work trajectories（天然適合 batch chains）。
3. Renyi engine（兩 replicas、channel remapping、topology update，風險最高）。

## 11. 驗證策略

不能只驗最終 observable；近期 seam bugs 曾經互相抵消，GPU backend必須有分層驗證。

### Deterministic structural tests

- CPU固定 operator string → GPU prefix states逐 slice/抽樣 slice比對。
- `state_at_M` 比對。
- vertex events 的 `(site,p,kind,bond,endpoint)` multiset與順序比對。
- `bond_spin[p]` 比對。
- 每 site segment boundaries比對。
- raw segment ratios與 zero/inf classification比對。
- boundary segments永遠 frozen。
- cluster後每 site off-diagonal parity符合 open-boundary closure。

### Statistical tests

- fixed slice/state 的 diagonal proposal histogram。
- small chain/N 的 CPU/GPU midpoint observables。
- 現有 ED tests 的 GPU parameterization。
- forward/backward profile symmetry/asymmetry regression。
- acceptance rates、operator fractions、autocorrelation time比較。

### Reproducibility contract

- 相同 GPU build、seed、chain id可 replay。
- CPU 與 GPU不承諾 bitwise相同 trajectory。
- 不同 GPU architecture只承諾統計等價；任何 fast-math或 float table壓縮需獨立 feature flag與驗證。

## 12. 不建議的路線

### 一條 chain = 一個 CUDA core

CUDA core不是可由程式長期綁定的 MPI-rank等價物。這會留下大部分 GPU閒置，且無法利用 prefix scan揭示的 slice parallelism。

### 只用 CuPy/Numba 包住現有 Python

真正熱路徑在 C++、含 irregular rejection與cluster event structures；array wrapper無法自動產生正確且有效率的 kernel。

### 只 port alias-table initialization

初始化只有數秒，production sampling是數小時；即使初始化變成零，總 runtime幾乎不變。

### diagonal 在 GPU、cluster 在 CPU

每 step至少搬 `8L` bytes operator arrays，`M=27.6M` 約 421 MiB；來回 PCIe 加同步會抵銷 diagonal收益。這只能當 microbenchmark，不能是 production architecture。

### 第一版直接移植 Renyi/string 全功能

standard engine已足以驗證 prefix scan、event sorting、segment update與 device binning。先把這四件事做對，再擴 topology/seam，較容易定位物理偏差。

## 13. 主要風險與對策

| 風險 | 影響 | 對策 |
| --- | --- | --- |
| alias random access / rejection divergence | diagonal throughput低 | group-local scheduling、量 proposals/accept、保留 batch chain維度 |
| event radix sort吃掉 cluster收益 | full-step加速不足 | 先量再改 stable tile bucketing |
| full-range bonds使 site無法平行 | cluster sequential stages | site內 segment平行、CUDA Graph；不破壞 detailed balance |
| 超長 `M` temporary memory爆 VRAM | batch size過小/無法執行 | streaming event construction、stable bucket、memory planner、明確拒絕超額配置 |
| GPU reduction改變 acceptance rounding | 軌跡分叉/難除錯 | correctness-first per-segment sequential product；統計契約而非bitwise CPU契約 |
| CPU/GPU ping-pong | 加速歸零 | operator string與accumulators全程device-resident |
| seam/topology invariant重演舊 bug | 物理結果錯 | seam/Renyi延後，逐層白箱 invariant tests |

## 14. 在正式寫 CUDA 前需要決定的提案問題

已確認 target node 是 1×A100 PCIe 40 GB（`sm_80`）+ 2×V100 PCIe 32 GB（`sm_70`）。剩餘需要決定：

1. 第一個勝負 benchmark要用哪個 production point？建議 `N=216, M=2.76e6` 作迭代，`M=2.76e7` 作 go/no-go，`M=1e8` 作 memory/scale壓測。
2. 要比較一張 GPU對一個 CPU core，還是對整個24/32-rank node？最終應兩者都報，但採購/排程決策應看整 node throughput。
3. 第一版是否只支援 `neighbor_cutoff=-1`？這是目前最難的 cluster case；若它能過，有限 cutoff只會提供更多 coloring機會。
4. 可否接受 CPU/GPU非 bitwise相同，但通過 exact small-system與統計檢驗？若要求同 RNG trajectory，GPU slice parallelism會被不必要地限制。

## 15. 建議的第一個實作 milestone

先不要一次建立完整 GPU engine。第一個 milestone應是一個可丟棄的 diagonal feasibility branch/target：

```text
input : CPU產生的固定 operator string + shared tables
GPU   : tiled prefix XOR + diagonal resample，重複多 sweeps但先不接cluster
check : prefix states、distribution、throughput、L2 hit、rejection divergence、VRAM
```

若它在 `M=2.76e7` 上無法達到至少 5× single-core diagonal throughput，先分析 alias table locality與scan成本，不進 cluster port。

若通過，再做 device-resident standard `mc_step`。真正的 project success gate不是 kernel漂亮，而是：

```text
在相同物理 transition kernel、相同統計精度下，
一張目標 GPU 的 effective samples/hour 明顯勝過現有 CPU node，
且能在 production M 下穩定 checkpoint/restart。
```
