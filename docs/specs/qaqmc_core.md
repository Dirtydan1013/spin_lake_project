# 規格：csrc/qaqmc_core.hpp / csrc/qaqmc_core.cpp

## 角色
`QAQMCEngine` 是 C++ 端的 single-replica QAQMC engine，用來模擬 detuning ramp 的 Rydberg Hamiltonian。它使用固定長度 operator string 表示 imaginary-time ramp，沒有 SSE 的 identity operator，也沒有可變 expansion order。

這個 engine 是 `QAQMCRenyiEngine` 的基礎：Renyi core 共用 Rydberg bond builder、alias table builder、bond weight convention，以及 QAQMC 的 open-boundary update 思想。

Python 端主要透過 `src/engines/qaqmc.py` 和 MPI drivers 使用它。

## 物件 / 函式

| 名稱 | 種類 | 可見性 | 用途 |
| --- | --- | --- | --- |
| `RydbergVij` | struct | public | 儲存 active bonds、bond strengths、bond endpoints、effective coordination number。 |
| `build_rydberg_vij()` | function | public | 由 coordinates 和 cutoff 建立 Rydberg interaction bonds。 |
| `AliasTable` | struct | public | 儲存每個 time slice 的 QAQMC diagonal proposal alias tables 和 bond weights。 |
| `build_qaqmc_alias_tables()` | function | public | 建立每個 time slice 的 alias proposal tables。 |
| `QAQMCEngine` | class | public | Single-replica detuning-ramp QAQMC 引擎。 |
| `MidpointObservables` | struct | public nested | `p=M` 對稱點的 density、loop/string observables。 |
| `ProfileObservables` | struct | public nested | 沿 imaginary-time profile 的 density、loop/string observables。 |
| `mc_step()` | method | public | 執行一次完整的 diagonal update 和 cluster update。 |
| `diagonal_update()` | method | private | 從 open boundary 掃過 `operator string`，重新抽樣 diagonal site/bond operators，並更新 `state_at_M_`。 |
| `cluster_update()` | method | private | 用 open boundary 和 frozen boundary segments 做 per-site segment Metropolis update。 |
| `build_vertex_lists()` | method | private | 建立 cluster update 使用的 per-site single-site-op / bond-op lists。 |
| `set_observable_sites()` | method | public | 設定 loop/string observable 的 site copies。 |
| `set_bulk_sites()` | method | public | 設定 density 使用的 bulk sites。 |
| `measure_at_midpoint()` | method | public | 量測 `p=M` 的 observables。 |
| `measure_profile(profile_step)` | method | public | 沿 imaginary time 每隔 `profile_step` 量測 observables。 |
| `get_rng_state()` / `set_rng_state()` | method | public | RNG checkpoint。 |
| `set_op_string(types, sites, len)` | method | public | 從外部恢復 `operator string`。 |
| `compute_bond_W_inline()` | static method | public | 計算單一 bond 在四個 spin states 下的 diagonal weights。 |

## 物理 / 演算法契約

- QAQMC 模擬 detuning schedule `delta_min -> delta_max -> delta_min` 的 ramp，而不是 finite-temperature trace。
- Imaginary time 使用 open boundary condition，`tau=0` 和 `tau=2M` 的 boundary state 都是 `|0...0>`。
- `operator string` 長度固定為 `M_total = 2 * M`。
- QAQMC 不使用 identity operator。每個 slot 都是 offdiagonal operator 或 diagonal site/bond operator。
- Diagonal update 不是 insertion/removal，而是對現有 diagonal slot 重新抽樣 operator kind/location。
- Cluster update 不允許 segment wrap around boundary。碰到 open boundary 的 first/last segments 是 frozen。
- `state_at_M_` 是 ramp 對稱點 `p=M` 的 spin state，用於 on-the-fly observables。
- Bond diagonal weight convention 和 SSE/Renyi 共用：`delta_i = delta / coord_number[i]`，`delta_j = delta / coord_number[j]`。

## 輸入

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `N` | `int` | Site 數。 |
| `Omega` | `double` | Rabi frequency。 |
| `delta_min`, `delta_max` | `double` | Detuning ramp 的範圍。 |
| `Rb` | `double` | Rydberg interaction scale。 |
| `M` | `int` | Half schedule length；引擎使用 `M_total = 2M`。 |
| `epsilon` | `double` | Bond diagonal weight shift 的 safety margin。 |
| `seed` | `uint64_t` | RNG seed。 |
| `pos` | `double*`, shape `(N, pos_dim)` | Site coordinates。 |
| `neighbor_cutoff` | `int` | Interaction bond cutoff；`-1` 表示 all-to-all。 |
| `precompute` | `bool` | 是否預先建立 alias/bond weight tables。 |
| `chunk_slices` | `int` | Chunked precompute 的 slice 數；`0` 表示 full precompute。 |
| `delta_groups` | `int` | Grouped alias table 數量；`0` 表示不用 grouping。 |
| `loop_sets`, `string_sets` | `vector<vector<int>>` | Observable site copies。 |
| `bulk_sites` | `vector<int>` | Density 的 bulk subset；空集合表示使用全部 sites。 |

## 輸出

| 輸出 | Type / Shape | 意義 |
| --- | --- | --- |
| `op_types_` | `int32[M_total]` | Operator type string。 |
| `op_sites_` | `int32[M_total]` | Operator site/bond index string。 |
| `delta_sched_` | `double[M_total]` | 每個 time slice 的 detuning schedule。 |
| `bond_sites_flat` | `int[2 * n_bonds]` | Row-major active bond endpoint list。 |
| `state_at_M_` | `int32[N]` | Midpoint spin state。 |
| `MidpointObservables` | struct | Midpoint density 和 loop/string copies。 |
| `ProfileObservables` | struct | Profile density 和 loop/string copies。 |
| `time_diag`, `time_clus` | `double` seconds | Update timing。 |

## 資料契約

- `op_types[p]` 的編碼：
  - `-1`: offdiagonal single-site spin flip。
  - `1`: diagonal site operator。
  - `2`: diagonal bond operator。
- `op_sites[p]` 的意義由 `op_types[p]` 決定：
  - `-1` 和 `1`: site index。
  - `2`: `vij_.bond_sites_flat` 裡的 bond index。
- 合法的 QAQMC `operator string` 不包含 `op_types[p] == 0` 的 identity。
- `AliasTable` arrays 的第一個維度是 `M_total`：
  - `bond_W_all`: `(M_total, n_bonds_pad, 4)` flattened。
  - `bond_W_max_all`: `(M_total, n_bonds_pad)` flattened。
  - `n_alias_all`: `(M_total)`。
  - `alias_prob_all`, `alias_idx_all`, `op_map_kind_all`, `op_map_loc_all`: `(M_total, max_alias)` flattened。
- `op_map_kind == 0` 代表 site proposal；`op_map_kind == 1` 代表 bond proposal。
- `GroupedAlias` 以 group index 取代 slice dimension，並用 `slice_to_group[p]` 紀錄每個 slice 屬於哪個 group。
- Loop/string observables 使用每個 registered copy 上的 signed product `1 - 2 n_i`。

## 狀態 / 不變量

- `M_total_ == 2 * M_`。
- `op_types_` 和 `op_sites_` 長度都必須是 `M_total_`。
- 合法的 `operator string` 只能包含 `-1`、`1`、`2`。
- `p=0` 的 propagated boundary state 從全零 state 開始。
- `state_at_M_` 必須等於從 open boundary 套用所有 `p < M_` operators 後的 state。
- 若 `delta_groups_ > 0`，grouped alias envelopes 必須 upper-bound 每個 slice 的 true bond weights。
- 若 `precompute_ && chunk_slices_ <= 0 && delta_groups_ == 0`，cluster update 可以直接使用 `alias_.bond_W_all`。
- 若不是 full precompute 模式，cluster update 必須由 `delta_sched_[p]` 計算 slice-specific bond weights。
- Vertex lists 是 scratch data；operator string 改變後必須重建。

## 行為

1. 建構子建立 Rydberg interaction bonds、`delta_sched_`、初始固定長度 `operator string`、alias proposal tables、observable scratch arrays，以及 vertex-list scratch arrays。
2. `diagonal_update()` 從 open boundary 開始，沿所有 `M_total_` slices propagate 一個暫時 spin state。
3. 當 `p == M_` 時，`diagonal_update()` 把當下 propagated state 複製到 `state_at_M_`。
4. 若 `op_types[p] == -1`，diagonal update 只負責 propagate 該 spin flip。
5. 若 `op_types[p] == 1` 或 `2`，diagonal update 會反覆抽樣新的 diagonal operator 直到接受：
   - Site proposals 直接接受。
   - Bond proposals 以 `W_actual(p, state) / W_envelope(p)` 接受。
6. `diagonal_update()` 支援四種 proposal 模式：
   - `delta_groups_ > 0` 時使用 grouped alias table；
   - full precomputed per-slice alias table；
   - chunked precomputed alias table；
   - on-the-fly cumulative weights。
7. `build_vertex_lists()` 用 `O(M_total + N)` 建立 per-site single-site-op 和 bond-op lists。
8. `cluster_update()` 從 `|0...0>` propagate 以計算 bond spin states，接著做 per-site segment Metropolis updates。
9. Cluster segments 使用 open boundaries：只 proposal 兩個 consecutive single-site operators 之間的 internal segments。Boundary segments 保持 frozen。
10. Segment flips 接受後，每個 single-site operator 是否在 diagonal `1` 和 offdiagonal `-1` 間切換，由相鄰 segments 的 flip parity 決定。
11. `mc_step()` 依序執行 diagonal update、cluster update，並記錄 timers。

Grouped alias proposal 對 slice `p` 上 bond `b` 的有效接受 proposal distribution 正比於：

```text
q_group(b) * accept(b, state, p)
  = [W_envelope_group(b) / Z_group] * [W_actual_p(b, state) / W_envelope_group(b)]
  = W_actual_p(b, state) / Z_group
```

Rejection sampling 會重複直到某個 operator 被接受；因此最後的 diagonal operator distribution 會正比於 true per-slice weights。

## 函數規格

### `build_rydberg_vij()`

**種類：** function  
**可見性：** public

**用途**  
由 site coordinates 建立 Rydberg interaction bonds，並計算每個 site 的 effective coordination number。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `N` | `int` | Site 數。 |
| `Omega`, `Rb` | `double` | Interaction scale 相關參數。 |
| `pos` | `double*`, `(N, pos_dim)` | Site coordinates。 |
| `neighbor_cutoff` | `int` | 使用的 distance shell cutoff；`-1` 表示 all-to-all。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| return value | `RydbergVij` | Active bonds、`vij_list`、`bond_sites_flat`、`coord_number`。 |

**演算法流程**

1. 由所有 site pairs 建立候選 bonds。
2. 若有 cutoff，依 distance shell 選出 active bonds。
3. 計算每個 bond 的 `V_ij`。
4. 填入 bond endpoints 和每個 site 的 `coord_number`。

**邊界情況**

- `neighbor_cutoff == -1` 時使用所有 pairs。
- 沒有 active bonds 時，回傳空 bond lists 但仍保留長度為 `N` 的 `coord_number`。

**不變量**

- `bond_sites_flat` 長度為 `2 * n_bonds`。

**測試**

- All-to-all 小系統檢查 `n_bonds = N(N-1)/2`。

### `build_qaqmc_alias_tables()`

**種類：** function  
**可見性：** public

**用途**  
建立 QAQMC diagonal update 使用的 per-slice alias proposal tables 和 bond weight tables。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `M_total`, `N`, `n_bonds` | `int` | 時間長度與系統大小。 |
| `Omega`, `delta_sched`, `epsilon` | scalars/array | Site/bond proposal weights。 |
| `bond_vij`, `bond_si`, `bond_sj`, `coord_number` | arrays | Bond weights 和 endpoints。 |
| `p_start`, `p_end` | `int` | 可選 chunk 範圍。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| return value | `AliasTable` | Per-slice proposal table、bond weights、proposal map。 |

**演算法流程**

1. 對每個 slice 計算 site proposal weights。
2. 對每個 active bond 計算四個 spin states 的 `W` 和 `W_max`。
3. 以 site weights 和 bond `W_max` 建立 alias proposal distribution。
4. 寫入 `op_map_kind_all` 和 `op_map_loc_all`，讓抽樣結果能映射回 site/bond。

**邊界情況**

- Chunked call 時只填入 local slice range。
- `n_bonds == 0` 時仍需讓 site proposal table 合法。

**不變量**

- `n_alias_all[p]` 必須和該 slice 的 proposal entries 數量一致。
- `bond_W_max_all[p,b] >= bond_W_all[p,b,w]`。

**測試**

- 檢查 alias table shape、proposal kind/location mapping、`W_max` envelope。

### `QAQMCEngine::QAQMCEngine(...)`

**種類：** constructor  
**可見性：** public

**用途**  
初始化 single-replica detuning-ramp QAQMC chain。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| physical parameters | scalars | `N`, `Omega`, `delta_min`, `delta_max`, `Rb`, `M`, `epsilon`。 |
| `pos`, `neighbor_cutoff` | array/scalar | 建立 Rydberg interaction graph。 |
| `precompute`, `chunk_slices`, `delta_groups` | config | 控制 diagonal proposal table 策略。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `delta_sched_` | `double[M_total]` | Ramp schedule。 |
| `op_types_`, `op_sites_` | `int32[M_total]` | 初始 operator string。 |
| `alias_` 或 `grp_alias_` | structs | Diagonal proposal tables。 |
| scratch arrays | arrays | Cluster update 和 observables 使用。 |

**演算法流程**

1. 設定 `M_total_ = 2 * M_`。
2. 建立 `vij_` 和 `delta_sched_`。
3. 初始化 operator string 與 RNG。
4. 依 `precompute` / `delta_groups` 建立 proposal tables。
5. 配置 midpoint/profile observable 和 cluster scratch arrays。

**邊界情況**

- `delta_groups > 0` 時建立 grouped alias envelopes。
- `precompute == false` 時不建立 full alias table。

**不變量**

- 初始化後 operator string 不含 identity。
- `delta_sched_` 長度等於 `M_total_`。

**測試**

- Constructor smoke test：檢查 shapes、mode flags、初始 operator encoding。

### `QAQMCEngine::mc_step()`

**種類：** method  
**可見性：** public

**用途**  
執行一次完整 QAQMC Monte Carlo step。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| internal state | `QAQMCEngine` fields | 目前 operator string、RNG、tables、scratch arrays。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `op_types_`, `op_sites_` | arrays | 更新後的 operator string。 |
| `state_at_M_` | `int32[N]` | 最新 midpoint state。 |
| timers, `mc_steps_` | diagnostics | Profiling counters。 |

**演算法流程**

1. 呼叫 `diagonal_update()`。
2. 呼叫 `cluster_update()`。
3. 更新 `time_diag_`、`time_clus_` 和 `mc_steps_`。

**邊界情況**

- `cluster_update()` 可以在 `M_total_ == 0` 時直接返回。

**不變量**

- `op_types_` 不應產生 identity。

**測試**

- 固定 seed 下檢查 reproducibility 和 operator encoding。

### `QAQMCEngine::diagonal_update()`

**種類：** method  
**可見性：** private

**用途**  
從 open boundary propagate spin path，並重新抽樣所有 diagonal slots。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `op_types_`, `op_sites_` | `int32[M_total]` | 目前 operator string。 |
| `delta_sched_` | `double[M_total]` | Slice-dependent detuning。 |
| `alias_`, `grp_alias_` | structs | Proposal tables。 |
| `vij_` | `RydbergVij` | Bond endpoints 和 weights。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `op_types_`, `op_sites_` | `int32[M_total]` | 重新抽樣後的 diagonal operators。 |
| `state_at_M_` | `int32[N]` | `p=M` 的 propagated state。 |

**演算法流程**

1. 從全零 spin state 開始。
2. 掃過 `p = 0 ... M_total_-1`。
3. `p == M_` 時記錄 `state_at_M_`。
4. `-1` operator 只 propagate spin flip。
5. Diagonal slot 從目前啟用的 proposal path 抽樣 site/bond。
6. Site proposal 直接接受；bond proposal 以 true `W_actual / W_envelope` 接受。
7. 若 proposal 被拒絕，重抽直到接受。

**邊界情況**

- `delta_groups_ > 0` 時 proposal 來自 group envelope，但 acceptance 用 true slice weight。
- Chunked precompute path 只在 chunk 生命週期內持有 alias data。
- On-the-fly path 每個 slice 重建 cumulative weights。

**不變量**

- 不產生 identity。
- `state_at_M_` 對應 open-boundary propagation 到 `p=M` 的結果。

**測試**

- 比較 `state_at_M_` 和 direct reconstruction。
- 比較 grouped alias / full precompute / on-the-fly 的統計一致性。

### `QAQMCEngine::cluster_update()`

**種類：** method  
**可見性：** private

**用途**  
在 open boundary condition 下做 per-site segment Metropolis update。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `op_types_`, `op_sites_` | `int32[M_total]` | 目前 operator string。 |
| `delta_sched_`, `alias_`, `vij_` | arrays/structs | 計算 segment flip weight ratios。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `op_types_` | `int32[M_total]` | Single-site operators 在 `1` 和 `-1` 間切換。 |
| `bond_spin_` | scratch | Segment Metropolis 使用的 bond spin cache。 |

**演算法流程**

1. 呼叫 `build_vertex_lists()`。
2. 從 `|0...0>` propagate，記錄每個 bond op 的 spin state。
3. 對每個 site 建立由 single-site ops 切出的 segments。
4. 只 proposal internal segments；boundary-touching segments frozen。
5. 對 accepted segment 更新 `bond_spin_`。
6. 用相鄰 segment flip parity 決定 single-site op type 是否切換。

**邊界情況**

- `n_sops == 0` 時不做整條 worldline flip。
- Boundary segments 永遠 frozen。

**不變量**

- 不允許 wrap through endpoints。
- `op_types_` 只在 `1` 和 `-1` 間切換。

**測試**

- 人工 operator string 檢查 boundary frozen 行為。

### `QAQMCEngine::build_vertex_lists()`

**種類：** method  
**可見性：** private

**用途**  
建立 cluster update 需要的 per-site event lists。

**Inputs / Outputs / Algorithm**  
同 `SSEEngine::build_vertex_lists()`，但掃描長度是 `M_total_`，且 QAQMC operator string 不含 identity。

**不變量**

- Lists 內 positions 依 `p` 遞增。

**測試**

- 人工 operator string 的 count/head/list consistency。

### `QAQMCEngine::measure_at_midpoint()`

**種類：** method  
**可見性：** public

**用途**  
用 `state_at_M_` 量測 midpoint density、loop observables 和 string observables。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `state_at_M_` | `int32[N]` | Midpoint spin state。 |
| `loop_site_sets_`, `string_site_sets_`, `bulk_sites_` | vectors | Observable definitions。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| return value | `MidpointObservables` | Density 和每個 copy 的 signed product。 |

**演算法流程**

1. 在 `bulk_sites_` 或全部 sites 上計算 density。
2. 對每個 loop/string copy 計算 `prod_i (1 - 2 n_i)`。

**邊界情況**

- `bulk_sites_` 空集合表示使用全部 sites。

**不變量**

- 不修改 Markov chain state。

**測試**

- 對人工 `state_at_M_` 比對手算 observable。

### `QAQMCEngine::measure_profile(profile_step)`

**種類：** method  
**可見性：** public

**用途**  
沿 imaginary time 每隔 `profile_step` 重建 spin state 並量測 observables。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `profile_step` | `int` | Profile sampling interval。 |
| `op_types_`, `op_sites_` | arrays | 用來重建各 profile point 的 state。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| return value | `ProfileObservables` | Density 和 loop/string copies 的 time profile。 |

**演算法流程**

1. 從 open boundary state 開始 propagate。
2. 每隔 `profile_step` 記錄當下 state 的 observables。
3. 掃完整個 `M_total_`。

**邊界情況**

- `profile_step` 太小會增加輸出量和測量成本。

**不變量**

- 不修改 operator string。

**測試**

- 對短 operator string 比對 direct reconstruction。

### `QAQMCEngine::set_op_string(types, sites, len)`

**種類：** method  
**可見性：** public

**用途**  
從外部資料恢復 operator string，用於 checkpoint 或測試。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `types`, `sites` | `int32*` | 要載入的 operator arrays。 |
| `len` | `int` | 必須等於 `M_total_`。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `op_types_`, `op_sites_` | `int32[M_total]` | 被外部資料覆蓋。 |

**演算法流程**

1. 檢查 `len == M_total_`。
2. 複製 `types` 和 `sites` 到內部 arrays。

**邊界情況**

- 長度不符時 raise。

**不變量**

- 呼叫者應確保 operator encoding 合法。

**測試**

- 載入人工 string 後量測 midpoint，和 direct reconstruction 比對。

## 邊界情況

- `set_op_string()` 會在輸入長度不等於 `M_total_` 時 raise。
- 若 `bulk_sites` 是空集合，midpoint density 使用全部 sites。
- 若某個 site 沒有 single-site operators，QAQMC cluster update 不會對該 site 做整條 worldline flip，因為那會碰到 frozen boundaries。
- 若 bond proposal 的 envelope 是零或不合法，該 proposal 會被拒絕。
- Chunked precompute 逐 chunk 建立並釋放 alias data，以降低 peak memory。
- On-the-fly 模式不需要 alias table memory，但每個 slice 都要重建 cumulative weights。

## 效能備註

- Full precompute 對 diagonal 和 cluster weight lookup 最快，但記憶體成本是 `O(M_total * n_bonds)`。
- Chunked precompute 降低 peak memory，但每次 diagonal update 都要重新建立 chunk alias tables。
- `delta_groups` 透過跨 slices 共用 alias proposal tables 來降低記憶體。正確性依賴使用 true per-slice weights 的 rejection correction。
- On-the-fly 模式 table memory 最低，但每次 update 的計算成本最高。
- Vertex lists 讓 cluster update 不需要對每個 site 掃過完整 operator string，而是接近 `O(M_total + number of touched bond events)`。
- `state_at_M_` 在 diagonal update 中順手捕捉，避免常用 observables 需要額外 midpoint reconstruction pass。

## 驗收標準

- 合法 run 的 `operator string` 不應出現 identity operators。
- `state_at_M_` 應該和從 `op_types_` / `op_sites_` 獨立重建的結果一致。
- Full precompute、chunked precompute、grouped alias、on-the-fly 模式在相同物理參數下，結果應在統計誤差內一致。
- 改變 `delta_groups` 可以改變 memory/runtime，但不應系統性改變 measured observables。
- QAQMC cluster update 必須遵守 open boundary condition，不能讓 segment wrap through endpoints。
- 小系統 observables 應該和 ED 或 saved regression data 一致。

## 測試

- Unit test `compute_bond_W_inline()`，和手算四個 spin states 的 weights 比對。
- Unit test `build_rydberg_vij()`，覆蓋 all-to-all 和 finite cutoff cases。
- Unit test alias table shapes 和 `op_map_kind/op_map_loc` consistency。
- Regression test 比較 `delta_groups=0` 和 grouped alias 的 density/profile observables。
- Test `set_op_string()` 後的 midpoint measurement，和 direct state reconstruction 比對。
- 小系統 QAQMC vs ED midpoint observable check。

## 相關檔案

- `csrc/qaqmc_core.hpp`
- `csrc/qaqmc_core.cpp`
- `csrc/qaqmc_renyi_core.hpp`
- `csrc/qaqmc_renyi_core.cpp`
- `csrc/sse_core.hpp`
- `csrc/bindings.cpp`
- `src/engines/qaqmc.py`
- `src/engines/qaqmc_updates.py`
- `src/mpi/qaqmc_mpi.py`
- `src/analysis/postprocess.py`

## 開放問題

- 是否要把 `precompute`、`chunk_slices`、`delta_groups` 的選擇策略整理成自動 policy，而不是由 driver 手動決定。
- 是否要在 debug mode 檢查 grouped alias envelope 的 `W_actual <= W_envelope` invariant。
- 是否要把 midpoint/profile observable 的 lattice-specific convention 移到 Python 層，讓 C++ core 只負責 reconstruct spin states。
