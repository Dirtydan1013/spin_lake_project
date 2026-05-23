# 規格：csrc/sse_core.hpp / csrc/sse_core.cpp

## 角色
`SSEEngine` 是 C++ 端的 finite-temperature Stochastic Series Expansion 引擎，用來模擬固定 Hamiltonian 的 Rydberg model。

SSE 使用固定容量、可變有效長度的 `operator string`。`M_` 是 buffer capacity，`n_ops_` 是目前非 identity operators 的數量。每個 `mc_step()` 會做 diagonal insertion/removal、cluster update，最後在需要時擴張 `M_`。

這個檔案主要服務 `src/engines/sse.py` 和 `src/mpi/sse_mpi.py`。

## 物件 / 函式

| 名稱 | 種類 | 可見性 | 用途 |
| --- | --- | --- | --- |
| `SSEEngine` | class | public | Finite-temperature SSE Rydberg QMC 引擎。 |
| `mc_step()` | method | public | 執行一次完整的 diagonal update、cluster update、capacity adjustment。 |
| `diagonal_update()` | method | private | 掃過 `operator string`，執行 identity/diagonal insertion-removal，並沿途 propagate spin state。 |
| `cluster_update()` | method | private | 用 periodic trace boundary 做 per-site cyclic segment Metropolis update。 |
| `build_vertex_lists()` | method | private | 建立 cluster update 使用的 per-site single-site-op / bond-op lists。 |
| `adjust_M_if_needed()` | method | private | 當 `n_ops_` 接近 capacity 時擴大 operator string buffer。 |
| `measure_energy()` | method | public | 估計目前 `operator string` 的 energy。 |
| `measure_density()` | method | public | 回傳目前 `state_` 的平均 occupation。 |
| `measure_mz()` | method | public | 回傳 staggered magnetization。 |
| `get_state()` | accessor | public | 回傳目前 trace boundary state。 |
| `get_op_types()` / `get_op_sites()` | accessor | public | 回傳 `operator string` buffer。 |
| `get_n_ops()` | accessor | public | 回傳目前非 identity operators 的數量。 |
| `get_rng_state()` / `set_rng_state()` | method | public | RNG checkpoint。 |
| `get_time_diag()` / `get_time_clus()` | accessor | public | Profiling timers。 |

## 物理 / 演算法契約

- SSE 模擬 finite-temperature partition function `Tr exp(-beta H)`，imaginary time boundary 是 periodic trace。
- `operator string` 有 identity slots，`diagonal_update()` 透過 `identity <-> diagonal operator` 的 insertion/removal 改變 `n_ops_`。
- Offdiagonal single-site operator 代表 `sigma_x`，掃描 `operator string` 時會翻轉對應 spin。
- `cluster_update()` 的 worldline 是 periodic 的，所以 per-site segment 可以 wrap around `tau=beta` 到 `tau=0`。
- Bond diagonal weight 使用和 QAQMC 相同的 asymmetric detuning split convention：`delta_i = delta / coord_number[i]`，`delta_j = delta / coord_number[j]`。
- 若 `neighbor_cutoff == -1`，使用 all-to-all bonds，且 `coord_number[i] == N - 1`。

## 輸入

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `N` | `int` | Site 數。 |
| `Omega` | `double` | Rabi frequency。 |
| `delta` | `double` | Fixed detuning。 |
| `Rb` | `double` | Rydberg interaction scale。 |
| `beta` | `double` | Inverse temperature。 |
| `epsilon` | `double` | Bond diagonal weight shift 的 safety margin。 |
| `seed` | `uint64_t` | RNG seed。 |
| `pos` | `double*`, shape `(N, pos_dim)` | Site coordinates。 |
| `neighbor_cutoff` | `int` | Interaction bond cutoff；`-1` 表示 all-to-all。 |

## 輸出

| 輸出 | Type / Shape | 意義 |
| --- | --- | --- |
| `state_` | `int32[N]` | 目前 trace boundary spin state。 |
| `op_types_` | `int32[M_]` | Operator type buffer。 |
| `op_sites_` | `int32[M_]` | Operator site/bond index buffer。 |
| `n_ops_` | `int` | 非 identity operators 的數量。 |
| `M_` | `int` | Operator string capacity。 |
| `bond_sites_flat` | `int[2 * n_bonds]` | Row-major bond endpoint list。 |
| `time_diag`, `time_clus` | `double` seconds | Update timing。 |

## 資料契約

- `op_types[p]` 的編碼：
  - `0`: identity slot。
  - `-1`: offdiagonal single-site spin flip。
  - `1`: diagonal single-site operator。
  - `2`: diagonal bond operator。
- `op_sites[p]` 的意義：
  - `-1` 和 `1`: site index。
  - `2`: bond index。
  - `0`: 通常是 `-1`。
- `bond_W_[b * 4 + w_idx]` 儲存 bond diagonal weights，其中 `w_idx = ni * 2 + nj`。
- `bond_W_max_[b]` 儲存該 bond 在四個 spin states 中的最大 weight，用於 bond insertion proposal 的 rejection correction。
- Alias table 抽樣 proposal category：
  - `op_map_kind == 0`: site diagonal proposal。
  - `op_map_kind == 1`: bond diagonal proposal。
  - `op_map_loc`: site index 或 bond index。

## 狀態 / 不變量

- `0 <= n_ops_ <= M_`。
- `op_types_` 和 `op_sites_` 長度都必須是 `M_`。
- `diagonal_update()` 後，`op_types[p] != 0` 的 entries 數量應該等於 `n_ops_`。
- `state_` 是 periodic trace boundary 上的 spin state，用來 propagate operator string。
- `norm_N_` 是 insertion/removal probabilities 使用的 diagonal proposal normalization。
- `energy_shift_` 是 diagonal bond weights 產生的 constant shift，供 `measure_energy()` 使用。
- Vertex lists 是 scratch data；operator string 改變後必須重建。

## 行為

1. 建構子建立 Rydberg bonds、diagonal bond weights、固定參數的一張 alias table、初始 state、operator buffers，以及 vertex-list scratch arrays。
2. `diagonal_update()` 掃過所有 `M_` slots：
   - `-1`: propagate spin，翻轉 `state_[site]`。
   - `1` 或 `2`: 以 `(M - n_ops + 1) / (beta * norm_N)` 的機率嘗試移除 diagonal operator。
   - `0`: 以 `beta * norm_N / (M - n_ops)` 的機率嘗試插入 diagonal operator。
3. 插入時，alias table 先 proposal site 或 bond。Site diagonal proposal 直接接受；bond diagonal proposal 以目前 propagated spin state 的 `W_actual / W_max` 接受。
4. `build_vertex_lists()` 用 counting sort 建立 per-site single-site-op 和 bond-op lists，成本是 `O(M + N)`。
5. `cluster_update()` 從 `state_` propagate 以計算 `bond_spin_[p]`，接著做 per-site segment Metropolis updates。
6. 若某個 site 沒有 single-site operators，SSE 可以 proposal 翻轉整條 periodic worldline。
7. 若某個 site 有 single-site operators，cluster segments 是 cyclic 的。從最後一個 single-site op wrap 到第一個 single-site op 的 segment 會穿過 `tau=0`，若接受就更新 `state_[site]`。
8. Segment decisions 完成後，若某個 single-site operator 左右兩側 segment 的 flip parity 不同，就在 diagonal/offdiagonal 之間切換。
9. `mc_step()` 依序執行 diagonal update、cluster update、`adjust_M_if_needed()`，並記錄 timers。

## 函數規格

### `SSEEngine::SSEEngine(...)`

**種類：** constructor  
**可見性：** public

**用途**  
建立 finite-temperature SSE 模擬需要的固定 Hamiltonian、operator buffer、alias table、RNG 和 scratch arrays。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `N`, `Omega`, `delta`, `Rb`, `beta`, `epsilon` | scalars | Rydberg Hamiltonian 和 SSE sampling 參數。 |
| `seed` | `uint64_t` | 初始化 `rng_`。 |
| `pos` | `double*`, `(N, pos_dim)` | Site coordinates。 |
| `neighbor_cutoff` | `int` | 決定 active interaction bonds。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `vij_` | `RydbergVij` | Active bonds 和 interaction strengths。 |
| `bond_W_`, `bond_W_max_` | arrays | Fixed-detuning bond diagonal weights。 |
| `alias_prob_`, `alias_idx_`, `op_map_*` | arrays | Diagonal insertion proposal alias table。 |
| `state_`, `op_types_`, `op_sites_` | arrays | 初始 spin state 和 operator string buffer。 |

**演算法流程**

1. 由 `pos`、`Rb`、`neighbor_cutoff` 建立 `vij_`。
2. 計算 single-site proposal weight 和每個 bond 的 diagonal weights。
3. 建立一張固定參數 alias table。
4. 初始化 `state_`、identity-filled operator buffer 和 cluster scratch arrays。

**邊界情況**

- `neighbor_cutoff == -1` 時建立 all-to-all bonds。
- 若沒有 active bonds，bond arrays 仍要保持合法空或 padded layout。

**不變量**

- `op_types_` 和 `op_sites_` 長度等於 `M_`。
- 初始 `n_ops_` 和 non-identity slots 數量一致。

**測試**

- 小系統 constructor smoke test：檢查 array shapes、`coord_number`、alias table normalization。

### `SSEEngine::mc_step()`

**種類：** method  
**可見性：** public

**用途**  
執行一次完整 SSE Monte Carlo step，是外部 wrapper 應呼叫的主要更新入口。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| internal state | `SSEEngine` fields | 目前 `state_`、operator string、RNG、alias table。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `op_types_`, `op_sites_`, `state_`, `n_ops_` | internal arrays/scalars | 更新後的 Markov chain state。 |
| `M_` | `int` | 可能因 `adjust_M_if_needed()` 增大。 |
| `time_diag_`, `time_clus_`, `mc_steps_` | diagnostics | 更新 profiling counters。 |

**演算法流程**

1. 呼叫 `diagonal_update()`。
2. 呼叫 `cluster_update()`。
3. 呼叫 `adjust_M_if_needed()`。
4. 累積 timers 並遞增 `mc_steps_`。

**邊界情況**

- 若 `cluster_update()` 因 `M_ == 0` 返回，仍會完成 timer/counter 更新。

**不變量**

- `0 <= n_ops_ <= M_`。
- `op_types_` 和 `op_sites_` 長度等於目前 `M_`。

**測試**

- 固定 seed 下連續呼叫 `mc_step()`，檢查 `n_ops_` accounting 和 reproducibility。

### `SSEEngine::diagonal_update()`

**種類：** method  
**可見性：** private

**用途**  
在 fixed-capacity operator string 中執行 diagonal insertion/removal，並沿途 propagate `state_`。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `op_types_`, `op_sites_` | `int32[M_]` | 目前 operator string。 |
| `state_` | `int32[N]` | 掃描起點的 trace boundary spin state。 |
| `alias_*`, `op_map_*` | arrays | Diagonal insertion proposal table。 |
| `bond_W_`, `bond_W_max_` | arrays | Bond proposal rejection correction。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `op_types_`, `op_sites_` | `int32[M_]` | 插入或移除 diagonal operators 後的 string。 |
| `state_` | `int32[N]` | 掃完整個 string 後的 propagated trace boundary state。 |
| `n_ops_` | `int` | 更新 non-identity operator count。 |

**演算法流程**

1. 逐 slot 掃描 `p = 0 ... M_-1`。
2. 遇到 `-1` 時翻轉 `state_[site]`。
3. 遇到 diagonal operator 時，以 removal probability 嘗試改成 identity。
4. 遇到 identity 時，以 insertion probability 嘗試插入 diagonal operator。
5. 插入時先由 alias table proposal site/bond；site 直接接受，bond 用 `W_actual / W_max` 接受。

**邊界情況**

- `n_ops_ == M_` 時不再插入。
- `W_max <= 0` 的 bond proposal 會被拒絕。

**不變量**

- 更新後 `n_ops_` 應等於 `op_types[p] != 0` 的數量。

**測試**

- 對小 `M_` 的人工 operator string 做 insertion/removal accounting test。

### `SSEEngine::cluster_update()`

**種類：** method  
**可見性：** private

**用途**  
在 periodic trace boundary 下做 per-site cyclic segment Metropolis update。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `op_types_`, `op_sites_` | `int32[M_]` | 目前 operator string。 |
| `state_` | `int32[N]` | `tau=0` trace boundary state。 |
| `bond_W_` | array | Segment flip 的 local weight ratio。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `op_types_` | `int32[M_]` | Single-site operators 在 `1` 和 `-1` 間切換。 |
| `state_` | `int32[N]` | 若 accepted segment wrap 過 `tau=0`，對應 site 會翻轉。 |
| `bond_spin_` | `int32[M_]` | Scratch；記錄 bond op 當下 spin state。 |

**演算法流程**

1. 呼叫 `build_vertex_lists()`。
2. 從 `state_` propagate，為每個 bond op 記錄 `bond_spin_[p]`。
3. 對每個 site 找出 single-site ops 切出的 cyclic segments。
4. 計算每個 segment flip 的 `log(W_new/W_old)` 並用 Metropolis 接受。
5. 根據相鄰 segment flip parity 切換 single-site operator type。

**邊界情況**

- `n_sops == 0` 時，可以 proposal 翻轉整條 periodic worldline。
- Wrapping segment 會跨過 `tau=0`，若接受要同步更新 `state_[site]`。

**不變量**

- Periodic boundary condition 必須保留。
- `op_types_` 只在 single-site positions 的 `1` / `-1` 間切換。

**測試**

- 人工構造 small operator string，檢查 wrapping segment 對 `state_` 的更新。

### `SSEEngine::build_vertex_lists()`

**種類：** method  
**可見性：** private

**用途**  
為 cluster update 建立 per-site single-site-op list 和 bond-op list。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `op_types_`, `op_sites_` | `int32[M_]` | 目前 operator string。 |
| `vij_.bond_sites_flat` | `int[2*n_bonds]` | Bond endpoints。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `site_op_count_`, `site_op_head_`, `site_op_list_` | arrays | 每個 site 的 single-site operator positions。 |
| `site_bond_count_`, `site_bond_head_`, `site_bond_list_` | arrays | 每個 site 相關的 bond operator positions。 |

**演算法流程**

1. 第一 pass 計算每個 site 的 single-site/bond event 數量。
2. Prefix sum 建立 head offsets。
3. 第二 pass 依 `p` 遞增填入 lists。

**邊界情況**

- 某些 site 可以沒有任何 single-site ops 或 bond ops。

**不變量**

- Lists 內 positions 依 `p` 遞增排序。

**測試**

- 對人工 operator string 檢查 count/head/list 是否一致。

### `SSEEngine::adjust_M_if_needed()`

**種類：** method  
**可見性：** private

**用途**  
當 active expansion order 接近 operator buffer capacity 時擴大 `M_`。

**輸入**

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `n_ops_`, `M_` | `int` | 目前 expansion order 和 buffer capacity。 |

**輸出 / 修改**

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `M_` | `int` | 可能增大到約 `1.33 * n_ops_`。 |
| `op_types_`, `op_sites_`, `bond_spin_` | arrays | resize 後保留既有資料。 |

**演算法流程**

1. 計算 `new_M = int(n_ops_ * 1.33)`。
2. 若 `new_M > M_`，resize operator buffers 和 scratch arrays。

**邊界情況**

- 若 `new_M <= M_`，不做任何事。

**不變量**

- resize 後舊有 operator entries 不應遺失。

**測試**

- 建立接近滿載的 buffer，確認 resize 後資料保持不變。

## 邊界情況

- 若 `n_ops_ == M_`，identity insertion 會被跳過，因為 buffer 已滿。
- `adjust_M_if_needed()` 在 active expansion order 接近 capacity 時，把 `M_` 增長到約 `1.33 * n_ops_`。
- 若 `W_max <= 0`，bond insertion 會被拒絕。
- 若 `M_ == 0`，cluster update 直接返回。
- 對沒有 single-site operators 的 site，整條 periodic worldline flip 是合法的 cluster proposal。
- RNG checkpoint 只恢復 RNG state；完整的 operator string/state checkpoint 由 wrapper 或上層程式處理。

## 效能備註

- Diagonal proposal 使用 alias table，所以 insertion proposal 是 `O(1)`。
- Cluster update 使用 vertex lists，避免對每個 site 掃過完整 operator string。
- `site_op_list_` 和 `site_bond_list_` 會按 operator position 排序，因為填入時外層迴圈是遞增的 `p`。
- SSE 只有一個 fixed detuning，所以只需要一張 alias table，不需要 QAQMC 的 per-slice 或 grouped alias tables。
- `M_` auto-growth 用較高記憶體換取較少 buffer saturation 和 failed insertions。

## 驗收標準

- 多次 `mc_step()` 後，`n_ops_` 仍應和 non-identity slots 數量一致。
- 增加初始 `M_` 或讓 `adjust_M_if_needed()` 自動擴張，不應在統計誤差外改變 observables。
- Energy、density、magnetization 應該和小系統 ED 或 Python-reference results 在誤差內一致。
- Cluster update 必須在 periodic imaginary-time boundary condition 下維持 detailed balance。
- 固定 seed 和相同 binary 的 serial runs 應該可重現。

## 測試

- Unit test operator encoding 和 diagonal update 後的 `n_ops_` accounting。
- 小系統 regression，比較 finite beta 下的 energy/density 和 ED。
- Test `adjust_M_if_needed()` 擴張 buffers 時不遺失 operator data。
- Test RNG state save/restore 是否重現後續 samples。
- 比較 `neighbor_cutoff=-1` 的 all-to-all 行為是否符合 `coord_number=N-1`。

## 相關檔案

- `csrc/sse_core.hpp`
- `csrc/sse_core.cpp`
- `csrc/qaqmc_core.hpp`
- `csrc/bindings.cpp`
- `src/engines/sse.py`
- `src/engines/sse_updates.py`
- `src/mpi/sse_mpi.py`

## 開放問題

- 是否要提供完整 operator-string checkpoint API，讓 SSE 和 QAQMC 的 checkpoint 介面一致。
- 是否要把 `measure_mz()` 的 staggered convention 移到 Python/spec 層明確定義，避免不同 lattice 下誤用。
- 是否需要對 `M_` auto-growth 的倍率做可設定參數，方便大型系統調整 memory/runtime tradeoff。
