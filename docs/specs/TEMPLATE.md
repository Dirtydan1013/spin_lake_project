# 規格：[path/to/file_or_feature]

## 角色
這個檔案/功能在整個專案中負責什麼？它位在什麼工作流裡？

## 物件 / 函式
列出這個檔案裡重要的 class / struct / dataclass / function / method / CLI entry point。Public API 要寫，private 但承載核心演算法的 method 也要寫。

| 名稱 | 種類 | 可見性 | 用途 |
| --- | --- | --- | --- |
| `[name]` | `[class/function/method/struct/dataclass/CLI]` | `[public/private/module/internal]` | `[一句話說明]` |

## 物理 / 演算法契約
這裡寫「不能因為重構而改變」的物理或演算法定義。

- `[例如：log_g 是 umbrella bias，不是 entropy。]`
- `[例如：A_mask 決定 Renyi topology 的 replica/channel 連接方式。]`
- `[例如：reference ensemble 的 log_z 只是一個 gauge normalization。]`

## 模組輸入
需要哪些輸入？包含型別、shape、單位、index convention。

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `[name]` | `[dtype/shape]` | `[意義與慣例]` |

## 模組輸出
會輸出什麼？包含 return value、檔案、HDF5 dataset、stdout diagnostics。

| 輸出 | Type / Shape | 意義 |
| --- | --- | --- |
| `[name]` | `[dtype/shape]` | `[意義與慣例]` |

## 資料契約
固定資料格式寫在這裡，特別是 array layout、operator encoding、HDF5 schema。

- `[例如：op_types: -1=offdiag, 1=site diagonal, 2=bond diagonal。]`
- `[例如：block_visit_counts 的 shape 是 (n_blocks, n_ensembles)。]`
- `[例如：region mask 是 uint8 array，長度等於 lattice site 數。]`

## 狀態 / 不變量
列出執行過程中必須一直成立的條件。

- `[例如：len(log_g) == number of ensembles。]`
- `[例如：ensemble ladder 的 neighbor masks 只能差一個 site。]`
- `[例如：所有 log_z 都會扣掉 reference ensemble 的 log_z。]`

## 整體流程
整個檔案/模組的主要流程。不要逐行翻譯程式碼，而是寫出外部可依賴的行為。

1. `[step 1]`
2. `[step 2]`
3. `[step 3]`

## 函數規格

### `[function_or_method_name]`

**種類：** `[function/method/static method/class/struct]`  
**可見性：** `[public/private/module/internal]`

**用途**  
這個函數/方法要完成什麼事？它在整體演算法中負責哪一步？

**輸入**  

| 輸入 | Type / Shape | 意義 |
| --- | --- | --- |
| `[arg or internal state]` | `[dtype/shape]` | `[意義與慣例]` |

**輸出 / 修改**  

| 輸出 / 修改 | Type / Shape | 意義 |
| --- | --- | --- |
| `[return value or mutated state]` | `[dtype/shape]` | `[意義與慣例]` |

**演算法流程**

1. `[step 1]`
2. `[step 2]`
3. `[step 3]`

**邊界情況**

- `[特殊情況與處理方式]`

**不變量**

- `[這個函數結束後必須成立的條件]`

**測試**

- `[建議測試]`

## 邊界情況
模組層級的特殊情況。遇到時應該 raise、回傳 NaN、跳過，還是做 fallback？

- `[例如：某個 ensemble visit count 為 0。]`
- `[例如：block size 大於樣本數。]`
- `[例如：transition matrix 不連通。]`

## 效能備註
哪些設計是為了速度或記憶體？哪些參數會改變 runtime / memory，但不應改變結果？

- `[例如：delta_groups 共用 alias table，但用 rejection correction 保持 exact。]`
- `[例如：bit-packed event list 降低 cluster update memory bandwidth。]`

## 驗收標準
怎樣算這個檔案/功能是正確的？

- `[可重現某個小系統結果。]`
- `[輸出 schema 和舊版相容。]`
- `[改變 block size 不應改變 central value，只改變 error estimate。]`

## 測試
需要哪些測試？包含 unit test、integration test、small-system exact check、regression check。

- `[test name or idea]`
- `[test name or idea]`

## 相關檔案
這個檔案和哪些檔案一起構成完整功能？

- `[src/... or csrc/...]`
- `[specs/...]`

## 開放問題
目前還不確定、需要之後實驗或討論的事情。

- `[question]`
