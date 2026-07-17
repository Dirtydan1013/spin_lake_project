# 文件索引

專案文件集中在 `docs/`，按**生命週期**分三類：

| 子目錄 | 性質 | 更新規則 |
| --- | --- | --- |
| [`design/`](design/) | 設計紀錄與提案 — 記錄「當時為什麼這樣做」 | 寫完凍結，只追加結果，不回改 |
| [`specs/`](specs/) | 引擎現況規格 — 描述「現在是什麼」 | code 改了就要跟著改；過期比沒有更糟 |
| [`progress/`](progress/) | 物理實驗日誌（E01, E02, …） | append-only，永不回改 |

不在 `docs/` 裡的：根目錄 `README.md`（build/執行操作手冊）、`CLAUDE.md`
（專案狀態摘要）、`paper/`（參考文獻，唯讀輸入）、`csrc/*/README.md`
（元件文件，跟著 code 走）。

## design/

| 文件 | 內容 | 什麼時候讀 |
| --- | --- | --- |
| [cpu_memory.md](design/cpu_memory.md) | CPU 記憶體優化：均衡版 40% RSS、shared-model batch、event layouts、全部 A/B gate 數據（jobs 26784–26787） | 調 production 記憶體/rank 數、選 `bond_event_storage` 時 |
| [gpu_acceleration_proposal.md](design/gpu_acceleration_proposal.md) | CUDA backend 的原始可行性提案與設計 | 想了解 GPU 化的取捨與演算法映射時 |
| [gpu_version_progress.md](design/gpu_version_progress.md) | CUDA backend 實作進度、驗證 gate 記錄（含 post-merge 108-test V100 gate） | 查某個 GPU 功能驗證到什麼程度時 |
| [gpu_batch_chains.md](design/gpu_batch_chains.md) | 單 process 多 chain batch 設計與 V100 B=1/2/4/8 benchmark | 開 GPU batch production、選 B 值時 |

## specs/

引擎規格（`TEMPLATE.md` 是格式）。四份主 spec 於 2026-07-17 依 merge 後
的引擎（compact memory layout、shared model、cut 泛化、seam 元件、χ_F、
CUDA bridge）全面重寫；`reviews/` 保留歷史紀錄不回改。

| 文件 | 對象 |
| --- | --- |
| [qaqmc_core.md](specs/qaqmc_core.md) | `QAQMCEngine`（single-replica δ-sweep） |
| [qaqmc_renyi_core.md](specs/qaqmc_renyi_core.md) | `QAQMCRenyiEngine`（two-replica S₂） |
| [qaqmc_renyi_work_core.md](specs/qaqmc_renyi_work_core.md) | `QAQMCRenyiWorkEngine`（Jarzynski ΔS₂） |
| [sse_core.md](specs/sse_core.md) | `SSEEngine`（thermal reference） |
| [reviews/](specs/reviews/) | 歷史 code review 紀錄 |

## progress/

實驗日誌總表見 [progress/INDEX.md](progress/INDEX.md)（每個實驗一檔，
格式照 [progress/TEMPLATE.md](progress/TEMPLATE.md)）。
