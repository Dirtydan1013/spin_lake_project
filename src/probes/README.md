# src/probes — 量測儀器（非 library）

這裡的模組是**操作性量測工具**（runtime sizing、RSS/throughput A/B、
GPU/CPU 比對），以 `python -m src.probes.<name>` 執行，launcher 在
`scripts/{probe,bench,test}/`。

契約刻意鬆：

- 輸出是 JSON / 純文字到 stdout，**沒有穩定 schema**，不保證版本相容。
- 不進測試矩陣（正確性由 `tests/` 蓋 engine 本身，不蓋這些儀器）。
- 可以隨實驗需要改；引用它們數據的結論寫進 `docs/design/` 或
  `docs/progress/` 時要連參數一起記。

`scan_order_bias.py` 是例外：它是 E04（scan-order domain bias，PR #2）
的判別實驗腳本，屬於科學證據，保留原樣不要「順手改進」。

Production runner 不在這裡：MPI drivers 在 `src/mpi/`，
非 MPI 的 CUDA runner 在 `src/runners/qaqmc_cuda.py`（有測試覆蓋）。
