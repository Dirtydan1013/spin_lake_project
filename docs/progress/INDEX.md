# 實驗記錄索引

每個實驗一個檔案：`experiments/EXX_<日期>_<slug>.md`。格式：目的 → 設定（引擎/參數/job/資料路徑）→ 結果（數字）→ 結論 → 後續。
文獻對照見 `paper/z2/COMPARISON.md`；專案整體狀態見 `CLAUDE.md`。

| # | 日期 | 實驗 | 一句話結論 | 狀態 |
|---|---|---|---|---|
| [E01](experiments/E01_2026-07-07_qaqmc-profile-sweep.md) | 07-07 | QAQMC δ-sweep profile 6×6（第一次occ-SF） | 假Bragg：64條chain鎖同一張stripe圖案，M3缺席 | 已被E04解釋 |
| [E02](experiments/E02_2026-07-09_sse-beta20-first.md) | 07-09 | SSE β=20 δ=5.5（舊.so，final-config分析） | 平衡態C3公平（21:20:23）→ sweep的偏向是動力學 | 完成 |
| [E03](experiments/E03_2026-07-09_c3-fairness-checks.md) | 07-09 | C3/動量/測量鏈三重公平性檢查 | 幾何與測量全部無罪，元兇只剩動力學 | 完成 |
| [E04](experiments/E04_2026-07-09_scan-order-probe.md) | 07-09 | 掃描順序判別實驗＋mitigation | **掃描順序決定domain選擇**；`--permute-site-labels`（PR #2） | 完成 |
| [E05](experiments/E05_2026-07-10_sse-beta20-full-obs.md) | 07-10 | SSE β=20 δ=5.5 完整觀測量＋熔化實驗 | 無SS相：stripe初態200 sweeps熔掉；sweep是phase-locked | 完成 |
| [E06](experiments/E06_2026-07-10_sse-delta-scan.md) | 07-10 | SSE δ-scan β=20（3.5–5.5）＋8×8 β=40 | 全窗口無序；SL候選窗δ≈4.0–4.5；排他性=shuffle-null | 完成 |
| [E07](experiments/E07_2026-07-10_sse-beta-scan.md) | 07-10 | β-scan（20/40/80 @5.5；20/40 @4.25） | 全平：β≤80下溫度不是限制 | 完成 |
| [E08](experiments/E08_2026-07-10_qaqmc-rerun-permuted.md) | 07-10 | QAQMC sweep重跑（permutation ON） | M3回歸24:15:16、phase-lock消失（std 0.047=平衡值） | 完成 |
| [E09](experiments/E09_2026-07-10_sse-deep-beta.md) | 07-10 | 深βSSE（160/320 @4.25、160 @5.5） | 預測命中：β×16全平無VBS/SS；4.25液體指標反增強→KP TEE可上 | 完成 |
| [E12](experiments/E12_2026-07-11_chi-f-estimator.md) | 07-11 | χ_F估計子：SSE WLT＋QAQMC速度校準（ED雙釘死） | 兩管線±1σ；slot-cut偏差→Binomial時間切割；非對稱點O(v²)無線性項 | 完成 |
| [E13](experiments/E13_2026-07-12_thermal-entropy-ladder.md) | 07-12 | 熱熵β-ladder（W–P Eq.7、全vdW不截斷） | 三層熵階梯（ln2→ln4/3→ln2/6）；S/N(β=6,δ=4.5,8×8)=0.140(11)命中W–P | 完成（生產待排） |
| [E14](experiments/E14_2026-07-12_sse-offdiagonal-string.md) | 07-12 | SSE非對角弦引擎（熱態⟨X_C⟩，Jarzynski） | ED 4/4＋driver端對端；週期邊界宇稱修復bug；回查QAQMC挖出3 bug（seam cache×2＋閉合宇稱重設）全修——舊two-site測試通過是互償假象 | 完成 |
| [E15](experiments/E15_2026-07-27_entropy-ladder-beta80.md) | 07-27 | 熵ladder低溫延伸β=20→81.75（δ=4.5 6×6） | S/N平台撐到T≈0.012不動、終點0.110(12)≈ln2/6；overlap −4.5σ=冷啟動偏差（丟棄該rung）；β=81.75 config已存可warm-start | 完成 |
| [E16](experiments/E16_2026-07-28_entropy-ladder-beta179.md) | 07-28 | 熵ladder再延伸β=82→178.7（warm-start首戰） | 平台撐到T≈0.0056、終點0.112(19)≈ln2/6（累計比E13低9倍溫度）；warm-start接縫無偏差；decimation check≲統計誤差 | 完成 |
