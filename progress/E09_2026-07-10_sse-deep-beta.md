# E09 — 深βSSE：直接測試VBS的ordering尺度（2026-07-10，進行中）

## 目的
E07到β=80全平；Vu(NQS)相圖說δ≈3.4–5.2是VBS。Wang–Pollet給的磁性（plaquette）尺度
~3Ω⁶/32δ⁵（δ=4.25→7×10⁻⁵Ω→β_c~10⁴）預言QMC**永遠看不到**VBS。
用β=160/320直接檢驗：Ψ_VBS若在β=160–320抬升→NQS相圖的T_c沒那麼小；仍平→
支持RCSL圖像＋變分偏差解讀（見`paper/COMPARISON.md`矛盾1）。

## 設定
- 6×6 kagome_bond periodic、δ=4.25：β=160（N_EQUIL=8000，job 26263）、
  β=320（N_EQUIL=16000，job 26264）；δ=5.5：β=160（job 26265，排隊）
- 實測速率：β=160約0.5 sweep/s（equil 4.7h）、β=320約0.25 sweep/s（equil ~18h）
- 資料（預期）`data/sse_6x6_kagome_bond_beta{160,320}_delta4.25`、`..._beta160_delta5.5`

## 預測（事前登記）
Ψ_VBS、conn λ(M3)、排他性全部維持平坦（=β≤80的值）；熵仍在CSL平台上。
若违反預測→重大發現（NQS相圖的VBS在可及溫度存在）。

## 結果
（待補——β=160@4.25約07-10 15:00完成、β=320約07-11凌晨）

## 結論
（待補）

## 分析指令
```bash
python plots/plot_sse/plot_observables.py --run_dir data/sse_6x6_kagome_bond_beta160_delta4.25
python plots/plot_diagonal/plot_m_domains.py --run_dir data/sse_6x6_kagome_bond_beta160_delta4.25
# β對照表：照E07的report()腳本加新β列
```
