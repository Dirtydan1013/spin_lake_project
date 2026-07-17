# E05 — SSE β=20 δ=5.5 完整觀測量＋stripe熔化實驗（2026-07-10）

## 目的
部署新`.so`後重跑E02拿完整觀測量（occ-SF/Z_l/C_m/VBS/snapshots）；並直接檢驗
「δ=5.5是否存在SS/stripe相（我們之前只是被困的過冷態？）」。

## 設定
- 同E02參數＋`--occ-sf-grid-n 12 --n-snapshots 1`（新.so）；job 26244
- 資料 `data/sse_6x6_kagome_bond_beta20_delta5.5`；圖 `figures/sse_6x6_kagome_bond_beta20_delta5.5/`
- 熔化實驗：從E08的M1-stripe snapshot（λM1=2.61、n=56）當SSE初態（空op string），
  δ=5.5 β=20跑3000 sweeps追λ(M)；對照組disorder-init

## 結果
- **平衡態unconn/conn(M)=1.0**（⟨s_M⟩≈0）：ensemble平均掉取向與registration → 對比E01
  的sweep（unconn≫conn）反推出**sweep是phase-locked**（跨rank⟨n_i⟩ std：sweep 0.239 vs 平衡0.049）
- per-bin domain（burn 0.5，256 bins）：M1:M2:M3=72:47:90（λ>1）、corr(M1,M2)=−0.146
- conn λ/cell：K=0.630 ≥ M1=0.611 > M3=0.542 > M2=0.438 —— **K與M競爭、皆短程**
- **熔化實驗**：stripe初態λM1第一個200-sweep block就掉到0.53，之後與disorder-init軌跡
  無法區分 → **不是過冷，是真的沒有SS相**（此β/參數下）
- 密度0.2548≈1/4（論文SS需n=1/3——熱力學量直接矛盾）

## 結論
δ=5.5、β=20平衡態＝K/M競爭的短程關聯量子無序態；無SS。sweep的「stripe秩序」大半是
動力學凍結。unconn模式在domain平均下失去診斷力（→ 改用per-bin分析，`plot_m_domains.py`）。

## 後續
→ E06 δ-scan、E07 β-scan。
