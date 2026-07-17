# E06 — SSE平衡態δ-scan β=20＋8×8 L-scaling（2026-07-10）

## 目的
掃描δ=3.5–5.5找平衡相邊界；8×8對照做Bragg的L-scaling判定。

## 設定
- 6×6 β=20：δ ∈ {3.5, 3.75, 4.0, 4.25, 4.5, 5.5}；8×8 β=40：δ=5.5（N=384=論文尺寸）
- 全部64 ranks×7 chunks×250 samples、occ 12×12、burn 0.3
- 資料 `data/sse_6x6_kagome_bond_beta20_delta{...}`、`data/sse_8x8_kagome_bond_beta40_delta5.5`
- 圖 `figures/sse_delta_scan_beta20/summary.png`

## 結果
- **序參量全平**：Ψ_SS≈0.37、Ψ_VBS≈0.16–0.18（低於1/√25=0.2基線）；密度0.226→0.255平滑
- **L-scaling（δ=5.5）**：conn λ/cell幾乎不動（K: 0.630→0.608；M avg: 0.53→0.52；
  per-bin λM 1.07→1.15，LRO應×1.78）→ **K、M皆短程，無LRO**
- conn λK隨δ成長0.375→0.630、λM平（0.47→0.53）→ K關聯主導
- loops/strings：Z_l(2)峰0.447@δ≈4.25–4.5、|A_v|峰0.82、**C_m全尺寸全δ≈0**、BFFM≈0
  （與Wang–Pollet Fig 4a一致）
- **排他性shuffle-null檢驗**：δ=3.5–4.5的「單Q bin比例」=null（±0.8σ）→ 純漲落本底、
  無對稱破缺；唯δ=5.5超出+2.2σ（萌芽傾向，β=80降為+0.6σ）

## 結論
整個δ∈[3.5,5.5]無對稱破缺相（=Wang–Pollet的RCSL、≠Vu的NQS相圖VBS/SS）。
**δ≈4.0–4.5=SL候選窗**（Z_l峰＋|A_v|峰＋C_m≡0＋無密度序）——KP TEE的目標參數。

## 後續
→ E07 β-scan排除溫度；→ 文獻對照`paper/COMPARISON.md`；→ KP TEE @δ≈4.25。
