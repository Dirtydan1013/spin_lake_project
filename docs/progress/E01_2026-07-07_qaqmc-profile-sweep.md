# E01 — QAQMC δ-sweep profile，第一次occ-SF量測（2026-07-07）

## 目的
非平衡δ-sweep沿路徑量diagonal觀測量（density/Z_l/C_m/A_v/VBS/SS）＋首次啟用
sublattice-resolved occupation SF矩陣與snapshot，看大δ端長出什麼序。

## 設定
- 引擎：`qaqmc_mpi --mode profile`，kagome_bond 6×6 **periodic**，a=4.0、Rb=2.4（=論文Rb=2.4a_nn）、全vdW
- M=2,760,000（M_total=5.52M）、δ: −2→6→−2、profile_step=10000（552點）
- 64 ranks×(4000 equil＋1563 samples)、chunk=200、occ 12×12 q-grid、11個δ點snapshot
- job 26195；資料 `data/M=2760000_6x6_20260707_172433`（**已刪除**，分析結論保留於此）
- ⚠️ 當時**沒有**site permutation（該功能是E04後才有）；δ=2.5/4.5的occ/snap點snap到了下坡段（後已修：forward優先）

## 結果
- unconn(M1)=2.70、unconn(M2)=2.49 ≫ conn≈0.5–0.66；M3全程平（0.26）→ 「M點假Bragg、M3缺席」
- per-rank取向 **M1:M2:M3 = 30:32:1**（λ>1，64 ranks）
- **跨rank每site平均密度std=0.239**（min/max 0.001/0.935）→ 64條獨立seed chain凍進同一張實空間圖案（後定名phase-lock）
- mode權重（unconn λ_max本徵向量）：δ≳3後M1鎖(α₀,α₁)、M2鎖(α₀,α₅)（C3配對）
- conn通道對比度僅1.4–2.0；K點conn在δ≥4與M同量級
- snapshot：δ≳4出現規則圖案，n→~54/216≈1/4

## 結論
sweep大δ端「看似」M-stripe對稱破缺＋ensemble相位相干——**之後被E02–E05證明是非平衡動力學假象**
（掃描順序選向＋phase-lock），非平衡物理本身真實，但不可解讀為平衡相。

## 後續
→ E02（平衡對照）、E03（公平性檢查）、E04（元兇判別）。
分析工具沉淀：`plots/plot_diagonal/`全套、`plot_m_domains.py`。
