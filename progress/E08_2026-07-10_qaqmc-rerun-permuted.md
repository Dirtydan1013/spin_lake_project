# E08 — QAQMC sweep重跑（permutation ON）：修復驗證（2026-07-10）

## 目的
E04的mitigation在production規模驗證：M3回不回來？phase-lock消不消失？

## 設定
- 同E01（kagome_bond 6×6 periodic、M=2.76M、64 ranks），差異：`--permute-site-labels`預設ON
  ＋forward-snapping（δ點全在上坡）；job 26251
- 資料 `data/qaqmc_profile_M=2760000_6x6_20260710_031323`（meta記錄`permute_site_labels=True`、
  configs含`site_perm`）
- 圖 `figures/qaqmc_profile_M=2760000_6x6_20260710_031323/m_domains_full_d5.5.png`

## 結果（occ點δ≈5.30，粗grid）
- per-rank取向：**M1:M2:M3 = 24:15:16**（weak 9）——三取向全佔據，對均分18.3±3.5僅+1.6σ內
  （對照E01：30:32:1）
- per-bin（1536 bins）：575:376:370、corr(M1,M2)=+0.006
- **phase-lock指標：跨rank⟨n_i⟩ site-std = 0.047**（E01鎖定值0.239、SSE平衡0.049）——
  **與平衡態的對稱恢復程度一致**

## 結論
修復完全生效：sweep ensemble恢復C3與translation對稱的統計恢復。從此sweep資料的
unconnected S(q)、誤差棒、深δ熵解讀恢復可信。

## 後續
production sweep（更大M）＋KP TEE可以在乾淨的ensemble上進行。
