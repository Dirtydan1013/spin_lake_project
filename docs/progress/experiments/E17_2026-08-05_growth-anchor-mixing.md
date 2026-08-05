# E17 — Growth anchor 的 window 漂移與混合診斷（2026-08-05）

## 目的
Growth residence-ladder anchor（O_C(δ=6)，seam-drag 曲線的絕對定標）在三個
取樣窗長度下互不一致且單調上升——判定這是 (a) 永不收斂的 glassy 混合、
(b) estimator bug、還是 (c) 有限 transient 被 burn 不足攤進平均。

## 設定
- 生產幾何：kagome_bond 6×6 **periodic**、M=227600、hexagon C={84..89}、
  Rb=2.4、δ∈[−2,6]、64 ranks（shared λ = 27119 rank-0 tune 值）
- Window 序列：27119/27120（1k samples/stage）、27125（8k）、27121（32k，production）
- 診斷：**27133** `src/mpi/growth_mixing_diag_mpi.py` — stage 1/3 拉長至
  T=48k、逐 sample 錄 bit occupancy、32 ranks 從 ON 起始/32 從 OFF、
  **不丟 burn**（transient 即量測對象）；其餘 stage 短通過（500）帶 worldline
- 修正版生產：**27135**（GROWTH_BURN=16000、GROWTH_SAMPLES=32000、10h wall）
- 資料：`data/growth_mixing_diag_27133.h5`；圖：`plots/growth_mixing_diag.png`；
  分析：`plots/plot_off_diagonal/plot_growth_mixing_diag.py`
- ⚠️ log 的 `flips=` 是 rank-0 單鏈數字非 pooled（本實驗曾因此誤判 60× 崩潰）

## 結果
Window 序列（總 anchor）：

| samples/stage | log O_C(δ=6) | job |
|---|---|---|
| 1,000 | −6.68 ± 0.30 | 27119/27120 |
| 8,000 | −5.50 ± 0.29 | 27125（1/T 模型預測 −4.68 → z≈2.9 否決） |
| 32,000 | −4.46 ± 0.15 | 27121 |

27133 兩臂診斷：
- transient **~10–15k samples** 後 p(t) 平台；平台區（後 32k）殘餘漂移
  z=−0.10（stage 1）/ −0.75（stage 3）＝統計零
- 兩臂合流到**同一**平台（末端 gap −0.036±0.030 等）→ 平穩分佈唯一、
  estimator 完整性確認；慢變數 = 共享 worldline σˣ dressing，非 bit sector
- 平台 stage 值 vs 窗平均：stage 1 = **−1.020±0.075**（1k 窗 −1.84、8k 窗
  −1.54）；stage 3 = **−0.999±0.089**（−1.96 / −1.21）
- flip 率時間平坦（首/末四分位一致）；每鏈 flips 中位數 2495（s1）/714（s3）

## 結論
答案是 (c)：**burn=200 遠短於 10–15k 的 dressing transient**，每個窗都把
爬升段平均進去——窗越長偏差越小，「趨近平台」被誤讀成「永不收斂」。
非 glassy、非 estimator bug（兩臂同平台 = 直接的 detailed-balance 檢查）。
window-scaling 多點法只能報警，**分不清慢收斂與固定 transient 攤薄**；
burn 長度必須由兩臂時序診斷決定。§9 的「定案」−6.65（1k 窗）與 27121
的 −4.46（32k 窗）都是偏低值；收斂 anchor 待 27135（預期 ≳ −4，六 stage
平台和）。

## 後續
- → 27135（burn 16k + record 32k 生產）＝最終 anchor；與 27121 drag 曲線
  組合出 M=227600 的完整 O_C(δ)
- 工具沉澱：`growth_mixing_diag_mpi.py`＋兩臂分析腳本（commit 3a74e1a）、
  `--growth-burn-per-stage`/`GROWTH_BURN`（commit 9d11905）、
  design doc §10
- 備案（未動用）：BRA Ω-anneal anchor（arXiv:2412.01384）——固定 string
  沿 Ω 重加權到 x-極化參考點，結構性繞開 sector toggle
