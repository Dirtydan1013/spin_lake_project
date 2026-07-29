# E15 — 熱熵 ladder 低溫延伸 β=20→81.75（2026-07-27）

## 目的
把 E13 的 6×6 δ=4.5 熵 ladder 從 β=20 往下積 4 倍（T 到 ~0.012Ω），檢驗 S/N 的
ln2/6 平台（RCSL/CSL 候選訊號）是否在更低溫維持、還是出現熵釋放（有序化）。

## 設定
- `src.mpi.sse_entropy_mpi`,kagome_bond 6×6 PBC,δ=4.5,full 1/r⁶,64 ranks
- 延伸段:10 rungs,β = 20×r^k(k=0..9,r=1.16934 精確接續原 prod 公比),
  β_max=81.754;每 rung 128 萬樣本(64×10×2000),seed=43,**冷啟動**
  (原 run 未存 config)+ 4000 sweeps 熱化;job 27049,9.4 h,~600 CPU-h
- 資料 `data/sse_entropy_6x6_delta4.5_prod_ext80/`(+`configs/` 64 rank 的
  final_config @β=81.75 — 以後可 `--config-in` warm-start 繼續往下)
- 合併分析:`plots/plot_sse/plot_entropy.py` 新增多 run_dir 合併
  (overlap rung pull 檢查 + inverse-variance);圖
  `figures/sse_entropy_6x6_delta4.5_prod/entropy_merged_beta80.png`
- ⚠️ driver 本次新增:rung 末寫 final_config、`--config-in` warm-start
  (未 commit 時已用於本 run 的僅 final_config 寫出;--config-in 本 run 未用)

## 結果
- **Overlap rung β=20 pull = −4.5σ**(ext −209.09±0.07 vs old −209.52±0.07)。
  per-chunk 診斷:ext β=20 的 chunk 均值 −0.9659→−0.9690 單調爬升到第 10 個
  chunk 仍未收斂 → **冷啟動殘留熱化偏差**(β=20 的 τ ~ 數千 sweeps);
  ext 第 3 個 rung(β=27.3)起完全平坦(−0.9692,無趨勢)→ ladder 鏈在
  2–3 個 rung 內自我修復,僅 ext β=20(可能含 β=23.4 輕微)受汙染。
- E/N 低溫端平坦:β=27→81.75 全部在 −0.9691±0.0005,無可見下沉。
- S/N 對 overlap 處理穩健(四種變體):

| 處理 | S/N(β=20) | S/N(β=81.75) |
|---|---|---|
| 合併 overlap,burn .25 | 0.1198(47) | 0.1069(120) |
| 合併 overlap,burn .50 | 0.1197(60) | 0.1144(156) |
| **丟棄 ext β=20,burn .25(建議)** | **0.1006(64)** | **0.1102(121)** |
| 丟棄 ext β=20,burn .50 | 0.1086(80) | 0.1162(156) |

  參考:ln2/6 = 0.1155,ln2(1/6+1/N) = 0.1187。

## 結論
S/N 平台從 T=0.05 延伸到 **T≈0.012(β=81.75)不動**,終點 0.110(12) 與 ln2/6
CSL 平台一致、與熵歸零(有序化/gap 開啟)不相容 — δ=4.5 的 RCSL 圖像再往低溫
4 倍成立。overlap 檢查抓到的 −4.5σ 是**新 run 自己的冷啟動偏差**(教訓:β≥20
的 rung 冷啟動需要 ≳2 萬 sweeps 熱化,或一律用 warm-start);正確做法是丟棄
ext 的 β=20 rung,只用它當診斷。

## 後續
- 以後繼續往下積:`CONFIG_IN=data/sse_entropy_6x6_delta4.5_prod_ext80`
  warm-start(configs/ 已存),免熱化、無冷啟動偏差問題。
- rung-decimation 收斂檢查尚未做(慣例);β·SEM(E) 端點誤差是低溫端主導項,
  要壓終點誤差 → 加倍 β_max 端樣本數。
- 工具沉淀(待 commit):sse_entropy_mpi final_config + --config-in;
  plot_entropy 多 run_dir 合併;run script BETAS/CONFIG_IN。
