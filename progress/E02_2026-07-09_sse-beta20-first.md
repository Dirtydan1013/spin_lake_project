# E02 — SSE平衡態 β=20 δ=5.5（第一次，舊.so）（2026-07-09）

## 目的
E01的sweep在δ=5.5看到M1/M2 stripe但M3缺席——平衡態對照：三個M取向在熱平衡下是否等佔？

## 設定
- `sse_mpi`，kagome_bond 6×6 periodic、a=4.0、Rb=2.4、δ=5.5、β=20
- 64 ranks×(4000 equil＋1563 samples)、chunk=250；job 26243
- 資料 `data/sse_6x6_kagome_bond_beta20_delta5.5`（**已被E05同名run覆蓋**）
- ⚠️ 跑在舊`.so`上（SSE observables API未部署），chunk只有legacy純量——分析改用
  `configs/rank{r}.h5`的final_config自旋組態（64張平衡snapshot）

## 結果
- final-config取向分類：**M1:M2:M3 = 21:20:23**（λ>1者18/14/21），weak 11 —— 對均分64/3≈21.3完美
- E = −262.007±0.263、密度 = 0.2549±0.0003 ≈ 1/4
- （combine_run的scalar shape bug順手修掉，commit `f2ba248`）

## 結論
**平衡態C3完全公平** → E01的30:32:1不是平衡物理，是sweep動力學的取向選擇。每條chain
仍卡單一取向（domain間ergodicity實際斷裂），但ensemble公平。

## 後續
→ E03排除幾何/測量嫌疑；→ E05用新.so重跑拿完整觀測量。
