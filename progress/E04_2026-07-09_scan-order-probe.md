# E04 — 掃描順序判別實驗＋mitigation（2026-07-09，PR #2）

## 目的
E03排除幾何/測量後，檢驗假說：所有rank共用的update site掃描順序（raster `(j*nx+i)*6+k`）
在sweep穿越ordering時決定了domain圖案。

## 設定
- `scripts/experiments/scan_order_bias_probe.py`：16條短sweep×兩組（M=552k、1000 equil、
  snapshot@fwd δ≈5.5），唯一差異＝site標籤
  - control：canonical標籤（=production）
  - permuted：每chain隨機permutation（物理不變、只變掃描幾何），輸出反映射回canonical

## 結果
| | 跨chain profile std | M1:M2:M3 (λ>1) | λ(M)平均 |
|---|---|---|---|
| control | 0.232（鎖定，=E01的0.24） | 6:9:**1** | 2.60, 2.54, 0.75 |
| permuted | **0.104（解鎖）** | 3:8:**5** | 1.23, 1.60, 1.56 |

圖案跟著「標籤」走 ⇒ **掃描順序就是元兇**（16條chain就重現E01的M3壓制）。

## Mitigation（merged, PR #2 → main）
- `--permute-site-labels`加入**四個driver**（profile/SSE/renyi work/string work），共用
  `src/mpi/site_permutation.py`；所有site-resolved輸入映射進engine標籤、輸出映射回canonical，
  **存檔格式不變**；warm-start config記錄`site_perm`並自動接續
- 之後預設改**ON**（`5cbc8f4`/`383bb28`；`PERMUTE_SITES=0`退回舊行為）
- 驗證：renyi work permuted vs canonical在K=64同步收斂到ED（1.4σ vs 1.6σ）；SSE
  permuted-vs-canonical差異落在canonical seed間散布內；63 unit tests＋warm-start round-trip

## 結論
非平衡sweep的domain選擇由演算法的確定性掃描結構主導；permutation恢復ensemble多樣性。
**教訓**：z檢定要用獨立chain誤差＋canonical-vs-canonical對照，少bin SEM會produce假警報。

## 後續
→ E08 production重跑驗證。
