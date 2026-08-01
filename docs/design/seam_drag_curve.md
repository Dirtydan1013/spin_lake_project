# Seam-drag：一次 QAQMC 模擬取得整條 ⟨X_C⟩(δ) 曲線（Tier 2 計畫書）

日期：2026-07-29（M0）；2026-07-30 M1–M3 完成　分支：`z2_spin_lake`
狀態：**已完成並通過最終驗收**（1D chain vs-ED gate 全綠，見 §6 實測結果）。
實作重點更動：主力估計子從 trajectory Jarzynski 改為 **RB ladder**（§2.4，
開發中發現的重尾問題）；trajectory 版保留作 bracketing 診斷。

## 1. 動機

Diagonal profile 一次 sweep 就給出整條參數曲線（slot t ↔ δ_t）。off-diagonal 弦
⟨X_C⟩ 目前只能在固定 cut 位置 `m_star` 用 λ-Jarzynski（seam 強度插值）測一個點，
逐 δ 掃描要逐點跑。本計畫把 **Jarzynski 的驅動參數從 seam 強度 λ 換成 cut 位置 m**
（"拖 seam"），使一條 trajectory 沿途在每個中間 m 都吐出一個 work 樣本，
一次模擬得到整條 Z_X(m)/Z_X(m₀) 曲線；配合現有 λ-Jarzynski 在錨點 m₀
的 Z_X(m₀)/Z_∅，組合出整條

    O_C(m) = Z_X(m) / Z_∅  ≈  ⟨X_C⟩ at δ_m        （Z_∅ 與 m 無關，全曲線共用分母）

## 2. 估計子與正確性論證

### 2.1 拖 seam 的權重比是「精確」的

配置 σ =（op string, seam mask b）固定時，cut 從 m 移到 m+1 只把 slot p=m 的
operator 從 seam 之後（C 上 frame 翻轉）換到 seam 之前（不翻轉），其餘所有 slot
的傳播態逐一相同（XOR 可交換）。因此

    W_{m+1}(σ) / W_m(σ) = w_p(unflipped) / w_p(flipped)

且逐項只有三種情形：
- single-site op（type ±1）：權重 = 常數 `site_W_` → ratio = 1；
  type −1 只需把 n⁻ 在該 site XOR（snapshot bookkeeping）。
- bond op 不碰 active string site → ratio = 1。
- bond op 碰 active string site → ratio = W[n⁻端點]/W[n⁺端點]，
  在**同一個** `delta_at(p)` 求值（δ schedule 不動，動的只是 cut）。

推論：
- 對固定 σ，跨一個「區塊」的總 log ratio 是精確可算的 O(區塊長) read-only walk
  （與 `build_half_line_proposal` 同款數值：`compute_bond_W_inline` + `inv_coord`
  + 1e±100 重整化乘積）。所以 block-shift（一次跨多個 slot、中間不 relax）
  是合法的 Jarzynski protocol——任意切換速度都無偏，只是 work 分佈變寬。
- mask = 0 時 n⁺ ≡ n⁻ → 全程 ratio ≡ 1，log ratio ≡ 0（免費的不變量測試）。
- 移動 cut 不改 per-site worldline closure parity（parity 約束不含 m_star），
  不消耗 RNG，對既有 run 是 opt-in、bit-compat。

### 2.2 Jarzynski 恆等式（curve 版）

trajectory 從 Z_X(m₀) 平衡態出發，交替「block shift（累積 log ratio）→ 在新 m
relax（`mc_step`，seam hook 已支援任意 m_star_）」。對路徑上每個記錄點 m：

    E[ exp(Σ log-ratio 累積到 m) ] = Z_X(m) / Z_X(m₀)

一條 trajectory 給出**整條**單調方向的曲線樣本；左右兩側各跑一族（m₀=M 出發，
family R 拖向 M_total−1、family L 拖向 1）。ED gate 驗的是這個恆等式本身
（對任意 M/v 精確成立）；「R(m) ≈ 瞬時基態 ⟨X_C⟩」的 adiabatic 詮釋
與 O(v) 修正（mirror-slot 平均消 odd 項）屬於 production 階段（M4），不進本輪 gate。

### 2.3 已知陷阱對照（沿用 E14 紀律）

- **snapshot staleness**：`cluster_update` 會弄髒 seam snapshots，只有
  `diagonal_update` 會刷新 → `seam_drag_to` 進場先 `recompute_seam_snapshots`
  （O(M)、無 RNG，與 `topology_sweep` 同款 guard）。
- **parity/sector**：drag 不動 op types → closure 不受影響；trajectory 重置用
  `seam_set_position`（只改 m_star_ + 重算 snapshots，mask 不動），之後必須
  decorrelate（重置後的 config 是舊 m 的 ensemble 樣本）。
- **零權重**：目前 config 的 w(flipped) 必 > 0；w(unflipped) 在 ε>0 的 cij margin
  下一般 > 0，仍保留 guard：遇 0 回 −inf，Python 端當 exact-zero trajectory
  處理（同 λ 端點邏輯，計入 zero_weight_fraction）。
- **單點 ED match 可能是互償假象**：gate 設計含 rate-independence 抽查
  （兩種 relax 步數同答案）與 drag-only / end-to-end 雙層比對。

### 2.4 重尾問題與 Rao-Blackwell 化（M3 開發中的關鍵發現）

實測（N=5 chain, ε=0.01）trajectory Jarzynski 對 ED 有 ~11% 的單邊系統偏差
（fwd 偏低、1/rev 偏高、真值被夾住），且偶發 n_eff 崩塌（1500→24）。根因：
**跨 slot 的原始權重比是離散重尾的** — cij 平移後 W[11] ≈ ε·m_abs 可為 O(0.01)
而另一 frame 的 W ≈ O(1)，單次 crossing ratio ~ 1/ε ~ 100；這種 whale 事件
出現機率 ∝ 其權重（典型 FEP 病態：稀有樣本攜帶 O(1) 貢獻），可達樣本數內
收不了尾。放慢 protocol（逐 slot + relax）無效——m 是離散參數，
單步 work 不會因 relax 變小。

解法：**Rao-Blackwell 化**。ratio ≠ 1 只發生在 crossing slot 持有 diagonal op
時，而 QAQMC 固定長度字串中，給定 state trajectory，diagonal slot 的 op 選擇
是條件自由且 ∝ 權重（`diagonal_update` 的 alias+rejection 平穩分佈）。對 op
選擇解析求和：

    E[ratio | state] = 1（flip slot；σˣ site 由連續性決定，frame-blind）
                     = Λ_tgt(s)/Λ_cur(s)（diagonal slot）
    Λ(s) = N·(Ω/2) + Σ_b W_b(s)   （crossing slot 自己的 delta_at(p)）

whale 變成兩個 O(N) 和的比值——**逐樣本有界**，期望不變（RB 保均值）。
估計子隨之從 trajectory 形式改為 **ladder 形式**（與熱熵 β-ladder 同款方法論）：
逐 rung（單 slot）在 Z_X(m) 平衡態下取樣 E[Λ_tgt/Λ_cur]，累積 ln，
per-rung SEM 平方和給出誠實誤差棒。C++ 曝露 `seam_rung_rb_ratio(right)`
（read-only、無 RNG）；`seam_drag_to` 保留（rung 間移 cut ＋ trajectory 版仍用）。

代價與注意：
- rung 樣本是 Markov 鏈相關的 → SEM 可能低估；用 `n_sweeps_between_samples`
  控制並靠 rate-independence gate 驗證。
- rung 間 warm-start（上一 rung 末樣本 + burn-in）與熵 ladder 同款的
  二階小相關偏差；gate 未見可測影響。
- trajectory Jarzynski（`run_drag_trajectories`）保留：單邊實現值會因 whale
  往任一方向擺（不可做方向斷言！），fwd × 1/rev 幾何平均（BAR-lite）
  仍是有用的交叉檢查。

## 3. API 設計

### 3.1 C++（`QAQMCOffDiagonalCore` + `QAQMCEngine` passthrough + bindings）

```cpp
// 把 cut 從 m_star_ 拖到 m_new（任意方向），回傳目前配置的精確
// log[W_{m_new}(σ)/W_{m_star}(σ)]（零權重 → -inf）。更新 m_star_ 與
// seam snapshots（增量）。進場先 recompute_seam_snapshots。O(|Δm|)。
double seam_drag_to(QAQMCEngine& eng, std::int64_t m_new);

// 無 work 重錨定：m_star_ = m_new + recompute snapshots，mask 不動。
void seam_set_position(const QAQMCEngine& eng, std::int64_t m_new);
```

### 3.2 Python（`src/engines/qaqmc_string_work.py` 擴充）

```python
# 主力（RB ladder，§2.4）：
res = eng.run_drag_ladder(
    m_grid,                       # 單調 int 陣列（record 點）
    n_samples_per_rung=400,       # 每個單-slot rung 的平衡樣本數
    n_sweeps_between_samples=1,   # 樣本間 mc_step 數（decorrelation）
    n_burn_per_rung=5, m_anchor=None)
# res: StringDragLadderResult — log_r / log_r_sem（quadrature 傳播）/ r
#      + rung_m / rung_log / rung_sem 逐 rung 診斷

# 診斷用（trajectory Jarzynski；重尾，見 §2.4）：
res = eng.run_drag_trajectories(
    m_grid, n_trajectories, decorrelation_steps=100,
    n_qaqmc_sweeps_per_shift=1,   # 每個 switch block 之間的 relax
    slots_per_block=1,            # switch block 大小（record grid 與之解耦）
    m_anchor=None)
# res: StringDragRunResult — log-sum-exp 平均 + n_eff/p_max/zero_frac 逐 m 診斷
```

C++ 新原語（`QAQMCOffDiagonalCore` + passthrough + bindings）：
`seam_drag_to(m_new)`（精確 log 權重比、更新 m_star/snapshots）、
`seam_rung_rb_ratio(right)`（read-only RB 單-rung 比值）、
`seam_set_position(m_new)`（無 work 重錨定）。

組合曲線：`O_C(m) = O_C(anchor)[現有 run_trajectories λ-協議] × r[m]`。

## 4. Milestones 與測試設計

（註：以下為 M0 原案。實作後主力估計子改為 RB ladder — Gate A/B 以
`run_drag_ladder` 執行、另增 RB brute-force 單元與 Jarzynski geo-mean
診斷測試；結構不變，實際清單見 §5。）

### M1 — C++ primitive（單元測試 `tests/engines/unit/test_qaqmc_string_drag.py`）
1. **brute-force 精確比對**：小系統（N=5, M=8）熱化後取 op string，Python 端
   直接展開兩個 m 的完整配置權重比（逐 slot 傳播 + `compute_bond_W` 重算），
   與 `seam_drag_to` 回傳值比到 1e-10。多組 (m, m′, mask) 隨機抽查。
2. **可逆性**：m→m′→m 的 log ratio 和 == 0（同一 config，precision 級）。
3. **mask=0 不變量**：任意 drag log ratio == 0。
4. **snapshot 一致性**：drag 後 `state_at_seam_minus/plus` == 從頭
   `recompute_seam_snapshots` 的結果。
5. **parity 保持**：drag + `mc_step` 交替後，per-site σˣ parity == seam mask
   （沿用 test_qaqmc_string_seam.py 的檢查手法）。

### M2 — Python orchestration
- `run_drag_trajectory` / `run_drag_trajectories` + 診斷；煙霧測試
  （2-3 traj 跑通、樣本形狀、單調 grid 驗證、重置-decorrelate 流程）。

### M3 — 1D chain vs-ED integration gate（**最終驗收**）
`tests/engines/integration/test_qaqmc_string_drag_vs_ed.py`，
參數對齊現有 Jarzynski 測試（N=5 chain, M=16, Ω=1, δ:0→1.5, Rb=1.2, ε=0.01,
site C={2}, neighbor_cutoff=1）：
- **Gate A（drag-only，尖銳）**：full-mask 在 m=M 熱化，左右兩族 drag 過
  grid（如 L: 12,8,4,2；R: 20,24,28,30），r̂(m) vs ED 的
  Z_B(m)/Z_B(M)（`qaqmc_exact_string_zratio` 逐 m_star 呼叫），逐點 rel_err < 0.10，
  n_eff > 0.05·n_traj。
- **Gate B（end-to-end）**：組合 λ-anchor → O_C(m) 曲線 vs ED O_B(m)，
  逐點 rel_err < 0.15。
- **rate-independence 抽查**：n_qaqmc_sweeps_per_shift ∈ {1, 4} 在一個代表 m
  上一致（差 < 合併誤差）——biased kernel 的指紋檢查（E14 教訓）。
- **回歸**：既有 string 測試（string_ed / halfline / seam / jarzynski_vs_ed /
  two_site_vs_ed）全綠。

### M4 —（本輪之外）production 化
MPI driver（kagome、KP 弦）、mirror-slot 平均消 O(v)、block 排程與 per-m
n_eff 自動監控、雙向 drag / BAR、與 diagonal profile 同 run 合測。

## 5. 實測結果（2026-07-30，M3 驗收）

單元（`tests/engines/unit/test_qaqmc_string_drag.py`，pytest 收，6 項）：
drag == brute-force 全重算（1e-9）、可逆性、mask=0 恆 0、snapshot/parity
不變量、RB ratio == Λ 比值 brute force（1e-12）、orchestration 煙霧。全綠。

整合 gate（`tests/engines/integration/test_qaqmc_string_drag_vs_ed.py`，
pytest 收，N=5 chain / M=16 / C={2}，ED = `qaqmc_exact_string_zratio` 逐 m）：
- **Gate A** RB ladder（1200/rung, sweeps=2）：左右 8 點 rel_err 0.5–2.6%
  （容差 8%）。
- **Rate-independence** sweeps 1 vs 4：0.2–1.3%（容差 10%）。
- **Jarzynski geo-mean**（fwd × 1/rev BAR-lite）：8.5%（容差 15%；
  單邊實現值 1.36 / 0.94 展示重尾擺動）。
- **Gate B** 組合曲線 O_C(m)（λ-anchor 1.0% × ladder）：8 點 0.5–2.8%
  （容差 12%）。
另：off-center m_star 的平衡 sector-residence 抽查（m=12,16,20,24）對 ED
0.2–1.1% — 首次驗證 mc_step/topology 在任意 cut 位置的採樣（此前所有測試
只跑過 m_star=M）。回歸：`tests/engines/{unit,integration}` 120 passed
+ 手動 string mains 全綠。

## 6. M4-1/2/3（2026-07-30 晚，mirror 平均、M-scaling、two-site）

**新 API**：`run_drag_curve_mirrored(m_grid_forward, ...)` — 左右兩族共用
m=M anchor、逐點幾何平均；`run_drag_ladder` 增 `n_equil_at_anchor`
（第二族從前一族遠端 config 重錨定時必須 > 0 — 順帶修掉 gate 裡原本的
輕微 sloppiness）。新 gate：mirrored 曲線 vs ED（0.3–0.7%）、two-site
C={1,3} ladder vs ED（0.1–0.8%，驗 bond 同時碰多個 active site 的 frame
bookkeeping）。QMC mirrored @M=64 對 ED geo-mean：2–5% ≈ 1σ。

**M-scaling 研究（ED 精確，M=16→2048，δ ∈ {−1, 0.5, 2, 4}，
`plots/seam_drag_mirror_vscaling.png`）— 結論比 §2 的天真預期更微妙：**

1. **單 branch 的 finite-v 誤差本來就幾乎是 v 的偶函數**：up/down 兩 branch
   幾乎重合（M=64 時 |L−R| ~ 5%，共同 lag 卻 40–80%）。palindrome 轉置對稱
   對 X_C 插入近似成立（差一個 one-slot schedule shift）。所以 mirror 平均
   **不改變收斂階** — 尾端斜率單 branch 與 mirrored 都 ≈ 2。
2. mirror 平均的實際價值：(a) 消掉殘餘奇數項；(b) 撫平單 branch 的
   **零交叉病態**（|O−G| 在大 M 端因變號而假性下沉/抖動，見 δ=−1、δ=2
   面板），mirrored 曲線是乾淨單調的 ~1/M² → **可安全 Richardson 外插**。
   成本幾乎為零（兩族共用 anchor）→ production 預設開啟。
3. **絕熱收斂很慢**：ε-offset 使 v = Δλ·Ẽ 在小 M 不小（N=5 chain 在 M=256
   仍差 GS 10–20%；M=2048 才到 10⁻³–10⁻⁴ 量級；δ=0.5（⟨σˣ⟩ 峰附近）
   最慢，M=2048 斜率仍只有 1.7）。v ∝ N/M ⇒ kagome 生產要「基態極限」的
   數字必須靠 **M-序列 + 1/M² 外插**（對 mirrored 曲線做），不能單靠加大 M；
   這跟熵 ladder 的 rung-decimation 紀律同精神。有限-v 曲線本身（非平衡
   sweep 的物理）則不受此限 — 那正是 spin-lake 情境關心的對象。

## 7. M4-4/6/8（2026-07-30 深夜：block-RB、driver、kagome 標定、scripts）

**Block-RB rung（生產可行性的關鍵）**：生產 M=10⁵ 下逐 slot ladder 要 10⁵ 個
rung，不可行。給定 worldline，各 crossed slot 的 diagonal-op 選擇條件獨立 →
block 的 RB 條件期望精確分解為逐 slot Λ 比值乘積，一次 O(k·n_bonds) read-only
walk（C++ `seam_rb_log_ratio_to(m_new)`；`seam_rung_rb_ratio` 改為其 k=1 特例）。
單元測試：block == 逐 slot 組合（1e-10）、read-only 不變量。

**MPI driver**（`qaqmc_string_work_mpi --drag-*`）：λ-anchor 之後接 drag phase
（reverse-sector 熱化 → 每 rank `--drag-repeats` 趟 mirrored ladder），per-pass
chunk 進 `<ckpt>/drag/rank{r}.h5`，rank0 以 pass 散佈聚合誤差、逐 K 組合
O_C(δ) 曲線存 HDF5 `drag/` group（含 `curves/K{K}/o_curve`）。anchor 誤差改由
`_aggregate_log_j` 內建 bootstrap（`log_o_c_sem_boot`）。無 drag flags 時行為
與舊版 bit-exact。測試：`tests/mpi/integration/test_qaqmc_string_drag_mpi_integration.py`
（單 rank plumbing）＋ 2-rank CLI 實跑（δ→slot 轉換、gather、HDF5 驗證，
數值對 ED 1σ 內）。⚠️ root PROD .so 尚未部署新 API — 生產前要 `mv` 部署。

**kagome 標定（kagome_bond 4×4 N=96 PBC，a=4、Rb=2.4、δ −2→6、M=4096，
size-2 中央 C_m 弦）**：

| δ | rung log-sd @spr=1 | 8 | 32 | 128 |
|---|---|---|---|---|
| 5.0 | 0.0034 | 0.011 | 0.20 | 0.55 |
| 4.25 | 0.0039 | 0.008 | 0.12 | 0.41 |
| 3.0 | 0.0038 | 0.023 | 0.09 | 0.43 |
| 1.0 | 0.0052 | 0.051 | 0.11 | 0.57 |
| −1.0 | 0.0057 | 0.045 | 0.18 | 1.68 |

1. **rung log-sd 不是 √spr 縮放**：spr ≳ 8 後轉為 ~線性（甚至超線性）增長 —
   worldline 虛時關聯讓 block 內逐 slot 貢獻同調累積。⇒「cost ∝ 1/spr」只在
   關聯 crossover 之內成立；**最佳 spr = 讓 rung log-sd ≈ 0.3**（此幾何 ≈ 64，
   scripts 預設已設 64；換幾何/M 用 probe 重標）。
2. 完整 11-δ mirrored 曲線（δ 6→−1、spr=64、200/rung、112 rungs）：**59 s**
   單鏈；log r 誤差 0.02（近端）→ 0.11（δ=−1）。timing：mc_step 0.92 ms、
   block eval 1.07 ms（O(M) snapshot recompute 主導）→ 每 rung 樣本 ≈ 2×mc_step。
3. rate check（sweeps 1 vs 3，獨立 seed，最遠點全程 drag）：z=1.1 ✓。
4. 分支不對稱 L−R 在此 M 很大（log 差 0.1–1.4）— M=4096 對 N=96 是很快的
   ramp（v ∝ N/M）；mirror 平均是必要的，且生產 M=10⁵（×25 慢）下會縮小。

**Scripts**：`run_kagome_string_work.sh` 增 `DRAG_*` 旋鈕（`DRAG_DELTAS` 非空
才啟用；預設行為不變）；`probe_string_work_runtime.sh` 增 drag 段 — probe 以
production grid/spr × 少量 samples 實跑，生產時間 = 實測 × (samples 比) ×
repeats（rung 數按構造相同），計入 CPU-hours 報告。

## 8. 成本備註

- drag 全程 O(M_total)/趟 + 每段 relax O(M)·steps + 每次進場 recompute O(M)
  → 一條 trajectory ≈ (n_grid × relax_steps + 2) 個 mc_step 的量級；
  對照 Tier 1（逐 m 獨立 λ-Jarzynski：每點 K×relax 個 sweep）為 n_grid 點
  共享一次 sector 橋接。
- work 分佈寬度隨 |m−anchor| 增長 → 曲線遠端 n_eff 下降是預期行為，
  由逐 m 診斷量化；必要時加密 relax 或縮短單族拖距（production 再調）。

## 9. Anchor v2：growth residence ladder（2026-08-01）

**動機**：λ-Jarzynski anchor 在 production 幾何（6×6 PBC、M=227600、vertex
hexagon loop C={84..89}、Rb=2.4）上失效——probe 27102/27103/27105：
n_eff 2–11%、O_C 有 4 倍 K-趨勢（K=1800→3600）、正反向括號反轉。根因
（toggle 接受率診斷）：half-line 混合慢 + 部分弦 sector 比值小（interior
stage 接受率 0–2%），一條 trajectory 要在單次 sweep 內穿越全部 6 個
bottleneck → work 分佈爆寬，K/relax 救不了。

**設計**：逐 bit 成長。stage k 凍結 bits 0..k−1（`set_seam_mask_consistent`）、
只讓 bit k 在 λ_k toggle；λ_k autotune 到 occupancy 平衡；
Z_{k+1}/Z_k = (P_on/P_off)·(1−λ_k)/λ_k 精確；O_C = ∏。平衡方法——慢混合
只花 sweep 數，不毀有效性。flip 數 = 誠實有效樣本數；未解析 stage
poison（±inf）不給假數字。結束停在 full-mask sector 直接接 drag。
wrapper `run_growth_residence_ladder`；driver `--growth-anchor`
（`--K-values none` 可全跳 λ phase）；scripts `GROWTH_*`。

**又一個 E14 快取坑（修掉）**：裸呼叫 `attempt_string_toggle` 吃到 cluster
弄髒的 seam snapshots → residence 比值隨 λ 系統性偏（λ=0.63 時 19%）。
取樣迴圈進場 recompute（同 topology_sweep 慣例）。教訓通則化：**任何**
繞過 `topology_sweep` 直接用 seam 原語的路徑都必須自己 recompute。
（也影響先前的接受率診斷數字——定性結論不變。）

**池化聚合（27113→27114 的教訓）**：per-rank autotune 下，hard stage 沒
flip 的 rank 整條 ladder 作廢（47/64），且「倖存者」有強選擇偏差
（27113 的 −6.8±1.7 是壞數字）。修正：rank 0 短 ladder 調 λ → broadcast
→ 全體同 λ 採樣 → 逐 stage 池化 occupancy/flips（可加、有效樣本 =
總 flip 數）。27114（1000 samples/stage 池化）：

    O_C(δ=6) = e^(−15.81 ± 0.43) ≈ 1.4e-7
    stages: −2.65(σ.10) −4.45(σ.25) −1.08(σ.04) −5.05(σ.32) −0.60(σ.04) −1.96(σ.10)
    hard stages 1,3（flips 129/80，λ≈0.998）＝主要成本

**狀態**：待兩個一致性檢查（27115 equil/rate ×3、27116 成長順序重排 —
不同中間 sector、同終點）確認後才定案；量級遠低於 λ-anchor 的 whale
軼事值（後者不可信）。物理註：drag 曲線往低 δ 續降 + anchor ~1e-7 意味
sweep 態的 loop coherence 遠低於平衡 |A_v|——若一致性檢查過，這本身就是
待理解的物理（finite-v 態 vs 平衡態的 off-diagonal 差異）。

**§9 後續（2026-08-01 深夜，定案）**：27114–16 的 z~8–20 不一致源於慢混合
transient：hard stage flip rate ~1%/sample，(a) stage 一律從 bit=OFF 起始、
(b) tune→production 的 λ 跳躍後立即開錄，兩者的 occupancy transient 佔了
錄製窗的可觀比例，且 flip-count 誤差棒看不見它。修正：λ-後平衡再開錄、
半數 rank 從 ON sector 起始（池化下首階互消）、toggle attempts 預設 4
（toggle walk ≪ mc_step 成本、flip rate 直接乘上去）、誤差棒以 64-chain
rank scatter 為下限。重跑一致性（27119/27120，兩種成長順序）：
log O_C = −6.653±0.409 vs −6.701±0.441，**z=0.08 ✓**，min flips 662–1097。
**O_C(δ=6) = (1.26±0.5)×10⁻³ 定案**（修正前的 1.4e-7 是 transient 假象，
λ-Jarzynski 的 0.006–0.079 是 whale 軼事——兩代錯誤數字都以一致性檢查
淘汰）。教訓入庫：稀有翻轉的 residence 估計，(1) 初始化要對稱、
(2) 換 λ 要重新平衡、(3) 誤差棒必須含鏈間散佈、(4) 永遠做路徑重排一致性檢查。
