# E14 — SSE 非對角弦引擎：熱態 ⟨X_C⟩ 的 Jarzynski 估計（2026-07-12）

## 目的
量熱平衡態的 O_C(β) = Tr[X_C e^{−βH}]/Z，X_C = ∏_{i∈C}σᵢˣ。
物理動機（承 E13 的古典/量子判別討論）：**對角混合系綜的閉 X-loop 恆等於零**
（X_C 把 covering 映到不同 covering，⟨c|X_C|c⟩=0）——熱態的 O_C 提供
「RCSL 基準線」；對照 QAQMC swept 態的同一量就是相干性的直接量度。
也是 X-BFFM（排 trivial）的基礎設施。

## 設定（實作）
- `csrc/cpu/detail/sse_off_diagonal_core.{hpp,cpp}`：QAQMCOffDiagonalCore 的熱態移植
  （seam mask ＋ half-line ＋ λ-topology Metropolis ＋ Jarzynski）
- 週期 τ 的三個結構差異：(1) walk 繞過 τ=0（過界 commit 翻 `state_`）、
  (2) 無效條件 = 該站整條 string 無單站算符、(3) seam 快取更新按方向分流
  （右走翻 plus、左走翻 minus、繞圈翻 state_——QAQMC 版 commit 一律翻 plus
  的疑點 07-12 晚回查**證實是 bug**，連帶再挖出兩個，見「QAQMC 回查」節）
- Hooks：`diagonal_update`/`build_vertex_lists` fill pass/`measure_chi_f_terms`
  在 p==m_star 做 seam XOR；`topology_sweep` 開頭 recompute snapshots
  （cluster update 會弄髒）；m_star 預設 0（τ=0，M 成長後仍合法）
- Python：`src/engines/sse_string_work.py`（軌跡邏輯直接繼承 QAQMC wrapper）、
  driver `src/mpi/sse_string_work_mpi.py`（CLI 同構、chunk checkpoint、warm start
  ——注意 SSE 的 config 是 β/δ-特定的，不像投影版 K-無關）
- 部署：`main_scripts/slurm_scripts/run_kagome_offdiagonal.sh`（LOOP=1 選閉環、
  0 選開弦；STRING_SITES 可顯式給）＋ `probe_sse_offdiagonal_runtime.sh`

## 結果（驗證）
1. **⚠ 週期邊界特有 bug（已修）**：軌跡間 `set_seam_mask(0)` 硬重設在 trace
   下破壞 worldline 閉合（宇稱約束 parity(σˣ ops)⊕bit=偶）→ 第二條軌跡起全錯
   （O_C 偏 5.7×）。修復：`set_seam_mask_consistent` 自動修宇稱（翻該站首個
   ±1 算符型別；無算符則把 identity 種成 −1 op，交給 decorrelation 重平衡）。
   QAQMC 當時被認為開放邊界天然免疫（原註解的論證）——**07-12 晚證實是錯的**，
   ⟨0|…|0⟩ 兩端釘死跟 trace 一樣有每站閉合約束，見「QAQMC 回查」節
2. **ED 驗證 4/4**（N=6 chain）：正/反向 Jarzynski、單站與雙站 string 全部
   |z|<1σ 級一致；固定-λ 扇區占比 = λO_C/(1−λ+λO_C) 精確命中（detailed
   balance 的獨立確認）；seam 空罩時 E/⟨n⟩ 與無 seam 引擎一致（hook 無副作用）
3. **Driver 端對端**（mpiexec×2）：O_C=0.3234 vs ED 0.3260，n_eff=548/1000、
   zero_frac 3%（健康的 Jarzynski 診斷）
4. 全套回歸 74 passed

## 結論
熱態非對角弦管線生產就緒。判別實驗（E13 討論的包）現在有了第一個工具：
6×6 δ=4.25 β=6–20 的閉 X-loop O_C(l) —— RCSL 預測隨 β 增大趨零/指數小，
且與 swept 態（QAQMC string-work，同幾何）的差 = 相干性。

## QAQMC 回查（07-12 晚，同日修復）：三個 bug，互償假驗證
回查左走 cache 疑點時在 QAQMC string 管線挖出**三個**bug（SSE 版三者皆無）：
1. **左走 commit 更新錯快照**（qaqmc_off_diagonal_core.cpp:160）：一律翻
   plus；左走 terminal 在 seam 前，真值是翻 minus、plus 不變（n⁺=n⁻⊕b，
   seam bit 與 n⁻ 同翻互消）。實測左走 commit 1626/1626 全汙染快取。
2. **topology_sweep 吃髒快照**：mc_step = diagonal→cluster，cluster segment
   flip 跨 m_star 就改 seam 占據但不刷新快照（只有 diagonal 會刷），實測
   mc_step 後 75% 髒。修法同 SSE：sweep 進場 recompute。
3. **軌跡重設破壞 worldline 閉合**：⟨0…0|…|0…0⟩ 兩端釘死 ⇒ 每站
   parity(σˣ ops)==seam bit（跟 SSE trace 宇稱同構——原「開放邊界免疫」
   註解只論證了單站算符權重不變，沒管閉合）。wrapper 軌跡間
   `set_seam_mask(reset)` 硬重設 ⇒ **軌跡交替 valid/broken**（實測宇稱
   [0,0]→[1,1]→[0,0]…），一半 J 樣本來自非物理扇區，decorrelation 修不掉
   （所有 kernel 都保宇稱）。修法：新增 `set_seam_mask_consistent`（鏡像
   SSE 的宇稱修復；raw setter 保留給「回存已錄 (ops,mask) 對」的白箱用途），
   wrapper 的 thermalize/軌跡重設改用之。
- **互償假驗證的教訓**：舊 two-site ED 測試（K=400）曾以 rel_err 0.002
  「通過」——bug 1+2 的 kernel 偏差（固定 λ=0.5 扇區占比 +13.7σ，把 O_C
  拉低）與 bug 3 的破缺扇區偏差（拉高）恰在該 K 相消；「K 越大越準」的
  表象其實是 kernel 偏差的指紋（正確 AIS 對任何 K 無偏）。單修一個 bug
  測試反而炸更兇（rel_err 0.42），三個全修後 K=50/150/400 全部 <0.6% 且
  n_eff 隨 K 單調升（1430→3686 @K=400：正確 kernel 效率也翻倍）。
- 固化回歸：halfline Test 5（commit/sweep 後快取==recompute）、Test 6
  （mask 重設宇稱守恆）；two-site 第二測改為「每個 K 都無偏 + n_eff 隨 K
  升」（舊的 err 單調性斷言獎勵的正是 kernel 偏差）。
- 歷史數據影響：QAQMC string-work 尚未跑過生產，無資料需重跑；受影響的
  只有舊測試的「驗證」本身。

## 後續
- ⚠ 生產前決定 X-loop 的正確幾何：auto-select 目前給 Z_l 幾何的 site sets，
  X('t Hooft) 迴圈的對偶路徑是否同一組 sites 要對 Semeghini/Vu 的定義確認
- probe → 6×6 pilot（s=2 閉環，β=6 對照你的 8×8 資料溫度）
