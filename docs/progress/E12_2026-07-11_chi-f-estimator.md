# E12 — χ_F（fidelity susceptibility）估計子：SSE WLT ＋ QAQMC 速度校準（2026-07-11）

## 目的
trivial→RCSL 轉變（W–P：VdW δ≈3.5）沒有局域序參量，定位要靠 χ_F。建兩條互相独立的路：
(A) SSE 平衡態 Wang–Liu–Troyer 估計子（W–P ref [32]，PRX 5, 031007）；
(B) QAQMC sweep 的密度落後 ∝ v·χ_F（Liu–Polkovnikov–Sandvik PRB 87, 174302 的 metric-tensor 修正）。
兩者都先對 ED 釘死歸一化，之後可互相驗證＋做 KZ 定位（配 FM 弦比值交點）。

## 設定
- 分支 `z2_spin_lake`。改動現位於：`csrc/cpu/sse_core.hpp`、
  `csrc/cpu/detail/sse_core.cpp`（`bond_dlnW_` 表＋`measure_chi_f_terms`）、
  `csrc/cpu/module/bindings_sse.cpp`（run() 加 `measure_chi_f`）、`src/mpi/sse_mpi.py`（`--chi-f`，chunk 存
  `chi_gl/chi_gr/chi_glgr` bin means）、`plots/plot_sse/plot_chi_f.py`（jackknife 組裝＋δ-scan 圖）
- 估計子：χ_F = ½(⟨G_L G_R⟩−⟨G_L⟩⟨G_R⟩)，G = Σ_p ∂δ ln W_p over 半條 string。
  δ 折在 bond 權重裡（raw1/raw2/raw3 與 cij 都依賴 δ）→ 每 (bond, spin-state) 的
  ∂δ ln W 解析預算（含 cij 的 min/argmin 分支）
- 驗證：ED 譜公式 χ_F(β)=∫₀^{β/2}τ[⟨V(τ)V(0)⟩−⟨V⟩²]dτ，N=6 chain、Rb=1.4
- 測試：`tests/engines/unit/test_sse_chi_f.py`（3）、`test_qaqmc_chi_f_calibration.py`（2）；
  全套 engines+mpi unit 68 passed

## 結果
1. **⚠ slot-cut 偏差（方法學教訓）**：把 β/2 切割取在 slot M/2 會系統性偏差
   （δ=1.6 高統計 +4.8%≈+4σ，三 seeds 同向）——`adjust_M` 讓 M≈1.33·n_ops，
   每格≤1個 op 的排他讓 slot↔τ 映射有 O(1) 反聚束修正。
   **正確做法**：時間切割用順序統計精確抽樣——j~Binomial(n_ops,½)，序列前 j 個 op 屬左半
   （n_ops 含 Ω-ops，其 g=0 但佔時間）。修正後六組高統計檢查全部 |z|<1σ。
2. **SSE vs ED**（N=6 chain，800k samples）：δ=0.8/1.2/1.6 @β=4 全部 ±1σ 內
   （例：δ=1.6 → 0.0798±0.0017 與 0.0822±0.0018 vs ED 0.08118）；driver 端對端
   （mpiexec×2→chunk→jackknife）+0.1σ/+2.3σ/+1.6σ（16 bins）
3. **QAQMC 速度校準（ED 精確演化）**：對稱點（sweep 終點）密度落後
   dens−dens_gs = −c·Δλ·Ẽ·χ_F/N，Ẽ=offset−E_gs；**c→2**（M=1600 時 2.20，隨 M 收斂）
   ——與虛時 APT 的預測常數 2 一致，速度認定 ṽ=Δλ·Ẽ 定案
4. **⚠ 非對稱點沒有線性項**：回文算符序列的轉置對稱（t↔2M−t）使對角觀測量對 v 是偶函數
   → 曲線中段的落後是 **O(v²)**（實測 dev∝1/M²）。定量 χ_F 只能在 sweep 終點取；
   中段 v² 落後係數是更高階 gap 冪次的感受率，仍可當峰位定位器

## 結論
兩條 χ_F 管線都通過 ED 釘死：SSE 端是生產級（--chi-f 開關、成本 O(n_ops)/樣本、
chunk 格式相容 warm start）；QAQMC 端的正確用法修正為「終點 χ_F（線性、c=2）＋
中段 v² 落後曲線（定位）」。方法學上抓到兩個 nontrivial 陷阱（slot-cut 反聚束、
回文偶性），都已寫進測試。

## 後續
→ 生產 δ-scan：6×6 kagome_bond periodic β=20–40、δ∈[3.0,4.5] 加密 3.3–3.7，`--chi-f`
  （對照 W–P VdW 截斷版 δ≈3.5 的峰；我們是全 vdW 不截斷）＋幾個 L 做峰位外推
→ FM 弦比值（C_e/BFFM 已在測）交點分析當第二定位器；QAQMC 多 M 終點 run 做定量橋接
→ ⚠ measure_chi_f=True 會消耗 RNG（Binomial 抽樣）→ 開關影響軌跡位元重現性（統計無關）
