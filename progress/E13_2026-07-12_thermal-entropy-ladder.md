# E13 — 熱熵 β-ladder：直接witness RCSL 的殘餘熵平台（2026-07-12，進行中）

## 目的
用 W–P PRL Eq. (7) 的能量溫度積分量 S(T)/N，找 CSL 平台（PXP：ln2/6≈0.1155；
VdW 截斷版：低於 ln2/6 且隨 δ 變）。我們的新內容：**全 1/r⁶ 不截斷**的平台值與
δ-依賴、有限尺寸修正 ln2(1/6+1/N)、可用 E09 機器推到 β=320。
平台 = 廣延簡併流形的直接證據 = RCSL 圖像在可及溫度窗的最後一塊拼圖
（與 χ_F 單峰、C_m≡0、排他性=null 四件套）。

## 設定
- `src/mpi/sse_entropy_mpi.py`：升冪 log β-ladder，級間記憶體內 warm-start
  （op-string 對任意 β 合法，平衡化只在熱端付一次）；每級 chunk 格式輸出
- 解析錨點：lnZ(β₀) = Nln2 − β₀E_∞ + ½β₀²Var(H)，E_∞、Var(H) 精確
  （subset-overlap 代數＋minimum-image V_ij）——ED trace 驗證到 1e-9
- 積分：`plots/plot_sse/plot_entropy.py`，梯形＋SEM 線性傳播（S 對 E_k 線性）
- 測試：`tests/mpi/unit/test_sse_entropy_ladder.py` 2/2（N=6 chain 全譜對照）
- ⚠ 誤差結構：S(β) 的誤差被 β·σ(E(β)) 主導 → 大 β 端的 rung 需要最多樣本；
  平台在 T~0.05–0.5（β~2–20），**不需要深β**就能看到
- Pilot: 6×6 kagome_bond periodic δ=4.25、β∈[3e-4,20] 40級、8 ranks
  （`data/sse_entropy_6x6_delta4.25_pilot`）

## 結果（pilot，6×6 δ=4.25 periodic，8 ranks×2000/級，40級 β∈[3e-4,20]）
圖 `figures/sse_entropy_6x6_delta4.25_pilot/entropy.png`。**三層熵階梯全部可見**：
1. T→∞：S/N→ln2 ✓（錨點殘差 +2.4σ，對 S/N 的影響 ~β₀·resid/N ≈ 2×10⁻⁴，可忽略）
2. T~30–300 釋放第一層（V_nn=191Ω：三角內 blockade 凍結）→
   **平台 S/N ≈ 0.45 ≈ ln4/3 = 0.462**（每三角 4 態：空＋3 選 1——正是
   Vu/Mauron restricted Hilbert space 的流形熵！他們的截斷=把系統釘死在這層）
3. T~0.5–3 釋放第二層（δ 尺度：≤1/三角 → 恰好 1/vertex 的 Gauss law）→
   T≲0.2 落向 **ln2/6 平台**（0.19→0.12→…），但 pilot 統計在此已不夠
   （err(S/N)~0.03–0.06，被 β·σ(E) 主導——與設計預期一致）
- E/N(β=20) = −0.9083 與 E07/E09 生產值精確吻合（管線交叉驗證）
- 大β端個別 rung 有 2–3σ 波動（β=4.8 的 E/N=−0.919 低於基態值）：純樣本量問題

## 結果（追加 2026-07-12：8×8 δ=4.5 的 S(β=6)，混合方案）
使用者的生產 run（`sse_8x8_kagome_bond_beta6_delta4.5`，64r×512bins，
E/N=−0.96088(116)）當終點＋新跑的 8×8 ladder（β∈[3e-4,6]×36級，8r）當積分段；
兩邊 E(β=6) 一致性 0.8σ ✓、錨點殘差 −0.9σ ✓。

**S/N(β=6, δ=4.5, 8×8, 全vdW) = 0.140 ± 0.011**
- W–P 對實驗溫度區（β_eff=5–8）的宣稱 S/N~0.12–0.15：**命中**——首次在全 1/r⁶
  不截斷＋論文尺寸 N=384 驗證
- 高於 CSL 平台線（ln2/6=0.1155、有限尺寸 0.1173）約 2σ：T=0.167 還在最後下降段，
  與平台在 T≲0.1 相容
- 同一筆資料的快照：|A_v|=0.86、Z_l(2)=0.51、n≈1/4、Ψ_VBS=基線——**電荷 sector
  基態化＋磁 sector 帶著 ~ln2/6 的廣延熵**：RCSL 圖像的兩半在一個資料集裡同時成立
- 誤差 0.011 由終點項(0.007)＋ladder 頂端幾級(8r 低統計)組成；生產統計可到 ~0.005

## 結論（pilot 階段）
管線端到端成立、物理結構正確。附贈發現：**ln4/3 blockade 平台**直接可視化了
restricted-HS 近似「凍住」的自由度層——放進論文對 Vu/Mauron 的方法學評論很有力。
CSL 平台（ln2/6 vs 更低的全 vdW 值）的定量判定需要生產統計：大β端樣本 ×50–100。

## 後續
- 生產：δ ∈ {3.5, 4.0, 4.25, 4.5, 5.5}（＋7.5 對照晶體側的熵塌縮）、64 ranks、
  大β端加密樣本；8×8 一條對照有限尺寸
- 方法備選：quantum Wang–Landau（Troyer–Wessel–Alet）一次 run 全溫度曲線——
  只有 diagonal update 要改，ladder 成本痛了再上
- 同批資料順便出 C(T)=dE/dT（兩尺度熵釋放的峰結構）與 ⟨n⟩(T)
