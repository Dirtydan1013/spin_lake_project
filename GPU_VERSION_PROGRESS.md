# QAQMC GPU Version Progress

最後更新：2026-07-15
分支：`gpu_version`  
基準分支：`z2_spin_lake` (`31d6c5c`)  
目前狀態：standard、off-diagonal string-work 與 two-replica Rényi-work CUDA
核心均已實作並成功產生 V100/A100 fat binary；兩個 work MPI driver 亦具備
transactional HDF5 exact resume。standard、off-diagonal string-work 與 Rényi-work
的 true in-process batch-chain 均已完成，最終 clean-build 版本由 Slurm job
`26721` 在 V100 通過 108 個 real-GPU tests 及三個 probes，stderr 為空；
job `26720` 完成 production-size
`B=1/2/4/8` benchmark，V100 最佳點為 `B=4`。

本機（無可見 GPU）最新 gate：CUDA extension build clean；engine/MPI 廣泛
regression `105 passed`，其中 exact interruption/resume、HDF5 space reuse、完整
model fingerprint、collective multi-K resume、compact theorem 與 MPI affinity 的
聚焦 gate 為 `31 passed`；
另有 4 支 CPU string/Rényi reference scripts 通過。GPU suite `13 skipped`（因
login node 無 GPU，非失敗）。最新 CUDA build 的 real-GPU gate 仍由 Slurm jobs
負責。

## CUDA source modularization

- [x] 原本約 3,843 行的 `qaqmc_cuda_scan.cu` 拆成 runtime、standard/off-diagonal
  與 Rényi 三個 host translation units。
- [x] 公開 API 拆成 runtime、`DiagonalEngine`、`RenyiEngine` 三個 headers；舊
  `qaqmc_cuda_scan.cuh` 保留為 7 行 compatibility umbrella。
- [x] private engine state、共用 scan primitive、diagonal、off-diagonal、Rényi
  transition/topology kernels 分別放入 `csrc/cuda/detail/`。
- [x] Python binding 拆成 module entry、runtime、diagonal、Rényi 四個 translation
  units；entry point 只剩 12 行。
- [x] `csrc/cuda/README.md` 記錄模組責任與單向 dependency rules。
- [x] `sm_70;sm_80` fat binary clean build、Python symbol/import gate、登入節點
  13 GPU tests 正確 skip、CPU/engine/MPI regression `105 passed`。
- [x] 核心拆分後 V100 job `26706`：`98 passed`，string/Rényi probes 均成功，
  stderr 為空。
- [x] binding 拆分後 V100 job `26707`：`98 passed`，string/Rényi probes 均成功，
  stderr 為空。

## 狀態圖例

- `[x]`：已實作並完成相應驗證。
- `[-]`：已實作，但最新版本尚未完成真 GPU 驗證。
- `[ ]`：尚未實作或尚未開始。
- `[blocked]`：目前受外部資源阻塞。

## 目前成果摘要

- CUDA backend 為 optional build，不影響既有 CPU-only deployment。
- operator string、event buffers、`bond_spin` 與 RNG state 跨 sweep 常駐 GPU。
- packed `uint64` prefix-XOR 支援 `N <= 384`。
- standard diagonal、event build/sort、open-boundary cluster update 已 device-resident。
- V100、`N=216`、`M=2,760,000`、23,220 bonds、`G=600`：
  - CPU portable reference：`1.03095 s/step`。
  - V100 CUDA：`22.05 ms/step`。
  - 單 chain transition speedup：`46.8x`。
  - event/cluster workspace 配置後約 `1.0 GiB VRAM`。
- 現行 batch 模型為一個 process／一張 GPU／`B` 條 independent chains；
  immutable Hamiltonian/alias tables 共享，mutable state 與 RNG counters 每條獨立。
- off-diagonal engine 已將 seam-aware scan、half-line topology toggle、closure
  repair 與 string-work protocol 接上同一個 device-resident transition backend。
- Rényi engine 已支援兩條 replica operator strings、cut 後 channel remapping、
  channel cluster、dynamic mask topology update 與 device-to-device checkpoint。
- 兩個既有 MPI work drivers 均可用 `--backend cuda`，並會依 node-local rank
  分配 GPU，避免多 rank 全部落在 visible device 0。

## Milestone 0：分支、硬體與基準

- [x] 從 `z2_spin_lake` 建立 `gpu_version` branch。
- [x] 記錄 CPU QAQMC update flow、資料相依與 GPU 可行性。
- [x] 建立 [GPU_QAQMC_ACCELERATION_PROPOSAL.md](GPU_QAQMC_ACCELERATION_PROPOSAL.md)。
- [x] 確認 `gpunode02` 硬體：1× A100 PCIe 40 GiB、2× V100 PCIe 32 GiB。
- [x] CUDA build 同時支援 `sm_70` 與 `sm_80`。
- [x] 保留 CPU engine 作為 deterministic/statistical reference。

## Milestone 1：CUDA prefix 與 diagonal update

- [x] packed multiword spin state (`uint64`)。
- [x] tiled CUB prefix-XOR scan。
- [x] 支援 tile、64-bit word 與 `N=384` 邊界。
- [x] grouped-alias tables 上傳與 device lookup。
- [x] Philox RNG，以 `(seed, slice, sweep)` 決定亂數並可重播。
- [x] diagonal rejection resampling。
- [x] 保留原本的 off-diagonal slots。
- [x] CPU/GPU prefix state exact comparison。
- [x] diagonal distribution 與 deterministic replay tests。

## Milestone 2：event lists 與 cluster update

- [x] 建立 site/bond vertex event stream。
- [x] 使用 CUB radix sort 依 `(site, p)` 排序。
- [x] 建立 exact `bond_spin[p]`。
- [x] 維持 CPU site update ordering，保留 Markov transition 相依性。
- [x] 每個 internal segment 使用一個 CUDA block 歸約 log-weight ratio。
- [x] accepted segments 更新 `bond_spin` 與 single-site operator types。
- [x] open-boundary frozen segments 與 worldline closure invariants。
- [x] V100 上完成 repeated MC structural tests。
- [x] cluster 從初版約 `655 ms` 降到約 `12.52 ms`。

## Milestone 3：profile、observables 與 checkpoint

- [x] sparse packed profile-state materialization。
- [-] midpoint/profile density、`Z_l`、`C_m_l`、`A_v` exact comparison。
- [-] VBS/SS observables exact comparison。
- [-] occupation-SF selected states 與 host reducer comparison。
- [x] operator string、seed、sweep id checkpoint format。
- [x] atomic checkpoint replacement。
- [-] checkpoint/restart exact Philox trajectory replay。
- [-] 不同 `N`、tile boundary 與 `profile_step` 的完整 GPU test matrix。

說明：上述 `[-]` 項目的程式與 tests 已寫入 repository；因最新 tests 加入後
無法取得 `gpunode02` allocation，尚不能宣稱最新 build 已通過 real-GPU gate。

## Milestone 4：production runner

- [x] 新增 `src/engines/qaqmc_cuda.py` high-level backend。
- [x] 新增 `main_scripts/python_scripts/run_qaqmc_cuda.py`。
- [x] 新增 runtime probe 與 Slurm scripts。
- [x] HDF5 bin/chunk streaming，不累積全部 raw samples。
- [x] manifest/config hash 與 checkpoint recovery 設計。
- [x] rank-local HDF5 samples/state 同一 transaction、pending transaction recovery，
  並只保留最新 large operator checkpoint；8-chunk bounded-space regression 通過。
- [x] fake CUDA engine scheduler-interruption/resume 與 uninterrupted raw sample stream
  逐元素相同；Hamiltonian/geometry/protocol fingerprint 不符會 fail-fast。
- [-] real string/Rényi CUDA engine HDF5 exact Philox continuation tests 已寫，待真 GPU。
- [ ] production measurement cadence 下的 end-to-end samples/s benchmark。
- [ ] 長跑 memory leak、CUDA error 與 checkpoint soak test。

## Milestone 5：目前最高優先驗證工作

- [blocked] 取得 `gpunode02` 的 Slurm GPU allocation。
- [-] 已提交 validation job `26600`；目前 `PD (Priority)`，Slurm 估計啟動時間
  為 `2026-07-16 22:52:53`（排程時間仍可能變動）。
- [-] 已提交三卡 architecture gate `26606`（`--gres=gpu:3`）；會在 device 0/1/2
  各跑完整 GPU suite 與 string/Rényi probes，確保 A100 及兩張 V100 都被驗到。
  目前 `PD (Priority)`；Slurm 顯示的 2027 placeholder 不具可用排程意義。
- [-] 已提交 production-size benchmark `26605`，依賴 `afterok:26600`；會以
  `N=216, M=2,760,000` 對 string/Rényi 各量 full step、topology、CPU ratio 與
  initial/checkpoint/lazy-workspace VRAM。
- [ ] 在乾淨工作目錄執行完整 `tests/gpu` suite。
- [ ] 確認先前通過的 40 個 real-GPU tests 在最新 build 無 regression。
- [ ] 執行後續新增約 30 個 profile/checkpoint/production tests。
- [ ] 分別在 V100 (`sm_70`) 與 A100 (`sm_80`) 跑 smoke test。
- [ ] 執行 small-`N` deterministic、statistical與 checkpoint gates。
- [ ] 執行 production `N=216, M=2,760,000` full-step regression。
- [ ] 記錄 kernel phase timing、wall time、peak VRAM 與 host RSS。

### 目前 Slurm blocker

2026-07-15 最新查詢 `gpunode02`：

- `CPUAlloc=28 / CPUTot=28`。
- `gpu` partition 設定 `OverSubscribe=NO`。
- 三個 jobs `26277`、`26529`、`26573_18` 分別配置 16、8、4 CPUs。
- Slurm 只替 `26529` 記錄 `TresPerNode=gpu:1`；另外兩個 jobs 未申請 GRES，
  因此帳面上仍有兩張 GPU，但沒有 CPU allocation slot 可合法啟動新 job。

因此目前已知的直接阻塞是沒有可用 CPU allocation slot，而不是已證實三張
GPU 都在滿載。任一 job 釋放至少一個 CPU slot 後，即可重新嘗試一個
CPU + 一張 GPU 的測試 allocation。

## Milestone 6：batched multi-chain GPU

- [x] 將 immutable Hamiltonian/grouped-alias tables 與 per-chain state 分離。
- [x] 設計單一 CUDA process 內的 batch ownership。
- [x] 支援 standard、string-work、Rényi-work 的 `B=2/4/8` independent chains。
- [x] 保持每條 chain 的 Philox stream、checkpoint、sweep/topology id 獨立。
- [x] V100 production-size `B=1/2/4/8` throughput 與 VRAM benchmark。
- [x] V100 throughput sweet spot 為 `B=4`；`B=8` 可容納但已降速。
- [-] A100 與第二張 V100 的三卡 job `26722` 已送出；目前另一
  job 佔有一張 GPU，無法同時 allocation 三卡而 pending。
- [ ] 比較 GPU node chain-steps/s 與 CPU64 chain-steps/s。
- [ ] 比較 effective samples/s，包括 integrated autocorrelation time。
- [x] 實作一個 process／GPU；現有 Slurm 可各卡單獨啟動，三卡
  throughput probe 也已送出。

V100、`N=216`、`M=2,760,000` 實測：

| Engine | B=1 chain-step/s | B=4 chain-step/s | B=4 gain | B=4 VRAM |
| --- | ---: | ---: | ---: | ---: |
| standard | 47.16 | 71.34 | 1.512x | 3058.3 MiB |
| string-work | 49.66 | 70.73 | 1.424x | 3058.3 MiB |
| Rényi-work | 23.25 | 34.81 | 1.497x | 4784.7 MiB |

設計、API、記憶體公式與完整 `B=1/2/4/8` 數字見
[GPU_BATCH_CHAINS.md](GPU_BATCH_CHAINS.md)。

### Batch go/no-go gate

- batched `B>1` 相對 `B=1` 的每 GPU throughput 至少提高 `1.5x`。
- 三張 GPU transition throughput 應達到 CPU64 的 `2.5x` 以上，才值得作為
  主要 production backend 長期維護。
- 同時報告 peak VRAM、host memory、checkpoint/observable/I/O overhead。

## Milestone 7：device-side measurement hardening

- [ ] 將 occupation-SF matrix accumulation 從 compact host states 移到 GPU。
- [ ] 在 GPU 直接累積 bins，只下載完成的 bin 結果。
- [ ] 驗證 measurement cadence 不造成 per-step operator-string transfer。
- [ ] 比較 transition-only 與完整 production samples/s。
- [ ] 100,000 samples 的 checkpoint、HDF5 容量與 I/O 壓力測試。

## Milestone 8：off-diagonal seam / string-work CUDA

- [x] 在既有 `DiagonalEngine` 延伸 seam-aware packed worldline scan，沒有另寫
  一份互相漂移的 diagonal/cluster engine。
- [x] device-resident string sites、physical seam words 與 64-bit active mask。
- [x] `set_seam_mask_consistent()` 在 GPU 修復 fixed-boundary parity；必要時可
  將 diagonal slot repurpose 成目標 site 的 off-diagonal slot。
- [x] sorted event stream 上以 terminal site event 實作左右 half-line move。
- [x] touching-bond interval parallel log-ratio reduction；接受後增量修改
  `bond_spin`、terminal type、seam mask。
- [x] read-only `half_line_proposal()` diagnostic 可逐方向和 CPU proposal 的
  terminal/log physical ratio exact comparison。
- [x] 任一 touching bond 的 old/new weight 非正時整個 proposal invalid，行為
  和 CPU half-line engine 一致，不沿用 cluster zero-crossing cancellation。
- [x] host seam mask 在每次 topology sweep 後同步，work driver 不讀 stale sector。
- [x] `QAQMCStringWorkRydbergCUDA` 重用既有 Jarzynski orchestration/result types。
- [x] trajectory start sector 採 rolling device-to-device checkpoint；首次建立 sector
  後，每條 trajectory 不再執行最多 `|C|` 次的 full-string closure repair scan。
- [x] MPI driver 新增 `--backend cuda`。
- [x] zero-bond closure、tile/64-bit boundary、cluster→topology cache tests 已寫。
- [-] interacting string-work versus ED 與長 trajectory test 待 job `26600`。

限制：目前一條 string 最多 64 sites，原因是 active topology mask 使用一個
`uint64_t`；物理 lattice 仍支援 `N <= 384`。

string-work checkpoint 會 lazy 增加一份 types/sites（`2 * 2M * 4` bytes）與
極小的 seam mask/words；probe 現在分別報告 initial、checkpoint、event workspace
三階段 VRAM，避免只看 transition workspace 而漏算 production reset state。

## Milestone 9：two-replica Rényi-work CUDA

- [x] 兩條 `[2, 2M]` operator strings 與 actual-replica packed scans 常駐 device。
- [x] cut 後依 `channel = replica XOR A_mask[site]` 建立 site/bond events。
- [x] channel-space cluster update，保持 CPU 的 physical-site→channel update order。
- [x] topology proposal 以 cut/terminal packed states 檢查 current/proposed closure。
- [x] 證明並套用 closure theorem：合法 single-site toggle 的兩 replica cut
  occupation 必相同，因此 actual-replica bond occupancy 不變、QAQMC log ratio
  精確為 0，accepted move 只需改 mask，不需 reproject operator string。
- [x] 移除原型的 full actual-prefix 常駐陣列與每 proposal 的 `O(M)` touching-bond
  掃描；topology state 降為 `4 * ceil(N/64)` 個 `uint64_t`。
- [x] topology kernel 直接回傳 `active_count`；每個 lambda step 不再下載完整
  mask。只有明確讀取 `B_mask` diagnostic 時才做 packed mask D2H copy。
- [x] 修正 fully-joined endpoint 的 Python work accounting：`unjoined == 0` 時
  不再求值 `0 * log(0)`；CPU-only fake-device regression 會固定檢查此路徑。
- [x] full-prefix prototype kernels 已退出 production compile path，避免額外 NVCC
  template instantiation；compact-boundary regression 通過後可完全刪除參考碼。
- [x] operator strings 的 save/restore 為 device-to-device copy，trajectory reset
  不經 PCIe 搬回 `4M` 個整數。
- [x] 新增 read-only topology log-ratio diagnostic，可逐 configuration 和 CPU
  `log_weight_ratio_for_toggle()` 對照 detailed balance。
- [x] `QAQMCRenyiWorkRydbergCUDA`、MPI `--backend cuda`、warm config export/import。
- [x] exact channel event、bond-spin、cluster、checkpoint、endpoint accounting tests 已寫。
- [x] CPU exhaustive closure-theorem gate：枚舉全部 `M=2` 雙 replica flip strings，
  並另驗證 100 組 interacting bond-operator configurations；所有合法 toggle 均
  ratio=0 且 CPU reproject 不改 operator types。
- [x] 500 組隨機雙 replica paths（含 endpoint/interior cuts）逐 site 比較 compact
  boundary formula 與 brute-force channel propagation，terminal closure 完全一致。
- [x] warm-config operator types/sites 在 H2D upload 前驗證，非法 site/bond/type
  直接丟例外，不讓損壞 checkpoint 變成 CUDA out-of-bounds。
- [x] constructor 的初始雙 replica operator strings 也套用同一套 host validation，
  不只保護後續 warm-config setter。
- [x] 多 trajectory 結果直接寫入固定 dtype NumPy buffers，不再先累積 Python
  dataclass objects 再轉陣列。
- [-] interacting closure-theorem ratio/mask-toggle exact test 待 job `26600`。
- [-] small-N Rényi work versus ED 與 asymmetric-cut statistical gate 待 real GPU。

### Rényi 額外記憶體

- compact actual-replica boundaries（兩 replica × cut/terminal）：
  `4 * ceil(N/64) * 8` bytes。原型曾使用
  `2 * (2M + 1) * ceil(N/64) * 8` bytes full prefix，現已移除。
- operator state：types/sites 共 `2 replicas * 2 arrays * 2M * 4` bytes。
- D2D checkpoint 再增加同樣大小的 types/sites state，但只在首次 checkpoint
  後配置。
- 每條 trajectory 必須保留的 host 結果為 `float64 + 2*int32 + 2*int64 = 32`
  bytes；100,000 條 raw arrays 約 `3.05 MiB`，沒有額外 Python-object list。
- event buffers 以最壞容量配置：site `2*(2M)`、bond `4*(2M)` events 加 CUB
  sort workspace；實際 MiB 由 `device_bytes` 在 lazy allocation 前後直接回報。
- batch-chain 中 immutable Hamiltonian/grouped-alias tables 已拆成 shared model
  data；Rényi 的雙 replica mutable state 仍是每條 chain 各自保有。

### 高頻同步巡檢

- work trajectory 內沒有 operator-string PCIe round trip；Rényi topology 現在每
  sweep 只回傳固定大小 stats（含 `active_count`）。
- off-diagonal topology 仍同步 8-byte seam mask，因上層需要 exact bit pattern
  驗證 forward/reverse sector；這不是隨 `M` 或 `N` 成長的傳輸。
- diagonal/event/cluster 目前各自同步 timing/stats；cluster 另下載 `O(N)` 的
  event heads/counts 以維持 CPU 的逐 site update order。這些屬 latency 項而非
  大 `M` 記憶體項，須等 V100/A100 phase timing 後再決定是否改 persistent event
  或 device scheduler，避免在沒有 profile 證據前改變 transition ordering。

## Milestone 10：work production launch / benchmark

- [x] `run_kagome_string_work_cuda.sh`：三 rank／三 GPU production launcher。
- [x] `run_kagome_renyi_work_cuda.sh`：三 rank／三 GPU production launcher。
- [x] MPI node-local GPU affinity，同時支援 per-task visibility 與全卡 visibility。
- [x] MPI drivers 依 backend lazy import：CUDA 路徑先解析
  `build_cuda/qaqmc_cpp`，CPU 路徑仍維持 generic `build/`；benchmark probe
  固定使用前者。已從 `/tmp` 驗證兩組路徑與三個 production entry points。
- [x] `probe_qaqmc_work_cuda.py` 同時報告 CPU/GPU full-step、topology 與 lazy VRAM。
- [x] `test_qaqmc_work_cuda.sh` 串接完整 `tests/gpu` 與兩個 probe。
- [x] CUDA-only `--resume`：每個 committed raw-sample chunk 與 rolling operator
  state、site permutation、`sweep_id/topology_id` 一起發布；resume 前驗證完整
  Hamiltonian、geometry、rank count、schedule 與 transition cadence。
- [x] continuation operator types 以 `int8`、site/bond index 以最小可容納的
  `uint8/uint16/uint32` lossless 儲存；`M=2.76e6` raw staging string 約
  `42.1 → 15.8 MiB`，Rényi 約 `84.2 → 31.6 MiB`（尚未計 gzip）。
- [x] production launchers 支援 `RESUME=1`，且必須明確指定上一個 `RUN_TAG`、
  `FILEPATH` 或 `CKPT_DIR`；fresh run 不會默默覆寫既有 chunks。
- [-] V100/A100 數字尚待 validation allocation；不可沿用 standard engine 的
  `46.8x` 宣稱兩個 work engines 也有同樣 speedup。

## 本輪新增功能的驗收原則

off-diagonal seam、string-work、Rényi channel remapping 與 Rényi work protocol
已不再是「明確延後」項目。它們仍必須各自通過 detailed-balance、exact
reference、checkpoint 與 ED tests；在 job `26600` 完成以前狀態維持 `[-]`，
不能因為 CUDA build 成功就宣稱 production-ready。

## 測試與執行注意事項

- repository 根目錄目前有一個以其他 CPU ISA 編譯的舊 `qaqmc_cpp*.so`；在
  `gpunode02` 從 repo root 啟動 Python 可能被目前目錄優先載入並造成
  `Illegal instruction`。
- GPU tests 應從 `/tmp` 啟動，並明確設定：

  ```bash
  PYTHONPATH=/home/tohenry20109/spin_lake_project/build_cuda:\
  /home/tohenry20109/spin_lake_project \
  /home/tohenry20109/miniconda3/envs/qaqmc/bin/python -m pytest -q \
  /home/tohenry20109/spin_lake_project/tests/gpu
  ```

- `build_cuda/` 是 generated build tree，不應加入版本控制。
- CPU regression tests 必須在每次 CUDA/C++ interface 變更後重跑。

## 完成定義

standard GPU backend 只有在以下條件全部成立後才算完成：

- [ ] 最新完整 GPU unit/integration suite 在 V100 通過。
- [ ] A100 smoke/performance test 通過。
- [ ] production checkpoint/resume 與 HDF5 recovery 通過。
- [ ] production `M` 下無 per-step PCIe operator-string round trip。
- [x] V100 batch-chain throughput 與 VRAM benchmark 完成；A100 三卡 gate 待
  `gpunode02` 恢復 allocation。
- [ ] off-diagonal interacting exact/ED gates 在 V100 與 A100 通過。
- [ ] Rényi interacting closure/ratio/mask-toggle、checkpoint 與 ED gates通過。
- [x] CPU reference regressions 全數通過（最新廣泛 engine/MPI gate 105 passed，
  另有 4 支直接 reference scripts）。
- [ ] 文件中的效能數字可由 repository scripts 重現。
