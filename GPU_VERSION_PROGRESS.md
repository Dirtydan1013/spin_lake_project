# QAQMC GPU Version Progress

最後更新：2026-07-14  
分支：`gpu_version`  
基準分支：`z2_spin_lake` (`31d6c5c`)  
目前狀態：standard single-replica QAQMC CUDA engine 已完成核心實作；production hardening 與 batch-of-chains 尚待真 GPU 驗證。

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
- 現行執行模型仍為一個 process／一條 chain／一張 GPU；尚未做 batched chains。

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
- [-] rank-local HDF5 output + interruption/resume integration test（需真 GPU）。
- [ ] production measurement cadence 下的 end-to-end samples/s benchmark。
- [ ] 長跑 memory leak、CUDA error 與 checkpoint soak test。

## Milestone 5：目前最高優先驗證工作

- [blocked] 取得 `gpunode02` 的 Slurm GPU allocation。
- [ ] 在乾淨工作目錄執行完整 `tests/gpu` suite。
- [ ] 確認先前通過的 40 個 real-GPU tests 在最新 build 無 regression。
- [ ] 執行後續新增約 30 個 profile/checkpoint/production tests。
- [ ] 分別在 V100 (`sm_70`) 與 A100 (`sm_80`) 跑 smoke test。
- [ ] 執行 small-`N` deterministic、statistical與 checkpoint gates。
- [ ] 執行 production `N=216, M=2,760,000` full-step regression。
- [ ] 記錄 kernel phase timing、wall time、peak VRAM 與 host RSS。

### 目前 Slurm blocker

2026-07-14 查詢 `gpunode02`：

- `CPUAlloc=28 / CPUTot=28`。
- `gpu` partition 設定 `OverSubscribe=NO`。
- 三個 jobs 分別配置 16、8、4 CPUs。
- Slurm 只明確顯示其中一個 job 申請 `gpu:1`；其他 jobs 是否直接使用 GPU
  無法從 GRES 狀態確認。

因此目前已知的直接阻塞是沒有可用 CPU allocation slot，而不是已證實三張
GPU 都在滿載。任一 job 釋放至少一個 CPU slot 後，即可重新嘗試一個
CPU + 一張 GPU 的測試 allocation。

## Milestone 6：batched multi-chain GPU

這是下一個決定 GPU node 是否真正勝過 CPU64 的關鍵階段。

- [ ] 將 immutable Hamiltonian/grouped-alias tables 與 per-chain state 分離。
- [ ] 設計單一 CUDA process 內的 batch dimension。
- [ ] 支援每張 GPU 同時推進 `B=2/4/8` independent chains。
- [ ] 保持每條 chain 的 Philox stream、checkpoint 與 sweep id 獨立。
- [ ] benchmark V100、A100 的 `B=1/2/4/8` throughput。
- [ ] 找出 throughput sweet spot，而不是只追求 VRAM 可容納的最大 chain 數。
- [ ] 比較 GPU node chain-steps/s 與 CPU64 chain-steps/s。
- [ ] 比較 effective samples/s，包括 integrated autocorrelation time。
- [ ] 實作一個 process／GPU，再以 Slurm job array 或 MPI 使用三張 GPU。

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

## 明確延後的功能

以下功能不屬於 standard single-replica GPU engine 的目前驗收範圍：

- [ ] off-diagonal seam / `X_C` string estimator。
- [ ] string-work trajectories。
- [ ] Rényi replicas、channel remapping 與 topology update。
- [ ] Rényi work protocol。

這些功能必須各自建立 detailed-balance、exact reference 與 checkpoint tests，
不能直接假設 standard kernel 可無條件重用。

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
- [ ] batch-chain throughput 與 VRAM benchmark 完成。
- [ ] CPU reference regressions 全數通過。
- [ ] 文件中的效能數字可由 repository scripts 重現。
