# Spin Lake Project — QAQMC

部署與執行說明。環境完全由 conda-forge 在 user space 提供（含 MPI、OpenMP、編譯器），目標機器**只需要 miniconda，不需要 root、不需要系統 MPI**。

> 設計文件、引擎規格、實驗日誌集中在 [`docs/`](docs/INDEX.md)
> （design / specs / progress 三類，入口是 `docs/INDEX.md`）。

## 部署到新 server

```bash
# 0. 若機器沒有 miniconda（裝在 $HOME，免 root）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# 1. 取得專案
git clone <repo-url> spin_lake_project
cd spin_lake_project

# 2. 建立環境（含 openmpi/mpiexec、mpi4py、g++/OpenMP、cmake/ninja/pybind11）
conda env create -f environment.yml          # 一般安裝（最新相容版本）
# conda env create -f environment.lock.yml   # 或：精確重現通過驗證的環境
conda activate qaqmc

# 3. 編譯 C++ 核心（預設 -march=native，在哪台跑就在哪台 build）
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DPYTHON_EXECUTABLE=$(which python)
cmake --build build -j
cp build/qaqmc_cpp*.so .

# 4. 驗證
python -c "import qaqmc_cpp; print('C++ extension OK')"
```

注意事項：

- **一顆 binary 要在多台不同 CPU 的機器共用**時，改用
  `-DQAQMC_ARCH=x86-64-v3`（AVX2 baseline，2015 後的 Intel/AMD 都能跑）。
  預設的 `native` 只保證在 build 的那台機器上執行。
- **絕對不要在任務執行中 `cp` 覆蓋活著的 `.so`** — 會讓執行中的行程
  segfault。要部署新版請用 `mv`（atomic rename），或等任務結束。
- 換 `-march` 會改變浮點捨入（FMA contraction），同 seed 的軌跡在不同
  build 間不會 bit-identical（統計上等價）。
- **`environment.yml` 沒有鎖版本**（conda-forge 持續更新）。每次成功部署並
  通過測試後，重新產生 known-good 快照：
  `conda env export -n qaqmc | grep -v '^prefix:' > environment.lock.yml`
  （保留檔頭註解）。歷史快照就在這個檔案的 git 歷史裡。
- ⚠️ **僅支援單節點**：conda-forge 的 Open MPI 沒有 InfiniBand/UCX 與
  SLURM PMIx 整合 — 跨節點 job 要嘛起不來、要嘛默默走 TCP。多節點需要
  系統 MPI 並對其重編 mpi4py（目前 out of scope，不要直接嘗試）。

## Standard QAQMC CPU 記憶體

CPU native backend 已集中在 `csrc/cpu/{include,detail,bindings}`，CUDA backend
在 `csrc/cuda/{include,src,detail,bindings}`；兩個 backend 用同一套三層慣例
（public headers / 實作 / pybind registrations）。舊 `csrc/*.hpp` root shims
已在 backend merge 完成後移除。目錄責任與 CPU/CUDA merge contract 見
[`csrc/cpu/README.md`](csrc/cpu/README.md)、[`csrc/cuda/README.md`](csrc/cuda/README.md)。

`QAQMCEngine` 對 `N=216` full-bond production 會自動使用 lossless 16-bit
alias/operator indices、`int8` operator types、按需 delta schedule，以及有界
event-scratch capacity。系統超過 16-bit index 範圍時會自動 fallback 到 32-bit，
不會截斷 bond/site index。Python wrapper 不再常駐一份完整 int32 operator mirror；
`engine.op_types`／`engine.op_sites` 仍可使用，但會在存 checkpoint 等真正需要時
才匯出。

可用 fresh process 量 init/steady-state RSS、vector capacity 與 phase timing：

```bash
cd /tmp
PYTHONPATH=/path/to/spin_lake_project/build:/path/to/spin_lake_project \
python -m src.probes.qaqmc_cpu_memory \
    --M 2760000 --warmup-steps 2 --timed-steps 5
```

`--export-checksum` 會額外建立相容的 int32 operator arrays，因此 memory gate
預設在 export 前取樣。寫 HDF5 時原有 `delta_schedule` schema 保持不變，
但改用有界 chunk 產生，不會在寫檔前突然重建完整 `2M` array。

同一個 Hamiltonian 的多條 chains 可使用 process-local shared model，避免每個
rank 重複約 241 MiB proposal/geometry tables：

```python
from src.engines.qaqmc_cpu_batch import QAQMCSharedModelBatch

with QAQMCSharedModelBatch(batch_size=8, seed=42, **engine_kwargs) as batch:
    batch.run_steps(100)
```

C++ transition 會釋放 GIL，batch 內一個 worker 對應一條獨立 chain。多 socket
production 建議每個 NUMA socket 一個 MPI rank，再於 rank 內建立多條 chains。
raw API 預設 event layout 是 `packed64`。目前 N=216 production 建議設定
`bond_event_storage="p_bond16"`（6 bytes/event，最多 65,535 bonds）；實測大 M
省 15–18% RSS且速度持平。只有 RAM 仍不足時才用 `"p_only32"`
（4 bytes/event、慢約 30%）。MPI CLI 同樣接受
`--bond_event_storage`，並把選擇寫進 HDF5 params attributes。

設計、公式與 A/B gates 見 `docs/design/cpu_memory.md`。

## CUDA QAQMC backend（gpu_version）

CUDA 是 optional build；CPU-only build 不需要 CUDA toolkit。gpunode02 的
A100 (`sm_80`) 與兩張 V100 (`sm_70`) 共用同一顆 extension：

```bash
conda activate qaqmc
cmake -S . -B build_cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DQAQMC_ENABLE_CUDA=ON \
      -DQAQMC_CUDA_ARCHITECTURES='70;80' \
      -DQAQMC_ARCH=x86-64 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
      -DPYTHON_EXECUTABLE=$(which python)
cmake --build build_cuda -j
```

`QAQMC_ARCH=x86-64` 讓同一顆 CPU reference module 可在 login/GPU node
之間載入；production CPU benchmark 若在目標 node 原地 build，仍可用
`native`。請不要把 login node 的 native `qaqmc_cpp*.so` 放在目前工作目錄
再到舊 CPU node 執行，因為 Python 的空 import path 會優先於
`PYTHONPATH`，可能直接 `Illegal instruction`。

**換機器部署 CUDA 的需求清單**：
- 系統 CUDA toolkit ≥ 12（不在 conda env 裡）；nvcc 路徑用
  `-DCMAKE_CUDA_COMPILER=<path>/bin/nvcc` 指定。
- NVIDIA driver ≥ R525（支援 CUDA 12 runtime）。
- GPU compute capability ≥ 7.0（Volta）；更新的卡走 PTX JIT 也能跑，
  更舊的卡 `qaqmc_cuda.is_available()` 會回 False（不會噴 kernel error）。
- **換機必重編**：`.so` 的 RPATH 烙著 build 機的 CUDA 路徑
  （`readelf -d build_cuda/qaqmc_cuda*.so | grep RPATH` 可驗）。

GPU 測試必須在 compute allocation 中執行：

```bash
srun --partition=gpu --nodelist=gpunode02 --gres=gpu:1 --cpus-per-task=2 \
  bash -lc 'cd /tmp && \
  PYTHONPATH=/path/to/spin_lake_project/build_cuda:/path/to/spin_lake_project \
  python -m pytest -q /path/to/spin_lake_project/tests/gpu'
```

單 GPU production job：

```bash
sbatch scripts/run/cuda/run_kagome_qaqmc_cuda.sh

# 多條獨立 chain 可用 job array；給所有 task 同一個絕對 RUN_DIR，
# SLURM_ARRAY_TASK_ID 會成為 rank/seed offset。
RUN_DIR=$PWD/data/qaqmc_cuda_ensemble \
  sbatch --array=0-2 scripts/run/cuda/run_kagome_qaqmc_cuda.sh
```

CUDA runner 目前輸出 rank-local batched profile：density、`Z_l`、`C_m_l`、
`A_v`、VBS/SS、指定 δ 點的 occupation-SF matrices，以及可精確 replay 的
operator/Philox checkpoint。SF 的 O(2M) worldline propagation 在 GPU；只把
選定的 `n_delta × N` packed states 傳回 host 做小矩陣歸約。standard
single-replica QAQMC 已 GPU 化；Rényi/work 與 off-diagonal seam/string 仍是
獨立後續 backend，不能用這個 runner 代替。

## 提交 / 執行腳本

production 腳本在 `scripts/run/`，同一份腳本在有無 SLURM 的
機器上都能跑（`#SBATCH` 標頭在直接執行時只是註解）：

```bash
# 統一入口：有 sbatch 就提交 job，沒有就 nohup 背景執行（log 寫到 logs/）
./scripts/submit.sh scripts/run/cpu/run_kagome_sse.sh

# 額外參數會透傳給 sbatch（覆寫 #SBATCH 標頭）
./scripts/submit.sh scripts/run/cpu/run_kagome_otf.sh --nodelist=cpunode02

# 也可以照舊直接用
sbatch scripts/run/cpu/run_kagome_otf.sh       # SLURM cluster
bash   scripts/run/cpu/run_kagome_otf.sh       # 一般 server（前景）
```

所有腳本參數都用環境變數覆寫，例如：

```bash
NX=8 NY=8 M=200000 N_TRAJ=8000 \
    ./scripts/submit.sh scripts/run/cpu/run_kagome_renyi_work.sh
```

資源與綁核（由 `scripts/common/env.sh` 統一處理）：

- **SLURM job 內**：ranks / cores-per-rank 自動取自 `SLURM_NTASKS` /
  `SLURM_CPUS_PER_TASK`。
- **無 SLURM**：預設 ranks = 實體核心數（排除超執行緒）、每 rank 1 核；
  用 `NTASKS=16 CPT=4` 這類環境變數覆寫。
- launcher 兩種情況都是 `mpiexec`（conda 的 Open MPI 沒有 SLURM/PMIx 整合，
  不要用 srun）。
- **site 專屬設定**（綁核策略、conda 路徑、scheduler 目標）：
  `cp scripts/common/site.conf.example scripts/common/site.conf`
  後編輯（此檔已 gitignore）。例如 AMD EPYC 建議
  `BIND_FLAGS="--map-by numa:PE=$CPT --bind-to core"`；container 內
  `BIND_FLAGS="--bind-to none"`。
- **換 cluster**：run 腳本 `#SBATCH` 標頭裡的 partition/nodelist 是
  本站預設；在 site.conf（或環境變數）設 `SBATCH_PARTITION` /
  `SBATCH_NODELIST`，`submit.sh` 會以 sbatch 參數注入覆寫標頭 —
  不用逐檔改 16 份腳本。

四個 production 腳本（輸出資料夾/檔名都帶引擎標籤）：

| 腳本 | 內容 | 輸出 |
| --- | --- | --- |
| `run_kagome_otf.sh` | 單 replica QAQMC diagonal-profile（`src.mpi.qaqmc_mpi --mode profile`） | `data/qaqmc_profile_M=..._<stamp>/` |
| `run_kagome_sse.sh` | 有限溫度 SSE 熱平衡對照（`src.mpi.sse_mpi`） | `data/sse_.../` |
| `run_kagome_renyi_work.sh` | Renyi-2 非平衡功引擎，KP 區域 ΔS₂ / γ（`src.mpi.qaqmc_renyi_work_mpi`） | `data/renyi_work_....h5` (+`_chunks/`) |
| `run_kagome_string_work.sh` | off-diagonal string-work Jarzynski 估計（`src.mpi.qaqmc_string_work_mpi`） | `data/string_work_....h5` (+`_chunks/`) |

## 畫圖

所有繪圖腳本從 repo 根目錄執行，圖預設存到 **`figures/<資料目錄同名>/<圖名>.png`**
（可用 `--out` 覆寫）。各腳本都有 `--help` 列出完整參數。

**Diagonal profile**（吃 `data/qaqmc_profile_M=*` chunked run dir；不給 `--run_dir` 自動用最新的）：

```bash
python plots/plot_diagonal/plot_profile_panels.py --run_dir data/qaqmc_profile_M=...   # density/A_v/Z_l/C_m 四合一
python plots/plot_diagonal/plot_vbs_ss.py         --run_dir ...                        # Ψ_VBS / Ψ_SS vs δ（上下坡）
python plots/plot_diagonal/plot_bffm.py           --run_dir ...                        # BFFM: C_m(l−1)/√|Z(l)|
python plots/plot_diagonal/plot_occ_sf_bz.py      --run_dir ...  # S_αβ(q) BZ heatmap；--mode connected|unconnected --stat max_eig|trace
python plots/plot_diagonal/plot_snapshots.py      --run_dir ...  # 實空間激發圖案（各 δ 點）
python plots/plot_diagonal/plot_m_domains.py      --run_dir ... --delta 5.5  # per-bin M-domain 分析
                                       # （取向分類/互斥檢驗；也吃 SSE run dir——domain 平均下看秩序用這個）
```

**SSE**（吃 `data/sse_*` run dir；單參數點）：

```bash
python plots/plot_sse/plot_observables.py --run_dir data/sse_...   # Z_l/A_v、C_m、序參量總覽（標題=E）
python plots/plot_sse/plot_occ_sf_bz.py   --run_dir ...            # S_αβ(q) BZ heatmap
python plots/plot_sse/plot_snapshots.py   --run_dir ...            # 激發圖案網格
```

**Work engines 誤差診斷**（blocking analysis：橫軸 bin size、縱軸 error、標題=估計值；
`--data` 給 chunk 目錄（`*_chunks/`，per-rank 時序完整，優先）或 aggregate `.h5`）：

```bash
python plots/plot_renyi_work/plot_error_vs_binsize.py   --data data/renyi_work_..._chunks    # ΔS₂（jackknife）
python plots/plot_off_diagonal/plot_error_vs_binsize.py --data data/string_work_..._chunks   # O_C（SEM+jackknife 對照）
```
