# Spin Lake Project — QAQMC

部署與執行說明。環境完全由 conda-forge 在 user space 提供（含 MPI、OpenMP、編譯器），目標機器**只需要 miniconda，不需要 root、不需要系統 MPI**。

## 部署到新 server

```bash
# 0. 若機器沒有 miniconda（裝在 $HOME，免 root）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# 1. 取得專案
git clone <repo-url> spin_lake_project
cd spin_lake_project

# 2. 建立環境（含 openmpi/mpiexec、mpi4py、g++/OpenMP、cmake/ninja/pybind11）
conda env create -f environment.yml
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

## 提交 / 執行腳本

production 腳本在 `main_scripts/slurm_scripts/`，同一份腳本在有無 SLURM 的
機器上都能跑（`#SBATCH` 標頭在直接執行時只是註解）：

```bash
# 統一入口：有 sbatch 就提交 job，沒有就 nohup 背景執行（log 寫到 logs/）
./main_scripts/submit.sh main_scripts/slurm_scripts/run_kagome_sse.sh

# 額外參數會透傳給 sbatch（覆寫 #SBATCH 標頭）
./main_scripts/submit.sh main_scripts/slurm_scripts/run_kagome_otf.sh --nodelist=cpunode02

# 也可以照舊直接用
sbatch main_scripts/slurm_scripts/run_kagome_otf.sh       # SLURM cluster
bash   main_scripts/slurm_scripts/run_kagome_otf.sh       # 一般 server（前景）
```

所有腳本參數都用環境變數覆寫，例如：

```bash
NX=8 NY=8 M=200000 N_TRAJ=8000 \
    ./main_scripts/submit.sh main_scripts/slurm_scripts/run_kagome_renyi_work.sh
```

資源與綁核（由 `main_scripts/common/env.sh` 統一處理）：

- **SLURM job 內**：ranks / cores-per-rank 自動取自 `SLURM_NTASKS` /
  `SLURM_CPUS_PER_TASK`。
- **無 SLURM**：預設 ranks = 實體核心數（排除超執行緒）、每 rank 1 核；
  用 `NTASKS=16 CPT=4` 這類環境變數覆寫。
- launcher 兩種情況都是 `mpiexec`（conda 的 Open MPI 沒有 SLURM/PMIx 整合，
  不要用 srun）。
- **site 專屬設定**（綁核策略、conda 路徑等）：
  `cp main_scripts/common/site.conf.example main_scripts/common/site.conf`
  後編輯（此檔已 gitignore）。例如 AMD EPYC 建議
  `BIND_FLAGS="--map-by numa:PE=$CPT --bind-to core"`；container 內
  `BIND_FLAGS="--bind-to none"`。

四個 production 腳本：

| 腳本 | 內容 |
| --- | --- |
| `run_kagome_otf.sh` | 單 replica QAQMC diagonal-profile（`src.mpi.qaqmc_mpi --mode profile`） |
| `run_kagome_sse.sh` | 有限溫度 SSE 熱平衡對照（`src.mpi.sse_mpi`） |
| `run_kagome_renyi_work.sh` | Renyi-2 非平衡功引擎，KP 區域 ΔS₂ / γ（`src.mpi.qaqmc_renyi_work_mpi`） |
| `run_kagome_string_work.sh` | off-diagonal string-work Jarzynski 估計（`src.mpi.qaqmc_string_work_mpi`） |
