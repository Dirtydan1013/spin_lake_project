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

## Standard QAQMC CPU 記憶體

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
python /path/to/spin_lake_project/main_scripts/python_scripts/probe_qaqmc_cpu_memory.py \
    --M 2760000 --warmup-steps 2 --timed-steps 5
```

`--export-checksum` 會額外建立相容的 int32 operator arrays，因此 memory gate
預設在 export 前取樣。寫 HDF5 時原有 `delta_schedule` schema 保持不變，
但改用有界 chunk 產生，不會在寫檔前突然重建完整 `2M` array。
設計、公式與 A/B gates 見 `CPU_MEMORY.md`。

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
