# Spin Lake Project: Quantum Annealing Monte Carlo (QAQMC)

這是一個以 Python 與 C++ 混合開發的量子退火蒙地卡羅 (Quantum Annealing Quantum Monte Carlo, QAQMC) 模擬框架。本專案透過 **C++ (搭配 OpenMP 平行運算) 處理核心的計算密集型任務**，並透過 **Python / Pybind11 提供高階 API 與資料分析**。

---

## 📂 專案架構 (Project Structure)

專案分為高階邏輯 (Python) 與底層引擎 (C++) 兩部分：

```text
spin_lake_project/
├── csrc/                           # C++ 核心引擎 (底層計算)
│   ├── qaqmc_core.hpp / .cpp       # QAQMC 的核心算法 (Alias Table, Diagonal/Cluster Update)
│   └── bindings.cpp                # Pybind11 的 Python 綁定介面
├── src/                            # Python 高階 API (主程式)
│   ├── qaqmc.py                    # QAQMC_Rydberg 類別 (自動偵測並調用 C++ 引擎)
│   ├── hamiltonian.py, lattices.py # 模型參數與晶格生成
│   ├── sse.py, sse_updates.py      # 標準 SSE (Stochastic Series Expansion) 實現
│   └── measurement.py, postprocess.py # 測量與後處理工具
├── CMakeLists.txt                  # C++ / Pybind11 的 CMake 建置設定
├── build.bat                       # Windows 一鍵編譯腳本
├── test.py                         # 快速端到端 (End-to-End) 測試與驗證腳本
└── test_cpp_vs_python.py           # 用於驗證 C++ 引擎與純 Python/Numba 引擎是否一致的腳本
```

---

## 🐳 安裝與執行 (Installation & Run — via Docker)

本專案推薦使用 **Docker** 來建立環境與執行模擬。你只需要安裝好 Docker，無需準備 Python 虛擬環境、C++ 編譯器或任何系統依賴。

---

### 1. 取得專案

```bash
git clone https://github.com/你的帳號/spin_lake_project.git
cd spin_lake_project
```

---

### 2. 建立 Docker Image（只需做一次）

```bash
docker build -t qaqmc_app .
```

Docker 會自動完成以下所有步驟：
- 下載 Python 3.12 基底環境
- 安裝 `g++`、`cmake`、`ninja` 及 OpenMP 函式庫
- 編譯 C++ 核心引擎（`qaqmc_cpp.so`）
- 安裝所有 Python 套件（`numpy`, `scipy`, `h5py`, `numba` 等）

---

### 3. 執行模擬

#### 前景執行（適合快速測試）

```bash
mkdir -p data
docker run -v $(pwd)/data:/app/data qaqmc_app python test.py
```

#### 背景執行（適合長時間運算，SSH 斷線後不中斷）

```bash
mkdir -p data
docker run -d \
    -v $(pwd)/data:/app/data \
    --name qaqmc_run \
    qaqmc_app python test.py
```

執行後，模擬結果（`.h5` 數據與圖片）會直接存入當前目錄的 `data/` 資料夾裡。

```bash
# 隨時查看計算進度
docker logs -f qaqmc_run

# 確認是否還在運行
docker ps
```

---

### 4. 在自己的腳本中呼叫

```python
from src.qaqmc import QAQMC_Rydberg

# omp_threads: 控制每個 Process 使用的 OpenMP 核心數（建議設為 1）
# n_jobs: 在 run_and_save 裡控制平行的馬可夫鏈條數
qmc = QAQMC_Rydberg(N=6, M=160, Omega=1.0, omp_threads=1)


qmc.run_and_save("data/my_run.h5", n_equil=4000, n_samples=30000, n_jobs=4)
```

---

## 📦 在 HPC Cluster 上執行（Singularity）

若 Cluster 使用 **Singularity**（大多數 HPC 環境皆支援），可將整個環境打包成 `.sif` 映像檔。

### 1. 建置映像檔（需在有 root 或 `--fakeroot` 權限的機器上）

```bash
git clone https://github.com/你的帳號/spin_lake_project.git
cd spin_lake_project

# 從 singularity.def 建置（推薦）
singularity build spin_lake.sif singularity.def

# 或者直接從現有的 Dockerfile 轉換
singularity build spin_lake.sif docker-daemon://qaqmc_app:latest
```

> 若 Cluster 不允許 `--fakeroot`，請在本地 Linux 機器（有 root 權限）上 build 完，再把 `spin_lake.sif` 傳到 Cluster。

### 2. 執行模擬

```bash
# 預設執行 test.py（資料輸出到 ./data/）
singularity run --bind ./data:/app/data spin_lake.sif

# 執行自訂腳本
singularity exec --bind ./data:/app/data spin_lake.sif \
    python /app/scripts/my_script.py

# 進入互動式 shell（除錯用）
singularity shell --bind ./data:/app/data spin_lake.sif
```

### 3. 搭配 SLURM 批次提交

```bash
#!/bin/bash
#SBATCH --job-name=qaqmc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=qaqmc_%j.log

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

singularity exec --bind ./data:/app/data spin_lake.sif \
    python /app/test.py
```

將上面內容存為 `submit.sh`，然後：

```bash
mkdir -p data
sbatch submit.sh
```

### 4. 驗證環境是否正常

```bash
# 確認 C++ extension 可載入
singularity exec spin_lake.sif \
    python -c "import qaqmc_cpp; print('C++ extension OK')"

# 確認 Python imports
singularity exec spin_lake.sif \
    python -c "from src.qaqmc import QAQMC_Rydberg; print('Python imports OK')"
```

---

## 🐍 替代方案：使用 Conda 安裝（無 Docker 環境適用）

若伺服器沒有安裝 Docker，可改用 Conda 建立隔離的 Python 環境並手動編譯 C++ 核心。

### 1. 建立 Conda 環境

```bash
conda create -n qaqmc python=3.12
conda activate qaqmc
```

### 2. 安裝 Python 套件與建置工具

```bash
pip install -r requirements.txt

# 確保 C++ 編譯相關的系統工具可用
conda install -c conda-forge cmake ninja openmp
```

> 若伺服器已有系統級的 `g++` 與 `libomp`（例如 Ubuntu 上的 `build-essential` 和 `libomp-dev`），可跳過 conda 的編譯工具安裝，改用系統版本。

### 3. 編譯 C++ 核心

```bash
cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=$(which python)

cmake --build build -j

cp build/qaqmc_cpp*.so .
```

### 4. 執行模擬

```bash
# 前景執行（測試用）
python test.py

# 背景執行（SSH 斷線後不中斷）
nohup python test.py > output.log 2>&1 &
echo "PID: $!"          # 記下 PID，之後可用 kill <PID> 停止

# 或使用 tmux（若伺服器有安裝）
tmux new -s qaqmc
python test.py
# Ctrl+B → D 放到背景；之後 tmux attach -t qaqmc 回來查看
```
