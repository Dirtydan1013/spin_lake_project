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

## 🛠️ 安裝與建置環境 (Installation & Build)

要在本機上運行此專案，你需要準備 Python 環境以及 C++ 編譯器。本專案的建置系統基於 **CMake** 與 **Ninja**，並強烈依賴 **OpenMP** 來達成多核心平行加速。

### 1. 準備 Python 依賴環境
請確保你的 Python 版本為 3.8 以上 (推薦使用虛擬環境 `.venv`)，並安裝以下套件：

```bash
pip install numpy scipy numba h5py matplotlib tqdm 
pip install cmake ninja pybind11
```
*(註：`cmake` 與 `ninja` 可直接透過 pip 安裝，不需污染系統環境)*

### 2. 準備 C++ 編譯器 (Windows)
因為本專案的 Windows 環境預設使用 MinGW GCC (為了更好地支援 OpenMP)，請 **不要** 使用 Visual Studio 的 MSVC。
1. 下載並安裝 [MSYS2](https://www.msys2.org/)。
2. 開啟 MSYS2 終端機，安裝 UCRT64 版本的 GCC：
   ```bash
   pacman -S mingw-w64-ucrt-x86_64-gcc
   ```
3. **重要：** 將 GCC 的路徑 (預設為 `C:\msys64\ucrt64\bin` 或 `D:\msys64\ucrt64\bin`) 加入到 Windows 系統的環境變數 `PATH` 中。

*(註：如果你是 Linux / macOS 使用者，只要確保系統裡有 `g++` 或 `clang++` 以及 `libomp` 即可)*

### 3. 一鍵編譯 C++ 核心 (.pyd)

在專案根目錄下 (且已啟動 Python 虛擬環境)，執行：

```bash
cmd /c build.bat
```

這支腳本會自動完成：
1. 呼叫 CMake 進行環境設定 (`-G Ninja`)。
2. 呼叫 Ninja 且使用多執行緒高速編譯 C++ 原始碼。
3. 把編譯好的 `.pyd` 擴充模組搬移到專案根目錄。

如果看到 `Build complete!` 且偵測到 OpenMP 多執行緒數量，代表編譯大功告成。

---

## 🚀 測試與使用 (Usage)

編譯完成後，就可以直接當作普通的 Python 套件來使用。

### 執行測試
驗證 QAQMC 結合 C++ 引擎是否正確運作 (會自動對比 Exact Diagonalization 結果)：
```bash
python test.py
```
*(你應該會看到終端機印出 `[QAQMC] Using C++ backend...` 以及進度條，並在 `data/` 產出一張圖片。)*

### 在自己的腳本中呼叫
Python 端的 `QAQMC_Rydberg` 已經封裝好所有細節，會自動呼叫 C++ 的平行處理：

```python
from src.qaqmc import QAQMC_Rydberg

# 建立 6 個原子的 Ruby 晶格，使用 160 個 Trotter 切片
# 底層會自動呼叫 qaqmc_cpp.QAQMCEngine 並開啟 OpenMP 運算
qmc = QAQMC_Rydberg(N=6, M=160, Omega=2.0)

# 進行模擬與採樣
qmc.run_and_save(n_equil=5000, n_samples=30000, save_name="my_simulation")
```
