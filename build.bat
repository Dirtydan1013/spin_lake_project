@echo off
REM Build script for qaqmc_cpp using CMake + Ninja + MinGW
REM Usage: build.bat

REM Ensure venv tools (cmake, python, ninja) and MSYS2 tools (g++) are on PATH
set PATH=%~dp0.venv\Scripts;D:\msys64\ucrt64\bin;%PATH%
set BUILD_DIR=build

echo [1/3] Configuring with CMake...
cmake -S . -B %BUILD_DIR% -G Ninja ^
    -DCMAKE_CXX_COMPILER=g++ ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_STRIP="" ^
    -DPYTHON_EXECUTABLE=%~dp0.venv\Scripts\python.exe ^
    -DPython3_EXECUTABLE=%~dp0.venv\Scripts\python.exe ^
    -DPython_EXECUTABLE=%~dp0.venv\Scripts\python.exe ^
    -Dpybind11_DIR=%~dp0.venv\Lib\site-packages\pybind11\share\cmake\pybind11

if %ERRORLEVEL% NEQ 0 (
    echo CMake configuration FAILED
    exit /b 1
)

echo.
echo [2/3] Building...
cmake --build %BUILD_DIR% --config Release -j

echo.
echo [3/3] Copying .pyd to project root...
for %%f in (%BUILD_DIR%\qaqmc_cpp*.pyd) do (
    copy /Y "%%f" . >nul
    echo   Copied %%~nxf
)

echo.
echo Build complete!
python -c "import os; os.add_dll_directory(os.path.dirname(os.path.realpath(__import__('shutil').which('g++')))); import qaqmc_cpp; print(f'  OpenMP: {qaqmc_cpp.has_openmp}, max_threads: {qaqmc_cpp.omp_max_threads}')"
