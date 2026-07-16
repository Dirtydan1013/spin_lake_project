cmake_minimum_required(VERSION 3.18)
project(qaqmc_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optional CUDA backend.  Keep this OFF by default so CPU-only deployments and
# the existing qaqmc_cpp module retain exactly the same build requirements.
option(QAQMC_ENABLE_CUDA "Build the optional qaqmc_cuda backend" OFF)
set(QAQMC_CUDA_ARCHITECTURES "70;80" CACHE STRING
    "CUDA architectures for qaqmc_cuda (gpunode02: V100=70, A100=80)")

# ── Find pybind11 (installed via pip) ────────────────────────────────────────
find_package(pybind11 REQUIRED)

# ── Find OpenMP ──────────────────────────────────────────────────────────────
find_package(OpenMP)

# ── CPU backend and Python extension module ──────────────────────────────────
# Keep the CPU source tree parallel to csrc/cuda so both backends can coexist
# cleanly after branch integration. Public C++ headers live in csrc/cpu;
# implementation details and pybind registration units are isolated below it.
set(QAQMC_CPU_ENGINE_SOURCES
    csrc/cpu/detail/qaqmc_core.cpp
    csrc/cpu/detail/qaqmc_off_diagonal_core.cpp
    csrc/cpu/detail/qaqmc_renyi_core.cpp
    csrc/cpu/detail/qaqmc_renyi_work_core.cpp
    csrc/cpu/detail/sse_core.cpp
    csrc/cpu/detail/sse_off_diagonal_core.cpp
)

set(QAQMC_CPU_MODULE_SOURCES
    csrc/cpu/module/module.cpp
    csrc/cpu/module/bindings_qaqmc.cpp
    csrc/cpu/module/bindings_renyi.cpp
    csrc/cpu/module/bindings_sse.cpp
    csrc/cpu/module/bindings_renyi_work.cpp
)

pybind11_add_module(qaqmc_cpp
    ${QAQMC_CPU_ENGINE_SOURCES}
    ${QAQMC_CPU_MODULE_SOURCES}
)

target_include_directories(qaqmc_cpp PRIVATE
    csrc
    csrc/cpu
)

# ── Optimization flags ───────────────────────────────────────────────────────
# Target microarchitecture is a cache variable so the same tree deploys to any
# CPU.  Build ON the machine you run on (default 'native'), or pick a portable
# baseline when one binary must run on several hosts:
#   -DQAQMC_ARCH=native        (default) full ISA of the build machine
#   -DQAQMC_ARCH=x86-64-v3     AVX2+FMA baseline, runs on ~2015+ Intel/AMD
#   -DQAQMC_ARCH=cascadelake   the original cpunode02/beehive production value
# NOTE: changing -march changes FMA contraction/rounding, so same-seed
# trajectories are NOT bit-identical across different QAQMC_ARCH builds
# (statistically equivalent; ED comparisons must use tolerances, not bits).
set(QAQMC_ARCH "native" CACHE STRING
    "-march target: native | x86-64-v3 | cascadelake | broadwell | ...")
# -mtune does not accept the x86-64-v* ABI levels; fall back to generic tuning.
if(QAQMC_ARCH MATCHES "^x86-64")
    set(QAQMC_MTUNE "generic")
else()
    set(QAQMC_MTUNE "${QAQMC_ARCH}")
endif()
#
# Notes on what's NOT enabled:
#   - -ffast-math: re-associates FP, can shift Metropolis accept boundaries
#     and break ED-comparison tests. Avoid.
#   - -ffinite-math-only: log_W cache uses log(W) which we clamp to -1e30
#     manually; allowing finite-only assumption is unsafe.
# Debug build: AddressSanitizer + UBSan for hunting the profile-mode segfault.
#   cmake -S . -B build_asan -G Ninja -DQAQMC_ASAN=ON -DPYTHON_EXECUTABLE=$(which python)
# Run under: LD_PRELOAD=$(gcc -print-file-name=libasan.so) ASAN_OPTIONS=detect_leaks=0 ...
option(QAQMC_ASAN "Build with AddressSanitizer + UBSan (-O1, no LTO) for debugging" OFF)

# Full-speed (-O3) production build WITH -g debug symbols and no strip, so the
# OpenMPI/glibc crash handler's offset backtrace can be addr2line'd to exact
# source lines. Configure with:
#   cmake -S . -B build_dbg -G Ninja -DQAQMC_DEBUGINFO=ON \
#         -DCMAKE_BUILD_TYPE=RelWithDebInfo   (RelWithDebInfo => pybind11 won't strip)
option(QAQMC_DEBUGINFO "Production -O3 build WITH -g symbols (not stripped)" OFF)

if(QAQMC_ASAN)
    # NOTE: no -march=cascadelake here — a generic build avoids any ISA mismatch
    # on the login node and gives cleaner (non-vectorized) sanitizer traces.
    target_compile_options(qaqmc_cpp PRIVATE
        -O1 -g -fno-omit-frame-pointer
        -fno-math-errno -fno-trapping-math
        -fsanitize=address)
    target_link_options(qaqmc_cpp PRIVATE -fsanitize=address)
    # libstdc++ hardened bounds checks (vector::operator[], etc.)
    target_compile_definitions(qaqmc_cpp PRIVATE _GLIBCXX_ASSERTIONS)
    # LTO left OFF for readable sanitizer stack traces.
else()
    target_compile_options(qaqmc_cpp PRIVATE
        -O3
        -march=${QAQMC_ARCH}
        -mtune=${QAQMC_MTUNE}
        -funroll-loops
        -fno-math-errno
        -fno-trapping-math
        # keep frame pointers + debug symbols when hunting the crash (negligible cost)
        $<$<BOOL:${QAQMC_DEBUGINFO}>:-g>
        $<$<BOOL:${QAQMC_DEBUGINFO}>:-fno-omit-frame-pointer>
    )
    # Link-time optimization: lets the compiler inline across translation units
    # (channel_for_actual, compute_bond_W_inline, etc.) and prune dead code.
    set_target_properties(qaqmc_cpp PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(qaqmc_cpp PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(qaqmc_cpp PRIVATE QAQMC_USE_OPENMP)
    message(STATUS "OpenMP enabled: ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP not found — building without parallelism")
endif()

# ── Static linking for MinGW (avoid DLL load failures on Windows) ────────────
if(MINGW)
    target_link_options(qaqmc_cpp PRIVATE -static-libgcc -static-libstdc++)
    # Disable strip to avoid MSYS2 strip.exe failing on non-ASCII paths
    set_target_properties(qaqmc_cpp PROPERTIES
        LINK_FLAGS_RELEASE ""
    )
    set(CMAKE_STRIP "" CACHE FILEPATH "" FORCE)
endif()

# ── Optional CUDA extension ──────────────────────────────────────────────────
if(QAQMC_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    pybind11_add_module(qaqmc_cuda
        csrc/cuda/bindings_cuda.cpp
        csrc/cuda/bindings_cuda_runtime.cpp
        csrc/cuda/bindings_cuda_diagonal.cpp
        csrc/cuda/bindings_cuda_renyi.cpp
        csrc/cuda/bindings_cuda_batch_diagonal.cpp
        csrc/cuda/bindings_cuda_batch_renyi.cpp
        csrc/cuda/qaqmc_cuda_scan.cu
        csrc/cuda/qaqmc_cuda_diagonal.cu
        csrc/cuda/qaqmc_cuda_renyi.cu
        csrc/cuda/qaqmc_cuda_batch_diagonal.cu
        csrc/cuda/qaqmc_cuda_batch_renyi.cu
    )
    target_include_directories(qaqmc_cuda PRIVATE csrc/cuda)
    target_link_libraries(qaqmc_cuda PRIVATE CUDA::cudart)
    set_target_properties(qaqmc_cuda PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${QAQMC_CUDA_ARCHITECTURES}"
        CUDA_SEPARABLE_COMPILATION OFF
    )
    target_compile_options(qaqmc_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        # Batch engines launch independent chains from separate host threads.
        # Per-thread default streams let those kernels overlap without changing
        # the established single-chain launch signatures.
        $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>
    )
endif()
