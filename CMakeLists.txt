cmake_minimum_required(VERSION 3.18)
project(qaqmc_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Find pybind11 (installed via pip) ────────────────────────────────────────
find_package(pybind11 REQUIRED)

# ── Find OpenMP ──────────────────────────────────────────────────────────────
find_package(OpenMP)

# ── Build the Python extension module ────────────────────────────────────────
pybind11_add_module(qaqmc_cpp
    csrc/qaqmc_core.cpp
    csrc/qaqmc_off_diagonal_core.cpp
    csrc/qaqmc_renyi_core.cpp
    csrc/qaqmc_renyi_work_core.cpp
    csrc/sse_core.cpp
    csrc/bindings.cpp
)

target_include_directories(qaqmc_cpp PRIVATE csrc)

# ── Optimization flags ───────────────────────────────────────────────────────
# Production target: cpunode02 (Intel Xeon Gold 5218, Cascade Lake, AVX-512).
# beehive (login) is also Cascade Lake so this binary runs on either; to add
# cpunode01 (Coffee Lake, no AVX-512) drop -march to broadwell.
#
# Notes on what's NOT enabled:
#   - -ffast-math: re-associates FP, can shift Metropolis accept boundaries
#     and break ED-comparison tests. Avoid.
#   - -ffinite-math-only: log_W cache uses log(W) which we clamp to -1e30
#     manually; allowing finite-only assumption is unsafe.
target_compile_options(qaqmc_cpp PRIVATE
    -O3
    -march=cascadelake
    -mtune=cascadelake
    -funroll-loops
    -fno-math-errno
    -fno-trapping-math
)

# Link-time optimization: lets the compiler inline across translation units
# (channel_for_actual, compute_bond_W_inline, etc.) and prune dead code.
set_target_properties(qaqmc_cpp PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)

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
