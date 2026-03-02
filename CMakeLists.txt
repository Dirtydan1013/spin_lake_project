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
    csrc/bindings.cpp
)

target_include_directories(qaqmc_cpp PRIVATE csrc)
target_compile_options(qaqmc_cpp PRIVATE -O2)

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
