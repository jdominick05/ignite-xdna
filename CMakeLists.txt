# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

cmake_minimum_required(VERSION 3.20)
project(ignite_xdna LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Compiler warnings and optimization flags
if(MSVC)
    add_compile_options(
        /W3
        /EHsc
        /fp:fast
        /utf-8
        /openmp
        /arch:AVX2
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Release>:/Oi>
        $<$<CONFIG:Release>:/GL>
    )
    add_link_options(
        $<$<CONFIG:Release>:/LTCG>
        $<$<CONFIG:Release>:/OPT:REF>
        $<$<CONFIG:Release>:/OPT:ICF>
    )
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra -O3 -march=native)
endif()

# Find OpenMP for SIMD ingress preprocessor
find_package(OpenMP REQUIRED)

# XRT Root and Conda Library paths
set(XRT_ROOT_DIR "C:/Xilinx/XRT/xrt_sdk/xrt" CACHE PATH "Path to XRT SDK")
if(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
    set(CONDA_PREFIX "$ENV{CONDA_PREFIX}")
else()
    set(CONDA_PREFIX "C:/Users/Ignis/miniforge3/envs/mlir-aie-iron")
endif()

message(STATUS "Using CONDA_PREFIX: ${CONDA_PREFIX}")
message(STATUS "Using XRT_ROOT_DIR: ${XRT_ROOT_DIR}")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${XRT_ROOT_DIR}/include
    "C:/Users/Ignis/miniforge3/envs/mlir-aie-iron/Library/include"
    ${CONDA_PREFIX}/Library/include
)

link_directories(
    ${XRT_ROOT_DIR}/lib
)

# -----------------------------------------------------------------------------
# Target: libignite_xdna (Shared Library)
# -----------------------------------------------------------------------------
add_library(ignite_xdna SHARED
    src/ignite_xdna/c_api/ignite.cpp
    src/ignite_xdna/pipelines/preprocess_simd.c
)

target_compile_definitions(ignite_xdna PRIVATE IGNITE_EXPORTS)
target_link_libraries(ignite_xdna PRIVATE
    xrt_coreutil
    OpenMP::OpenMP_C
    OpenMP::OpenMP_CXX
)

set_target_properties(ignite_xdna PROPERTIES
    OUTPUT_NAME "ignite_xdna"
    PREFIX "lib"
)

# -----------------------------------------------------------------------------
# Target: ignite-run (Standalone C++ CLI)
# -----------------------------------------------------------------------------
add_executable(ignite-run
    tools/ignite_run_native.cpp
)

target_link_libraries(ignite-run PRIVATE
    ignite_xdna
    xrt_coreutil
    gdiplus
    mfplat
    mfreadwrite
    mfuuid
)

set_target_properties(ignite-run PROPERTIES
    OUTPUT_NAME "ignite-run"
)
