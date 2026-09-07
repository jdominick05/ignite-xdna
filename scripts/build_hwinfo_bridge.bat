@echo off
setlocal enabledelayedexpansion
REM Build tools\hwinfo_npu_bridge.exe -- the XDNA1 NPU monitor / HWiNFO64 bridge -- with MSVC.
REM
REM   scripts\build_hwinfo_bridge.bat            (from the repo root; or via scripts/build-hwinfo-bridge.sh)
REM
REM Headers: nlohmann/json and boost (header-only) from the npu_monitor_build conda env
REM   (conda create -n npu_monitor_build -c conda-forge libboost-headers nlohmann_json),
REM   overridable with NPU_MONITOR_INCLUDE=<dir containing nlohmann\ and boost\>.
REM XRT SDK: if %XRT_SDK% (default C:\Xilinx\XRT\xrt_sdk\xrt) has include\xrt\xrt_device.h the
REM   monitor is built with HAVE_XRT -- in-process device queries and the live NPU clock --
REM   and links xrt_coreutil.lib (the DLL ships with the NPU driver in C:\Windows\System32).
REM   Without it the monitor still builds; the clock line then reads n/a.
REM Note: a running hwinfo_npu_bridge.exe locks its file; stop it before building.

if not exist tools\hwinfo_npu_bridge.cpp (
    echo [ERROR] run from the repository root: tools\hwinfo_npu_bridge.cpp not found
    exit /b 1
)

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "!VSWHERE!" (
    echo [ERROR] vswhere.exe not found at "!VSWHERE!" -- install Visual Studio 2022 Build Tools with the C++ workload
    exit /b 1
)
for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set "VS_DIR=%%i"
)
if not defined VS_DIR (
    echo [ERROR] Visual Studio C++ build tools not found.
    exit /b 1
)
set "VCVARS=!VS_DIR!\VC\Auxiliary\Build\vcvars64.bat"
if not exist "!VCVARS!" (
    echo [ERROR] vcvars64.bat not found at "!VCVARS!"
    exit /b 1
)
call "!VCVARS!" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to initialize the MSVC x64 build environment.
    exit /b 1
)

if not defined NPU_MONITOR_INCLUDE set "NPU_MONITOR_INCLUDE=%USERPROFILE%\miniforge3\envs\npu_monitor_build\Library\include"
if not exist "!NPU_MONITOR_INCLUDE!\nlohmann\json.hpp" (
    echo [ERROR] nlohmann\json.hpp not under "!NPU_MONITOR_INCLUDE!" -- create the env:
    echo         conda create -n npu_monitor_build -c conda-forge libboost-headers nlohmann_json
    exit /b 1
)

if not defined XRT_SDK set "XRT_SDK=C:\Xilinx\XRT\xrt_sdk\xrt"
set "XRT_FLAGS="
set "XRT_LINK="
if exist "!XRT_SDK!\include\xrt\xrt_device.h" (
    if exist "!NPU_MONITOR_INCLUDE!\boost\any.hpp" (
        set "XRT_FLAGS=/DHAVE_XRT /I "!XRT_SDK!\include""
        set "XRT_LINK=/LIBPATH:"!XRT_SDK!\lib" xrt_coreutil.lib"
        echo [INFO] XRT SDK found at "!XRT_SDK!" -- building with HAVE_XRT
    ) else (
        echo [WARN] XRT SDK found but boost headers missing under "!NPU_MONITOR_INCLUDE!" -- building WITHOUT the XRT API
    )
) else (
    echo [WARN] no XRT SDK at "!XRT_SDK!" -- building WITHOUT the XRT API ^(no clock line^)
)

echo [INFO] cl.exe tools\hwinfo_npu_bridge.cpp
cl.exe /nologo /std:c++17 /EHsc /O2 /MD /utf-8 /W3 !XRT_FLAGS! /I "!NPU_MONITOR_INCLUDE!" tools\hwinfo_npu_bridge.cpp /Fo:tools\hwinfo_npu_bridge.obj /Fe:tools\hwinfo_npu_bridge.exe /link !XRT_LINK!
if errorlevel 1 (
    echo [ERROR] C++ compilation failed.
    exit /b 1
)
if exist tools\hwinfo_npu_bridge.obj del tools\hwinfo_npu_bridge.obj

echo [SUCCESS] built tools\hwinfo_npu_bridge.exe
echo          try: tools\hwinfo_npu_bridge.exe --once   ^(one plain sample^)   or   tools\hwinfo_npu_bridge.exe   ^(live dashboard^)
