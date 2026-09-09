@echo off
setlocal enabledelayedexpansion
REM Build kernels\dispatch_floor\dispatch_runner.exe -- the C++ XRT host for the AIE
REM dispatch floor -- with MSVC.
REM
REM   scripts\build_dispatch_runner.bat          (from the repository root)
REM
REM WHY A C++ HOST: results/aie/dispatch_runlist_npu.log reached 36.3 us per dispatch from
REM   raw pyxrt and results/aie/dispatch_iron_batch_npu.log showed IRON adds ~500 us of
REM   host work per call, flat in batch size.  Both were measured through pybind11, so
REM   neither separates the binding's cost from the driver's.  This host removes Python.
REM
REM XRT SDK: %XRT_SDK% (default C:\Xilinx\XRT\xrt_sdk\xrt) must have include\xrt\xrt_kernel.h
REM   and include\xrt\experimental\xrt_kernel.h (xrt::runlist).  Links xrt_coreutil.lib;
REM   the DLL ships with the NPU driver in C:\Windows\System32.
REM Boost: the XRT public headers include boost/any.hpp, which is header-only and comes
REM   from the same npu_monitor_build conda env scripts\build_hwinfo_bridge.bat uses
REM   (conda create -n npu_monitor_build -c conda-forge libboost-headers nlohmann_json).
REM   Override with NPU_MONITOR_INCLUDE=<dir containing boost\>.

if not exist kernels\dispatch_floor\dispatch_runner.cpp (
    echo [ERROR] run from the repository root: kernels\dispatch_floor\dispatch_runner.cpp not found
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

if not defined XRT_SDK set "XRT_SDK=C:\Xilinx\XRT\xrt_sdk\xrt"
if not exist "!XRT_SDK!\include\xrt\experimental\xrt_kernel.h" (
    echo [ERROR] no xrt::runlist header at "!XRT_SDK!\include\xrt\experimental\xrt_kernel.h"
    echo         set XRT_SDK to an XRT SDK that has it.
    exit /b 1
)

if not defined NPU_MONITOR_INCLUDE set "NPU_MONITOR_INCLUDE=%USERPROFILE%\miniforge3\envs\npu_monitor_build\Library\include"
set "BOOST_FLAGS="
if exist "!NPU_MONITOR_INCLUDE!\boost\any.hpp" (
    set "BOOST_FLAGS=/I "!NPU_MONITOR_INCLUDE!""
) else (
    echo [WARN] boost headers not under "!NPU_MONITOR_INCLUDE!" -- if the XRT headers need
    echo        boost/any.hpp this build will fail; create the npu_monitor_build env.
)

echo [INFO] cl.exe kernels\dispatch_floor\dispatch_runner.cpp
cl.exe /nologo /std:c++17 /EHsc /O2 /MD /utf-8 /W3 /I "!XRT_SDK!\include" !BOOST_FLAGS! kernels\dispatch_floor\dispatch_runner.cpp /Fo:kernels\dispatch_floor\dispatch_runner.obj /Fe:kernels\dispatch_floor\dispatch_runner.exe /link /LIBPATH:"!XRT_SDK!\lib" xrt_coreutil.lib
if errorlevel 1 (
    echo [ERROR] C++ compilation failed.
    exit /b 1
)
if exist kernels\dispatch_floor\dispatch_runner.obj del kernels\dispatch_floor\dispatch_runner.obj

echo [SUCCESS] built kernels\dispatch_floor\dispatch_runner.exe
echo          try: kernels\dispatch_floor\dispatch_runner.exe --cache-newest --iters 100
