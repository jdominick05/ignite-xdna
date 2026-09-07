@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo Building HWiNFO64 AMD XDNA1 NPU Custom Sensor Bridge
echo ===================================================

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "!VSWHERE!" (
    echo [ERROR] vswhere.exe not found at "!VSWHERE!"
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
    echo [ERROR] Failed to initialize MSVC x64 build environment.
    exit /b 1
)

echo [INFO] Compiling native C++ executable with MSVC...
cl.exe /std:c++17 /EHsc /O2 /MT /I "C:\Users\Ignis\miniforge3\envs\resnet_env17\Library\include" tools\hwinfo_npu_bridge.cpp /Fe:tools\hwinfo_npu_bridge.exe /link Advapi32.lib User32.lib
if errorlevel 1 (
    echo [ERROR] C++ compilation failed.
    exit /b 1
)

if exist hwinfo_npu_bridge.obj del hwinfo_npu_bridge.obj

echo [SUCCESS] Built tools\hwinfo_npu_bridge.exe successfully!
