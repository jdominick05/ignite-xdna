@echo off
rem Build an explicitly named research-only D3D11 GPU activity helper.
setlocal
if "%~1"=="" exit /b 2
set "activity_output=%~f1"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cl /nologo /std:c++17 /EHsc /O2 /W4 "%~dp0gpu_activity.cpp" /Fo"%activity_output%.obj" /Fe"%activity_output%" /link d3d11.lib dxgi.lib d3dcompiler.lib
exit /b %errorlevel%
