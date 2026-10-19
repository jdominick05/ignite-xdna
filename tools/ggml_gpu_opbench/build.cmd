@echo off
rem Build opbench.exe into scratch\ggml-opbench. From the repo root: cmd /d /c tools\ggml_gpu_opbench\build.cmd
set "VSWHERE=C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -property installationPath`) do set "VS=%%i"
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
if not exist scratch\ggml-opbench mkdir scratch\ggml-opbench
cl /nologo /std:c++17 /EHsc /O2 /W3 /I L:\llm-work\src\llama.cpp\ggml\include tools\ggml_gpu_opbench\opbench.cpp /Foscratch\ggml-opbench\opbench.obj /Fescratch\ggml-opbench\opbench.exe
