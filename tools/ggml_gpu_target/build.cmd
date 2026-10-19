@echo off
rem Build an explicitly named candidate; the serving executable is not replaced.
setlocal
if "%~1"=="" exit /b 2
set "target_output=%~f1"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cl /nologo /std:c++17 /EHsc /O2 /W4 /I "L:\llm-work\src\llama.cpp\ggml\include" /I "L:\llm-work\src\llama.cpp\include" "%~dp0worker.cpp" /Fo"%target_output%.obj" /Fe"%target_output%"
exit /b %errorlevel%
