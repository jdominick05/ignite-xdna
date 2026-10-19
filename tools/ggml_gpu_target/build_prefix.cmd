@echo off
rem Build the explicit prefix-reuse candidate; no serving binary is replaced.
setlocal
if "%~1"=="" exit /b 2
set "prefix_output=%~f1"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /I "L:\llm-work\src\llama.cpp\ggml\include" /I "L:\llm-work\src\llama.cpp\include" "%~dp0worker_prefix.cpp" /Fo"%prefix_output%.obj" /Fe"%prefix_output%"
exit /b %errorlevel%
