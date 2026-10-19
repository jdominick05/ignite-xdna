@echo off
rem Explicit research output only; never replaces the serving executable.
setlocal
if "%~1"=="" exit /b 2
set "stage_output=%~f1"
if exist "%stage_output%" exit /b 2
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /I "L:\llm-work\src\llama.cpp\ggml\include" /I "L:\llm-work\src\llama.cpp\include" "%~dp0worker_stages.cpp" /Fo"%stage_output%.obj" /Fe"%stage_output%"
exit /b %errorlevel%
