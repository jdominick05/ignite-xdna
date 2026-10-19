@echo off
rem Explicit research-only logits export; never replaces serving outputs.
setlocal
if "%~2"=="" exit /b 2
set "logit_source="
if "%~1"=="serving" set "logit_source=worker_logits.cpp"
if "%~1"=="blocks" set "logit_source=worker_blocks_logits.cpp"
if "%logit_source%"=="" exit /b 2
set "logit_output=%~f2"
if exist "%logit_output%" exit /b 2
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /I "L:\llm-work\src\llama.cpp\ggml\include" /I "L:\llm-work\src\llama.cpp\include" "%~dp0%logit_source%" /Fo"%logit_output%.obj" /Fe"%logit_output%"
exit /b %errorlevel%
