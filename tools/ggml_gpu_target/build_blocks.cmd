@echo off
rem Fixed-block research worker; the serving executable is never replaced.
setlocal
if "%~1"=="" exit /b 2
set "block_output=%~f1"
if exist "%block_output%" exit /b 2
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /I "L:\llm-work\src\llama.cpp\ggml\include" /I "L:\llm-work\src\llama.cpp\include" "%~dp0worker_blocks.cpp" /Fo"%block_output%.obj" /Fe"%block_output%"
exit /b %errorlevel%
