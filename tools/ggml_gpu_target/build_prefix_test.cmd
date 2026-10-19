@echo off
setlocal
if "%~1"=="" exit /b 2
set "prefix_test_output=%~f1"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX "%~dp0prefix_history_test.cpp" /Fo"%prefix_test_output%.obj" /Fe"%prefix_test_output%"
exit /b %errorlevel%
