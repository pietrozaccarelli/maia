@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1

echo ==========================================
echo      M.A.I.A. Environment Setup
echo ==========================================

:: ---------------------------------------------------------
:: STEP 0: FIND A COMPATIBLE PYTHON VERSION (3.10 - 3.12)
:: ---------------------------------------------------------
echo [0/7] Checking Python Version...

set "TARGET_PY="

:: 1. Try to find Python 3.11 (Ideal)
py -3.11 --version >nul 2>nul
if %errorlevel% equ 0 set "TARGET_PY=py -3.11" & goto :FOUND_PY

:: 2. Try to find Python 3.12 (Good)
py -3.12 --version >nul 2>nul
if %errorlevel% equ 0 set "TARGET_PY=py -3.12" & goto :FOUND_PY

:: 3. Try to find Python 3.10 (Okay)
py -3.10 --version >nul 2>nul
if %errorlevel% equ 0 set "TARGET_PY=py -3.10" & goto :FOUND_PY

:: 4. Fallback: Check default 'python' command
python --version >nul 2>nul
if %errorlevel% neq 0 goto :NO_PYTHON

:: Check if default python is too new (3.13 or 3.14)
python -c "import sys; exit(1 if sys.version_info >= (3, 13) else 0)"
if %errorlevel% neq 0 (
    echo.
    echo [CRITICAL ERROR]
    echo You are using Python 3.13 or 3.14.
    echo These versions are TOO NEW and do not support AI libraries yet.
    echo.
    echo Please install Python 3.11 from python.org
    echo (Make sure to check "Add python.exe to PATH" during install)
    echo.
    pause
    exit /b 1
)

:: If default python is okay (3.10-3.12), use it
set "TARGET_PY=python"

:FOUND_PY
echo Using compatible Python: %TARGET_PY%

:: ---------------------------------------------------------
:: STEP 1: OLLAMA CHECK
:: ---------------------------------------------------------
echo [1/7] Checking/Installing Ollama...
where ollama >nul 2>nul
if %errorlevel% neq 0 (
    echo Ollama not found. Installing via Winget...
    winget install Ollama.Ollama --accept-source-agreements --accept-package-agreements
    if exist "%LOCALAPPDATA%\Programs\Ollama" (
        set "PATH=%LOCALAPPDATA%\Programs\Ollama;%PATH%"
    )
)

:: ---------------------------------------------------------
:: STEP 2: RUN PYTHON INITIALIZATION
:: ---------------------------------------------------------
echo [2/7] Initializing Environment and Dependencies...

:: We pass the specific python command to the script so it knows what to use for venv
%TARGET_PY% "initialize_maia.py"

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] The initialization script encountered errors.
    pause
    exit /b 1
)

:: ---------------------------------------------------------
:: STEP 3: GENERATE LAUNCHER
:: ---------------------------------------------------------
echo.
echo [7/7] Generating 'MAIA.bat'...

set "ABS_PATH=%~dp0"
if "%ABS_PATH:~-1%"=="\" set "ABS_PATH=%ABS_PATH:~0,-1%"

(
    echo @echo off
    echo set PYTHONUTF8=1
    echo echo Starting M.A.I.A...
    echo cd /d "%ABS_PATH%"
    echo call "%ABS_PATH%\venv\Scripts\activate.bat"
    echo python "%ABS_PATH%\launcher.py"
    echo pause
) > MAIA.bat

echo.
echo ==========================================
echo Setup Complete! 
echo ==========================================
pause
goto :eof

:NO_PYTHON
echo [ERROR] Python not found. Please install Python 3.11.
pause
exit /b 1