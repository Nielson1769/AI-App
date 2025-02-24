@echo off
title AI Environment Setup
echo ====================================================
echo        AUTOMATED AI APP SETUP (Python 3.11)
echo ====================================================
echo.
echo This script will:
echo 1. Uninstall Python 3.13 (if installed)
echo 2. Download and Install Python 3.11
echo 3. Create a virtual environment
echo 4. Install all required dependencies
echo 5. Launch your AI app
echo.
pause

:: Step 1: Uninstall Python 3.13 if detected
echo Checking for Python 3.13 installation...
where python > nul 2>&1
if %errorlevel% == 0 (
    python --version | findstr "3.13" > nul
    if %errorlevel% == 0 (
        echo Python 3.13 detected. Uninstalling...
        wmic product where "name like 'Python%%3.13%%'" call uninstall /nointeractive
        echo Uninstalling Python 3.13. Please wait...
        timeout /t 10
    ) else (
        echo Python 3.13 NOT detected. Skipping uninstall.
    )
) else (
    echo Python is not installed or not found in PATH.
)

:: Step 2: Download and Install Python 3.11
echo Downloading Python 3.11 installer...
powershell -Command "& {Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.6/python-3.11.6-amd64.exe' -OutFile 'python-3.11.6.exe'}"
echo Installing Python 3.11...
start /wait python-3.11.6.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
del python-3.11.6.exe
echo Python 3.11 installed successfully.

:: Step 3: Ve
