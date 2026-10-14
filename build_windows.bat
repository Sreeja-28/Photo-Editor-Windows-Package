@echo off
title Photo Editor - Windows EXE Builder
echo ==========================================
echo        PHOTO EDITOR EXE BUILDER
echo ==========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or is not in PATH.
    echo Install Python 3.11 or newer and try again.
    pause
    exit /b 1
)

echo Installing required packages...
python -m pip install --upgrade pyinstaller pillow opencv-python numpy
if errorlevel 1 (
    echo.
    echo Package installation failed.
    pause
    exit /b 1
)

echo.
echo Cleaning previous build...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo.
echo Building PhotoEditor.exe...
python -m PyInstaller --clean --noconfirm PhotoEditor.spec
if errorlevel 1 (
    echo.
    echo BUILD FAILED.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo BUILD COMPLETE
echo ==========================================
echo.
echo EXE:
echo %CD%\dist\PhotoEditor.exe
echo.
echo Double-click dist\PhotoEditor.exe to run it.
echo.
pause
