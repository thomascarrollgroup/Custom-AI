@echo off
echo ================================
echo TC AI Prediction Tool Builder
echo ================================
echo.

cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and add it to your PATH
    pause
    exit /b 1
)

REM Check if pip is available
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: pip is not available
    echo Please ensure pip is installed with Python
    pause
    exit /b 1
)

echo Installing/updating dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Cleaning previous build artifacts...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist __pycache__ rmdir /s /q __pycache__
if exist "*.spec" del "*.spec"

echo.
echo Building executable with PyInstaller...
pyinstaller ^
    --onefile ^
    --windowed ^
    --icon=icon.ico ^
    --name="TC_AI_Prediction_Tool" ^
    --add-data="ui;ui" ^
    --add-data="core;core" ^
    --add-data="Images;Images" ^
    --add-data="Font;Font" ^
    --add-data="icon.ico;." ^
    --hidden-import=PyQt5.QtCore ^
    --hidden-import=PyQt5.QtGui ^
    --hidden-import=PyQt5.QtWidgets ^
    --hidden-import=pandas ^
    --hidden-import=numpy ^
    --hidden-import=sklearn ^
    --hidden-import=matplotlib ^
    --hidden-import=seaborn ^
    --hidden-import=psycopg2 ^
    --hidden-import=requests ^
    --hidden-import=PIL ^
    --hidden-import=joblib ^
    --collect-submodules=sklearn ^
    --collect-submodules=matplotlib ^
    --collect-submodules=seaborn ^
    --collect-submodules=PyQt5 ^
    --exclude-module=tkinter ^
    --exclude-module=tests ^
    --exclude-module=pytest ^
    main.py

if %errorlevel% neq 0 (
    echo.
    echo ERROR: PyInstaller failed to build the executable
    echo Check the output above for specific error messages
    pause
    exit /b 1
)

echo.
echo ================================
echo Build completed successfully!
echo ================================
echo.
echo The executable has been created in the 'dist' folder:
echo %~dp0dist\TC_AI_Prediction_Tool.exe
echo.
echo File size and details:
dir dist\TC_AI_Prediction_Tool.exe

echo.
echo Testing the executable...
start "" "dist\TC_AI_Prediction_Tool.exe"

echo.
echo Build process completed!
echo You can find your executable at: dist\TC_AI_Prediction_Tool.exe
echo.
pause
EOF
)