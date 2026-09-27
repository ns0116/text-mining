@echo off
cd /d "%~dp0.."
echo =======================================================
echo Text Mining Application Builder
echo =======================================================
echo.

echo Activating virtual environment...
call .\.venv\Scripts\activate
echo Virtual environment activated.
echo.

echo Installing build requirements (PyInstaller and app dependencies)...
pip install -r requirements-build.txt
echo Done.
echo.

echo Deleting old build folders (dist, build)...
rmdir /s /q dist
rmdir /s /q build
echo Done.
echo.

echo Starting PyInstaller build...
echo This may take several minutes. Please wait.
echo.

REM --- PyInstaller Command ---
pyinstaller TextMiningApp.spec --clean

echo.
echo =======================================================
echo Build process finished.
echo Check the 'dist' folder for the executable file.
echo =======================================================
echo.
pause