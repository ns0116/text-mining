@echo off
cd /d "%~dp0.."
REM ========================================================
REM requirements.txt 自動生成バッチファイル
REM ========================================================

echo.
echo =======================================================
echo Creating requirements.txt for the current environment...
echo =======================================================
echo.

echo Activating virtual environment...
call .\.venv\Scripts\activate
echo Virtual environment activated.
echo.

echo Generating library list using 'pip freeze'...
pip freeze > requirements.txt
echo.

echo =======================================================
echo 'requirements.txt' has been created successfully!
echo =======================================================
echo.
pause