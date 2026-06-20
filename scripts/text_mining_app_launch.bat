@echo off
cd /d "%~dp0.."
echo =======================================================
echo Text Mining Application Launcher
echo =======================================================
echo.

echo Activating virtual environment...
call .\.venv\Scripts\activate
echo Virtual environment activated.
echo.

echo Starting ...
echo.

streamlit run src\text_mining_app.py 

echo.
echo =======================================================
echo =======================================================
echo.
pause