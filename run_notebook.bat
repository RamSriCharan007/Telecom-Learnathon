@echo off
cd /d "%~dp0"
echo ===================================================
echo Re-executing all cells in Jupyter Notebook...
echo ===================================================
"%~dp0.venv\Scripts\python.exe" "%~dp0create_notebook.py"
echo.
echo ===================================================
echo Notebook successfully executed and saved!
echo File: telco_churn_pipeline.ipynb
echo ===================================================
pause
