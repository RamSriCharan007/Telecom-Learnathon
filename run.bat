@echo off
cd /d "%~dp0"
echo ===================================================
echo Running Telco Churn Machine Learning Pipeline...
echo ===================================================
"%~dp0.venv\Scripts\python.exe" "%~dp0run_pipeline.py"
echo.
echo ===================================================
echo Pipeline execution finished!
echo Check reports/figures and reports/metrics for output.
echo ===================================================
pause
