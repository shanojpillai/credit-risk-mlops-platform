@echo off
echo Activating virtual environment and running MLflow setup...
call banking-mlops-env\Scripts\activate.bat
python scripts\setup_mlflow.py
pause
