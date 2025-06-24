@echo off
echo Setting up Banking MLOps Environment...

:: Check if conda is available
conda --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Conda not found. Please install Anaconda or Miniconda first.
    exit /b 1
)

:: Create conda environment
echo Creating conda environment 'banking-mlops'...
conda create -n banking-mlops python=3.9 -y

:: Activate environment
echo Activating environment...
call conda activate banking-mlops

:: Install core packages
echo Installing core packages...
conda install -c conda-forge pandas numpy scikit-learn matplotlib seaborn jupyter -y

:: Install MLflow and related packages
echo Installing MLflow and ML packages...
pip install mlflow[extras]==2.8.1
pip install xgboost lightgbm catboost
pip install optuna hyperopt
pip install evidently great-expectations
pip install fastapi uvicorn pydantic
pip install pytest pytest-cov black flake8 isort
pip install docker kubernetes
pip install plotly dash streamlit

:: Install banking-specific packages
pip install imbalanced-learn shap lime
pip install category-encoders feature-engine

echo Environment setup complete!
echo.
echo To activate the environment, run:
echo conda activate banking-mlops
echo.
echo Next steps:
echo 1. Run git init to initialize repository
echo 2. Configure MLflow tracking server
echo 3. Download datasets from Kaggle