@echo off
echo Creating Banking MLOps Project Structure...

:: Create main directories
mkdir data\raw data\processed data\external data\interim
mkdir models\trained models\artifacts models\registry
mkdir notebooks

:: Create MLflow core directories
mkdir mlflow\server mlflow\experiments mlflow\projects mlflow\models mlflow\registry mlflow\artifacts
mkdir mlflow\experiments\credit_risk_scoring mlflow\experiments\fraud_detection mlflow\experiments\ensemble_models
mkdir mlflow\projects\training_project mlflow\projects\validation_project mlflow\projects\deployment_project
mkdir mlflow\models\credit_risk mlflow\models\fraud_detection mlflow\models\ensemble
mkdir mlflow\registry\staging mlflow\registry\production mlflow\registry\archived
mkdir mlflow\artifacts\datasets mlflow\artifacts\features mlflow\artifacts\metrics mlflow\artifacts\plots

:: Create source directories with MLflow integration
mkdir src\banking_mlops
mkdir src\banking_mlops\mlflow_integration
mkdir src\banking_mlops\data src\banking_mlops\features src\banking_mlops\models
mkdir src\banking_mlops\training src\banking_mlops\evaluation src\banking_mlops\serving
mkdir src\banking_mlops\monitoring src\banking_mlops\utils src\banking_mlops\api
mkdir src\banking_mlops\api\endpoints src\banking_mlops\api\middleware

:: Create experiment directories
mkdir experiments\baseline_models experiments\advanced_models experiments\ensemble_experiments experiments\feature_experiments
mkdir pipelines workflows

:: Create Docker directories
mkdir docker\mlflow-server docker\mlflow-training docker\mlflow-serving docker\mlflow-gateway

:: Create Kubernetes directories
mkdir infrastructure\kubernetes\storage infrastructure\kubernetes\mlflow infrastructure\kubernetes\monitoring
mkdir infrastructure\kubernetes\ingress infrastructure\kubernetes\configmaps

:: Create configuration directories
mkdir configs\mlflow configs\models configs\environments

:: Create testing directories
mkdir tests\unit tests\integration tests\mlflow tests\models tests\performance

:: Create monitoring directories
mkdir monitoring\mlflow_metrics monitoring\dashboards monitoring\alerts

:: Create scripts directories
mkdir scripts\setup scripts\mlflow scripts\training scripts\deployment

:: Create documentation directories
mkdir docs\mlflow docs\deployment docs\tutorials

:: Create __init__.py files
echo. > src\banking_mlops\__init__.py
echo. > src\banking_mlops\mlflow_integration\__init__.py
echo. > src\banking_mlops\data\__init__.py
echo. > src\banking_mlops\features\__init__.py
echo. > src\banking_mlops\models\__init__.py
echo. > src\banking_mlops\training\__init__.py
echo. > src\banking_mlops\evaluation\__init__.py
echo. > src\banking_mlops\serving\__init__.py
echo. > src\banking_mlops\monitoring\__init__.py
echo. > src\banking_mlops\utils\__init__.py
echo. > src\banking_mlops\api\__init__.py
echo. > src\banking_mlops\api\endpoints\__init__.py
echo. > src\banking_mlops\api\middleware\__init__.py

echo Project structure created successfully!
echo Run create_environment.bat to set up the Python environment.