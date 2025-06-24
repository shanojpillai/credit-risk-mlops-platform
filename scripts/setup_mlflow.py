#!/usr/bin/env python3
"""
MLflow Setup Script for Credit Risk MLOps Platform
Creates experiments and initializes the MLflow environment
"""

import mlflow
from mlflow.tracking import MlflowClient
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_mlflow():
    """Initialize MLflow tracking server and experiments"""

    # Set tracking URI
    tracking_uri = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(tracking_uri)

    logger.info(f"Setting up MLflow with tracking URI: {tracking_uri}")

    # Create MLflow directories
    mlflow_dirs = [
        "mlflow/artifacts",
        "mlflow/experiments",
        "mlflow/models",
        "mlflow/registry"
    ]

    for dir_path in mlflow_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

    # Create experiments with detailed configuration
    experiments = [
        {
            "name": "credit-risk-baseline",
            "artifact_location": "mlflow/artifacts/credit-risk-baseline/",
            "tags": {
                "project": "credit-risk-mlops",
                "team": "ml-engineering",
                "use_case": "credit_scoring",
                "model_type": "baseline"
            }
        },
        {
            "name": "credit-risk-advanced",
            "artifact_location": "mlflow/artifacts/credit-risk-advanced/",
            "tags": {
                "project": "credit-risk-mlops",
                "team": "ml-engineering",
                "use_case": "credit_scoring",
                "model_type": "advanced"
            }
        },
        {
            "name": "fraud-detection",
            "artifact_location": "mlflow/artifacts/fraud-detection/",
            "tags": {
                "project": "credit-risk-mlops",
                "team": "ml-engineering",
                "use_case": "fraud_detection"
            }
        },
        {
            "name": "ensemble-models",
            "artifact_location": "mlflow/artifacts/ensemble-models/",
            "tags": {
                "project": "credit-risk-mlops",
                "team": "ml-engineering",
                "use_case": "ensemble",
                "model_type": "ensemble"
            }
        },
        {
            "name": "hyperparameter-tuning",
            "artifact_location": "mlflow/artifacts/hyperparameter-tuning/",
            "tags": {
                "project": "credit-risk-mlops",
                "team": "ml-engineering",
                "use_case": "optimization"
            }
        }
    ]

    client = MlflowClient()

    for exp_config in experiments:
        try:
            # Create artifact directory
            artifact_path = Path(exp_config["artifact_location"])
            artifact_path.mkdir(parents=True, exist_ok=True)

            experiment_id = client.create_experiment(
                name=exp_config["name"],
                artifact_location=exp_config["artifact_location"],
                tags=exp_config["tags"]
            )
            logger.info(f"✅ Created experiment: {exp_config['name']} (ID: {experiment_id})")
        except Exception as e:
            if "already exists" in str(e):
                experiment = client.get_experiment_by_name(exp_config["name"])
                logger.info(f"✅ Experiment exists: {exp_config['name']} (ID: {experiment.experiment_id})")
            else:
                logger.error(f"❌ Failed to create experiment {exp_config['name']}: {e}")

def create_model_registry_entries():
    """Create initial model registry entries"""

    client = MlflowClient()

    models = [
        {
            "name": "credit_risk_logistic_regression",
            "description": "Baseline logistic regression model for credit risk prediction"
        },
        {
            "name": "credit_risk_xgboost",
            "description": "XGBoost model for credit risk prediction"
        },
        {
            "name": "credit_risk_lightgbm",
            "description": "LightGBM model for credit risk prediction"
        },
        {
            "name": "credit_risk_ensemble",
            "description": "Ensemble model combining multiple algorithms"
        }
    ]

    for model_config in models:
        try:
            client.create_registered_model(
                name=model_config["name"],
                description=model_config["description"]
            )
            logger.info(f"✅ Created registered model: {model_config['name']}")
        except Exception as e:
            if "already exists" in str(e):
                logger.info(f"✅ Registered model exists: {model_config['name']}")
            else:
                logger.error(f"❌ Failed to create registered model {model_config['name']}: {e}")

def main():
    """Main setup function"""

    logger.info("🚀 Starting MLflow setup for Credit Risk MLOps Platform")

    # Setup MLflow experiments
    setup_mlflow()

    # Create model registry entries
    create_model_registry_entries()

    logger.info("✅ MLflow setup completed successfully!")
    logger.info("📊 Next steps:")
    logger.info("   1. Start MLflow server: mlflow server --host 0.0.0.0 --port 5000")
    logger.info("   2. Download dataset: python scripts/download_dataset.py")
    logger.info("   3. Run first experiment: python experiments/baseline_models/logistic_regression.py")
    logger.info("   4. Open MLflow UI: http://localhost:5000")

if __name__ == "__main__":
    main()