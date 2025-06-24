import os
import mlflow
from mlflow.tracking import MlflowClient

def setup_mlflow():
    """Initialize MLflow tracking server and experiments"""
    
    # Set tracking URI
    mlflow.set_tracking_uri("file:./mlflow")
    
    # Create experiments
    experiments = [
        "credit-risk-scoring",
        "fraud-detection", 
        "customer-segmentation",
        "loan-default-prediction"
    ]
    
    client = MlflowClient()
    
    for exp_name in experiments:
        try:
            experiment_id = mlflow.create_experiment(exp_name)
            print(f"Created experiment: {exp_name} (ID: {experiment_id})")
        except Exception as e:
            print(f"Experiment {exp_name} already exists or error: {e}")

if __name__ == "__main__":
    setup_mlflow()