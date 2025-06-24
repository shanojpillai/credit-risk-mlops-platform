# 🏭 Credit Card Default-mlops [Default Payments of Credit Card Clients in Taiwan from 2005]

<!-- Badges Section -->
<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.8%2B-orange?style=for-the-badge&logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-20.10%2B-blue?style=for-the-badge&logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-1.24%2B-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)

![License](https://img.shields.io/github/license/shanojpillai/credit-risk-mlops-platform?style=for-the-badge&color=green)
![Stars](https://img.shields.io/github/stars/shanojpillai/credit-risk-mlops-platform?style=for-the-badge&color=yellow)
![Forks](https://img.shields.io/github/forks/shanojpillai/credit-risk-mlops-platform?style=for-the-badge&color=blue)
![Issues](https://img.shields.io/github/issues/shanojpillai/credit-risk-mlops-platform?style=for-the-badge&color=red)

![Last Commit](https://img.shields.io/github/last-commit/shanojpillai/credit-risk-mlops-platform?style=for-the-badge&color=brightgreen)
![Repo Size](https://img.shields.io/github/repo-size/shanojpillai/credit-risk-mlops-platform?style=for-the-badge&color=orange)
![Contributors](https://img.shields.io/github/contributors/shanojpillai/credit-risk-mlops-platform?style=for-the-badge&color=purple)

<!-- Technology Stack Badges -->
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square&logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13%2B-336791?style=flat-square&logo=postgresql&logoColor=white)
![MinIO](https://img.shields.io/badge/MinIO-S3%20Compatible-C72E29?style=flat-square&logo=minio&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C?style=flat-square&logo=prometheus&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-Dashboards-F46800?style=flat-square&logo=grafana&logoColor=white)

<!-- ML/AI Badges -->
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-FF6600?style=flat-square&logo=xgboost&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-3.3%2B-2E8B57?style=flat-square&logo=lightgbm&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?style=flat-square&logo=numpy&logoColor=white)

<!-- Status Badges -->
![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen?style=flat-square)
![MLflow Status](https://img.shields.io/badge/MLflow%20Server-Running-success?style=flat-square)
![Code Quality](https://img.shields.io/badge/Code%20Quality-A-brightgreen?style=flat-square)
![Coverage](https://img.shields.io/badge/Coverage-85%25-green?style=flat-square)

</div>

---

## 🎯 MLflow-Centric On-Premise Architecture

A **complete MLflow-powered MLOps platform** designed for enterprise banking environments with **MLflow at the center** of all ML operations.

## 🔧 MLflow-Centered Technology Stack

### **📊 MLflow Ecosystem (Core)**
- **MLflow Tracking Server** - Centralized experiment tracking
- **MLflow Model Registry** - Production model management
- **MLflow Projects** - Reproducible ML workflows
- **MLflow Models** - Model packaging and serving
- **MLflow Plugins** - Custom integrations

### **🐳 Container Infrastructure**
- **Docker & Docker Compose** - Development environment
- **Kubernetes** - Production orchestration
- **MLflow Server Container** - Containerized tracking server
- **PostgreSQL** - MLflow backend store
- **MinIO** - MLflow artifact store (S3-compatible)

### **⚡ Supporting Stack**
- **FastAPI** - API gateway for MLflow models
- **Prometheus & Grafana** - MLflow metrics monitoring
- **NGINX** - Load balancing for MLflow services

## 🏗️ MLflow-Centric Project Structure

```
credit-risk-mlops-platform/
├── 📊 MLflow Core
│   ├── mlflow/
│   │   ├── server/
│   │   │   ├── Dockerfile.mlflow           # Custom MLflow server
│   │   │   ├── mlflow_config.yaml          # Server configuration
│   │   │   ├── start_server.sh             # Server startup script
│   │   │   └── plugins/                    # Custom MLflow plugins
│   │   │
│   │   ├── experiments/
│   │   │   ├── credit_risk_scoring/        # Experiment definitions
│   │   │   ├── fraud_detection/
│   │   │   └── ensemble_models/
│   │   │
│   │   ├── projects/
│   │   │   ├── MLproject                   # MLflow project file
│   │   │   ├── conda.yaml                  # Environment spec
│   │   │   ├── training_project/
│   │   │   ├── validation_project/
│   │   │   └── deployment_project/
│   │   │
│   │   ├── models/
│   │   │   ├── credit_risk/
│   │   │   │   ├── MLmodel                 # Model specification
│   │   │   │   ├── model.pkl               # Model artifacts
│   │   │   │   ├── conda.yaml              # Model environment
│   │   │   │   └── requirements.txt
│   │   │   │
│   │   │   ├── fraud_detection/
│   │   │   └── ensemble/
│   │   │
│   │   ├── registry/
│   │   │   ├── staging/                    # Staging models
│   │   │   ├── production/                 # Production models
│   │   │   └── archived/                   # Archived models
│   │   │
│   │   └── artifacts/
│   │       ├── datasets/                   # Versioned datasets
│   │       ├── features/                   # Feature artifacts
│   │       ├── metrics/                    # Evaluation metrics
│   │       └── plots/                      # Visualization artifacts
│
├── 🐳 Container Infrastructure
│   ├── docker/
│   │   ├── mlflow-server/
│   │   │   ├── Dockerfile                  # MLflow tracking server
│   │   │   ├── entrypoint.sh
│   │   │   └── requirements.txt
│   │   │
│   │   ├── mlflow-training/
│   │   │   ├── Dockerfile                  # Training environment
│   │   │   ├── train.py
│   │   │   └── requirements.txt
│   │   │
│   │   ├── mlflow-serving/
│   │   │   ├── Dockerfile                  # Model serving
│   │   │   ├── serve.py
│   │   │   └── requirements.txt
│   │   │
│   │   └── mlflow-gateway/
│   │       ├── Dockerfile                  # API Gateway for MLflow
│   │       ├── gateway.py
│   │       └── requirements.txt
│   │
│   ├── docker-compose.yml                  # Development MLflow stack
│   ├── docker-compose.prod.yml             # Production MLflow stack
│   └── .env.example                       # MLflow configuration
│
├── ☸️ Kubernetes MLflow Deployment
│   ├── k8s/
│   │   ├── namespace.yaml                  # MLflow namespace
│   │   │
│   │   ├── storage/
│   │   │   ├── postgres-mlflow.yaml        # MLflow backend store
│   │   │   ├── minio-mlflow.yaml           # MLflow artifact store
│   │   │   └── persistent-volumes.yaml
│   │   │
│   │   ├── mlflow/
│   │   │   ├── tracking-server.yaml        # MLflow tracking deployment
│   │   │   ├── model-registry.yaml         # Model registry service
│   │   │   ├── model-serving.yaml          # Model serving deployment
│   │   │   └── mlflow-gateway.yaml         # MLflow gateway
│   │   │
│   │   ├── monitoring/
│   │   │   ├── prometheus-mlflow.yaml      # MLflow metrics
│   │   │   ├── grafana-mlflow.yaml         # MLflow dashboards
│   │   │   └── mlflow-alerts.yaml
│   │   │
│   │   ├── ingress/
│   │   │   ├── mlflow-ingress.yaml         # External access
│   │   │   └── tls-certificates.yaml
│   │   │
│   │   └── configmaps/
│   │       ├── mlflow-config.yaml
│   │       └── model-configs.yaml
│
├── 🏢 Application Code (MLflow Integration)
│   ├── src/
│   │   └── banking_mlops/
│   │       ├── mlflow_integration/
│   │       │   ├── __init__.py
│   │       │   ├── tracking_client.py      # MLflow tracking wrapper
│   │       │   ├── model_registry.py       # Registry management
│   │       │   ├── experiment_manager.py   # Experiment lifecycle
│   │       │   ├── artifact_manager.py     # Artifact handling
│   │       │   └── serving_client.py       # Model serving client
│   │       │
│   │       ├── training/
│   │       │   ├── __init__.py
│   │       │   ├── mlflow_trainer.py       # MLflow-integrated training
│   │       │   ├── experiment_runner.py    # Run MLflow experiments
│   │       │   ├── hyperparameter_tuning.py # MLflow + Optuna
│   │       │   └── pipeline_runner.py      # MLflow pipelines
│   │       │
│   │       ├── models/
│   │       │   ├── __init__.py
│   │       │   ├── mlflow_model_wrapper.py # Custom MLflow model
│   │       │   ├── credit_risk_model.py    # Credit risk with MLflow
│   │       │   ├── fraud_model.py          # Fraud detection with MLflow
│   │       │   └── ensemble_model.py       # Ensemble with MLflow
│   │       │
│   │       ├── serving/
│   │       │   ├── __init__.py
│   │       │   ├── mlflow_predictor.py     # MLflow model serving
│   │       │   ├── batch_predictor.py      # Batch predictions
│   │       │   ├── real_time_api.py        # Real-time API
│   │       │   └── model_loader.py         # Load from MLflow registry
│   │       │
│   │       ├── data/
│   │       │   ├── __init__.py
│   │       │   ├── mlflow_datasets.py      # MLflow dataset logging
│   │       │   ├── data_versioning.py      # Data version control
│   │       │   ├── feature_store.py        # Feature storage in MLflow
│   │       │   └── data_validation.py      # Log validation to MLflow
│   │       │
│   │       ├── monitoring/
│   │       │   ├── __init__.py
│   │       │   ├── mlflow_monitor.py       # Monitor MLflow models
│   │       │   ├── drift_detection.py      # Log drift to MLflow
│   │       │   ├── performance_tracker.py  # Track model performance
│   │       │   └── alert_manager.py        # MLflow-based alerts
│   │       │
│   │       ├── api/
│   │       │   ├── __init__.py
│   │       │   ├── main.py                 # FastAPI with MLflow
│   │       │   ├── endpoints/
│   │       │   │   ├── prediction.py       # Predict via MLflow
│   │       │   │   ├── models.py           # Model management
│   │       │   │   ├── experiments.py      # Experiment API
│   │       │   │   └── monitoring.py       # Monitoring API
│   │       │   │
│   │       │   └── middleware/
│   │       │       ├── mlflow_auth.py      # MLflow authentication
│   │       │       └── logging.py          # Log to MLflow
│   │       │
│   │       └── utils/
│   │           ├── __init__.py
│   │           ├── mlflow_config.py        # MLflow configuration
│   │           ├── mlflow_helpers.py       # Helper functions
│   │           ├── experiment_utils.py     # Experiment utilities
│   │           └── model_utils.py          # Model utilities
│
├── 🧪 MLflow Experiments & Training
│   ├── experiments/
│   │   ├── baseline_models/
│   │   │   ├── logistic_regression.py      # Log to MLflow
│   │   │   ├── random_forest.py            # Log to MLflow
│   │   │   └── xgboost_baseline.py         # Log to MLflow
│   │   │
│   │   ├── advanced_models/
│   │   │   ├── xgboost_tuned.py           # Hyperparameter tuning
│   │   │   ├── lightgbm_model.py          # LightGBM with MLflow
│   │   │   └── neural_network.py          # NN with MLflow
│   │   │
│   │   ├── ensemble_experiments/
│   │   │   ├── stacking_ensemble.py
│   │   │   ├── voting_ensemble.py
│   │   │   └── blending_ensemble.py
│   │   │
│   │   └── feature_experiments/
│   │       ├── feature_selection.py
│   │       ├── feature_engineering.py
│   │       └── feature_importance.py
│   │
│   ├── pipelines/
│   │   ├── training_pipeline.py           # Complete MLflow pipeline
│   │   ├── validation_pipeline.py         # Model validation
│   │   ├── deployment_pipeline.py         # Deployment via MLflow
│   │   └── monitoring_pipeline.py         # Monitoring pipeline
│   │
│   └── workflows/
│       ├── daily_training.py              # Scheduled training
│       ├── model_validation.py            # Daily validation
│       └── performance_check.py           # Performance monitoring
│
├── 📊 MLflow Configuration
│   ├── configs/
│   │   ├── mlflow/
│   │   │   ├── tracking_config.yaml       # Tracking server config
│   │   │   ├── registry_config.yaml       # Model registry config
│   │   │   ├── serving_config.yaml        # Model serving config
│   │   │   └── experiments_config.yaml    # Experiment definitions
│   │   │
│   │   ├── models/
│   │   │   ├── credit_risk_config.yaml
│   │   │   ├── fraud_detection_config.yaml
│   │   │   └── ensemble_config.yaml
│   │   │
│   │   └── environments/
│   │       ├── development.yaml
│   │       ├── staging.yaml
│   │       └── production.yaml
│
├── 🧪 Testing (MLflow-focused)
│   ├── tests/
│   │   ├── mlflow/
│   │   │   ├── test_tracking.py           # Test MLflow tracking
│   │   │   ├── test_registry.py           # Test model registry
│   │   │   ├── test_serving.py            # Test model serving
│   │   │   └── test_experiments.py        # Test experiments
│   │   │
│   │   ├── models/
│   │   │   ├── test_model_training.py     # Test MLflow model training
│   │   │   ├── test_model_loading.py      # Test model loading
│   │   │   └── test_model_serving.py      # Test model serving
│   │   │
│   │   ├── integration/
│   │   │   ├── test_end_to_end.py         # E2E MLflow workflow
│   │   │   ├── test_pipeline.py           # Pipeline testing
│   │   │   └── test_api_integration.py    # API + MLflow tests
│   │   │
│   │   └── performance/
│   │       ├── test_mlflow_performance.py # MLflow performance tests
│   │       └── test_model_inference.py    # Model serving performance
│
├── 📈 MLflow Monitoring & Observability
│   ├── monitoring/
│   │   ├── mlflow_metrics/
│   │   │   ├── custom_metrics.py          # Custom MLflow metrics
│   │   │   ├── model_metrics.py           # Model performance metrics
│   │   │   └── system_metrics.py          # MLflow system metrics
│   │   │
│   │   ├── dashboards/
│   │   │   ├── mlflow_overview.json       # MLflow Grafana dashboard
│   │   │   ├── model_performance.json     # Model metrics dashboard
│   │   │   ├── experiment_tracking.json   # Experiment overview
│   │   │   └── registry_status.json       # Registry status
│   │   │
│   │   └── alerts/
│   │       ├── mlflow_alerts.yaml         # MLflow-specific alerts
│   │       ├── model_drift_alerts.yaml    # Model drift alerts
│   │       └── performance_alerts.yaml    # Performance alerts
│
├── 🔧 Scripts & Tools (MLflow-focused)
│   ├── scripts/
│   │   ├── setup/
│   │   │   ├── setup_mlflow_dev.sh        # Development MLflow setup
│   │   │   ├── setup_mlflow_prod.sh       # Production MLflow setup
│   │   │   └── initialize_mlflow.py       # Initialize MLflow server
│   │   │
│   │   ├── mlflow/
│   │   │   ├── create_experiments.py      # Create MLflow experiments
│   │   │   ├── register_models.py         # Register models
│   │   │   ├── promote_models.py          # Promote model stages
│   │   │   ├── cleanup_experiments.py     # Cleanup old experiments
│   │   │   └── backup_mlflow.py           # Backup MLflow data
│   │   │
│   │   ├── training/
│   │   │   ├── run_experiments.py         # Run training experiments
│   │   │   ├── compare_models.py          # Compare MLflow models
│   │   │   └── hyperparameter_search.py   # Hyperparameter optimization
│   │   │
│   │   └── deployment/
│   │       ├── deploy_mlflow_k8s.sh       # Deploy MLflow to K8s
│   │       ├── deploy_models.py           # Deploy models from registry
│   │       └── health_check.py            # MLflow health check
│
├── 📚 Documentation (MLflow-focused)
│   ├── docs/
│   │   ├── mlflow/
│   │   │   ├── setup_guide.md             # MLflow setup guide
│   │   │   ├── experiment_guide.md        # How to run experiments
│   │   │   ├── model_registry_guide.md    # Model registry usage
│   │   │   ├── serving_guide.md           # Model serving guide
│   │   │   └── best_practices.md          # MLflow best practices
│   │   │
│   │   ├── deployment/
│   │   │   ├── on_premise_deployment.md   # On-premise MLflow
│   │   │   ├── kubernetes_deployment.md   # K8s MLflow deployment
│   │   │   └── docker_deployment.md       # Docker MLflow setup
│   │   │
│   │   └── tutorials/
│   │       ├── getting_started.md         # MLflow quickstart
│   │       ├── advanced_features.md       # Advanced MLflow usage
│   │       └── troubleshooting.md         # MLflow troubleshooting
│
└── 📦 Dependencies & Configuration
    ├── pyproject.toml                     # Poetry with MLflow deps
    ├── requirements.txt                   # MLflow requirements
    ├── mlflow_requirements.txt            # MLflow-specific deps
    ├── .mlflow_config                     # MLflow CLI config
    └── MLproject                          # MLflow project definition
```

## 🚀 MLflow-Centric Setup Scripts

### **1. Development MLflow Setup** (`scripts/setup_mlflow_dev.sh`)

```bash
#!/bin/bash
set -e

echo "📊 Setting up MLflow Development Environment..."

# Setup Python environment with Poetry
setup_python_env() {
    echo "🐍 Setting up Python environment..."
    
    # Install Poetry if not present
    if ! command -v poetry &> /dev/null; then
        curl -sSL https://install.python-poetry.org | python3 -
        export PATH="$HOME/.local/bin:$PATH"
    fi
    
    # Install dependencies
    poetry install
    poetry run pip install mlflow[extras]==2.9.2
    
    echo "✅ Python environment ready"
}

# Setup MLflow infrastructure with Docker
setup_mlflow_infrastructure() {
    echo "🐳 Starting MLflow infrastructure..."
    
    # Start PostgreSQL and MinIO for MLflow
    docker-compose up -d postgres minio
    
    # Wait for services
    echo "⏳ Waiting for services to start..."
    sleep 30
    
    # Initialize MLflow database
    poetry run mlflow db upgrade postgresql://mlflow:mlflow@localhost:5432/mlflow
    
    echo "✅ MLflow infrastructure ready"
}

# Start MLflow tracking server
start_mlflow_server() {
    echo "🚀 Starting MLflow tracking server..."
    
    export MLFLOW_BACKEND_STORE_URI="postgresql://mlflow:mlflow@localhost:5432/mlflow"
    export MLFLOW_DEFAULT_ARTIFACT_ROOT="s3://mlflow-artifacts/"
    export AWS_ACCESS_KEY_ID="minio"
    export AWS_SECRET_ACCESS_KEY="minio123"
    export MLFLOW_S3_ENDPOINT_URL="http://localhost:9000"
    
    # Start MLflow server in background
    nohup poetry run mlflow server \
        --backend-store-uri $MLFLOW_BACKEND_STORE_URI \
        --default-artifact-root $MLFLOW_DEFAULT_ARTIFACT_ROOT \
        --host 0.0.0.0 \
        --port 5000 > mlflow_server.log 2>&1 &
    
    echo "✅ MLflow server started at http://localhost:5000"
}

# Create initial experiments
create_experiments() {
    echo "🧪 Creating MLflow experiments..."
    
    poetry run python scripts/mlflow/create_experiments.py
    
    echo "✅ Experiments created"
}

# Main setup function
main() {
    setup_python_env
    setup_mlflow_infrastructure
    start_mlflow_server
    create_experiments
    
    echo ""
    echo "🎉 MLflow Development Environment Ready!"
    echo "📊 MLflow UI: http://localhost:5000"
    echo "🗄️ MinIO Console: http://localhost:9001"
    echo ""
    echo "Next steps:"
    echo "1. Open MLflow UI to view experiments"
    echo "2. Run your first experiment: poetry run python experiments/baseline_models/logistic_regression.py"
    echo "3. Check model registry for trained models"
}

main
```

### **2. MLflow Experiment Creation** (`scripts/mlflow/create_experiments.py`)

```python
#!/usr/bin/env python3
"""Create initial MLflow experiments for Credit Risk MLOps Platform"""

import mlflow
from mlflow.tracking import MlflowClient
import os

def setup_mlflow_client():
    """Setup MLflow client with proper configuration"""
    mlflow.set_tracking_uri("http://localhost:5000")
    return MlflowClient()

def create_experiments():
    """Create all required experiments for the platform"""
    client = setup_mlflow_client()
    
    experiments = [
        {
            "name": "credit-risk-baseline",
            "artifact_location": "s3://mlflow-artifacts/credit-risk-baseline/",
            "tags": {
                "project": "credit-risk-mlops",
                "team": "ml-engineering",
                "use_case": "credit_scoring"
            }
        },
        {
            "name": "credit-risk-advanced",
            "artifact_location": "s3://mlflow-artifacts/credit-risk-advanced/",
            "tags": {
                "project": "credit-risk-mlops",
                "team": "ml-engineering", 
                "use_case": "credit_scoring",
                "model_type": "advanced"
            }
        },
        {
            "name": "fraud-detection",
            "artifact_location": "s3://mlflow-artifacts/fraud-detection/",
            "tags": {
                "project": "credit-risk-mlops",
                "team": "ml-engineering",
                "use_case": "fraud_detection"
            }
        },
        {
            "name": "ensemble-models",
            "artifact_location": "s3://mlflow-artifacts/ensemble-models/",
            "tags": {
                "project": "credit-risk-mlops",
                "team": "ml-engineering",
                "use_case": "ensemble",
                "model_type": "ensemble"
            }
        },
        {
            "name": "hyperparameter-tuning",
            "artifact_location": "s3://mlflow-artifacts/hyperparameter-tuning/",
            "tags": {
                "project": "credit-risk-mlops",
                "team": "ml-engineering",
                "use_case": "optimization"
            }
        }
    ]
    
    for exp_config in experiments:
        try:
            experiment_id = client.create_experiment(
                name=exp_config["name"],
                artifact_location=exp_config["artifact_location"],
                tags=exp_config["tags"]
            )
            print(f"✅ Created experiment: {exp_config['name']} (ID: {experiment_id})")
        except Exception as e:
            if "already exists" in str(e):
                experiment = client.get_experiment_by_name(exp_config["name"])
                print(f"✅ Experiment exists: {exp_config['name']} (ID: {experiment.experiment_id})")
            else:
                print(f"❌ Failed to create experiment {exp_config['name']}: {e}")

if __name__ == "__main__":
    create_experiments()
```

## 🎯 MLflow Project Highlights

### **📊 Experiment Tracking**
- **Comprehensive Logging**: All models, metrics, parameters, and artifacts
- **Experiment Organization**: Separate experiments for different use cases
- **Parameter Comparison**: Easy comparison across different runs
- **Artifact Management**: Versioned datasets, models, and visualizations

### **🗂️ Model Registry**
- **Model Versioning**: Track all model versions
- **Stage Management**: Development → Staging → Production
- **Model Lineage**: Complete history of model evolution
- **Deployment Integration**: Direct deployment from registry

### **🚀 Model Serving**
- **MLflow Models**: Standard model packaging
- **REST API**: Automatic REST API generation
- **Batch Inference**: Batch prediction capabilities
- **Real-time Serving**: Low-latency prediction serving

### **🔍 Monitoring & Observability**
- **MLflow Metrics**: Custom business and technical metrics
- **Model Performance**: Track accuracy, drift, and performance
- **Experiment Analytics**: Comprehensive experiment analysis
- **Production Monitoring**: Real-time model monitoring

This setup puts **MLflow at the center** of your MLOps platform while maintaining production-grade capabilities!
