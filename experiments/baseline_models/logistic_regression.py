#!/usr/bin/env python3
"""
Baseline Logistic Regression Model for Credit Risk Prediction
MLflow-integrated experiment
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLflow configuration
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "credit-risk-baseline"

def load_and_prepare_data():
    """Load and prepare the credit card default dataset"""
    
    # Load data
    data_path = Path("data/raw/default_credit_card_clients.csv")
    if not data_path.exists():
        logger.error(f"Dataset not found at {data_path}")
        logger.info("Please run: python scripts/download_dataset.py")
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    
    # Basic data cleaning
    # Rename target column for clarity
    if 'default payment next month' in df.columns:
        df = df.rename(columns={'default payment next month': 'default'})
    elif 'default.payment.next.month' in df.columns:
        df = df.rename(columns={'default.payment.next.month': 'default'})
    
    # Remove ID column if present
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
    
    # Handle any missing values
    df = df.dropna()
    
    logger.info(f"Cleaned dataset shape: {df.shape}")
    logger.info(f"Target distribution:\n{df['default'].value_counts()}")
    
    return df

def create_features(df):
    """Create features for the model"""
    
    # Separate features and target
    X = df.drop('default', axis=1)
    y = df['default']
    
    # Feature engineering
    # Create utilization ratio features
    for i in range(1, 7):
        bill_col = f'BILL_AMT{i}'
        pay_col = f'PAY_AMT{i}'
        
        if bill_col in X.columns and pay_col in X.columns:
            # Payment ratio
            X[f'payment_ratio_{i}'] = X[pay_col] / (X[bill_col] + 1)
            
            # Utilization ratio (bill amount / credit limit)
            X[f'utilization_ratio_{i}'] = X[bill_col] / (X['LIMIT_BAL'] + 1)
    
    # Average payment status
    pay_cols = [col for col in X.columns if col.startswith('PAY_')]
    if pay_cols:
        X['avg_pay_status'] = X[pay_cols].mean(axis=1)
        X['max_pay_status'] = X[pay_cols].max(axis=1)
    
    # Average bill amount
    bill_cols = [col for col in X.columns if col.startswith('BILL_AMT')]
    if bill_cols:
        X['avg_bill_amt'] = X[bill_cols].mean(axis=1)
        X['total_bill_amt'] = X[bill_cols].sum(axis=1)
    
    # Average payment amount
    pay_amt_cols = [col for col in X.columns if col.startswith('PAY_AMT')]
    if pay_amt_cols:
        X['avg_pay_amt'] = X[pay_amt_cols].mean(axis=1)
        X['total_pay_amt'] = X[pay_amt_cols].sum(axis=1)
    
    logger.info(f"Created features. Final feature count: {X.shape[1]}")
    
    return X, y

def train_logistic_regression(X_train, y_train, X_test, y_test, params):
    """Train logistic regression model with MLflow tracking"""
    
    with mlflow.start_run(run_name="logistic_regression_baseline") as run:
        
        # Log parameters
        mlflow.log_params(params)
        
        # Log dataset info
        mlflow.log_params({
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "n_features": X_train.shape[1],
            "target_distribution_train": f"{y_train.value_counts().to_dict()}",
            "target_distribution_test": f"{y_test.value_counts().to_dict()}"
        })
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LogisticRegression(**params)
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba)
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        metrics["cv_roc_auc_mean"] = cv_scores.mean()
        metrics["cv_roc_auc_std"] = cv_scores.std()
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="credit_risk_logistic_regression"
        )
        
        # Log scaler
        mlflow.sklearn.log_model(scaler, "scaler")
        
        # Create and log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Logistic Regression')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()
        
        # Log classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_dict(report, "classification_report.json")
        
        # Feature importance (coefficients)
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'coefficient': model.coef_[0],
            'abs_coefficient': np.abs(model.coef_[0])
        }).sort_values('abs_coefficient', ascending=False)
        
        # Log top 20 features
        top_features = feature_importance.head(20)
        plt.figure(figsize=(10, 8))
        sns.barplot(data=top_features, x='coefficient', y='feature')
        plt.title('Top 20 Feature Coefficients - Logistic Regression')
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        plt.close()
        
        # Log feature importance as CSV
        feature_importance.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")
        
        logger.info(f"Model trained successfully. Run ID: {run.info.run_id}")
        logger.info(f"Metrics: {metrics}")
        
        return model, scaler, metrics

def main():
    """Main function to run the logistic regression experiment"""
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Set experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    logger.info("Starting Logistic Regression Baseline Experiment")
    
    # Load and prepare data
    df = load_and_prepare_data()
    X, y = create_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Model parameters
    params = {
        "random_state": 42,
        "max_iter": 1000,
        "solver": "liblinear",
        "C": 1.0,
        "penalty": "l2"
    }
    
    # Train model
    _, _, metrics = train_logistic_regression(
        X_train, y_train, X_test, y_test, params
    )
    
    logger.info("✅ Logistic Regression experiment completed successfully!")
    logger.info(f"🎯 ROC AUC Score: {metrics['roc_auc']:.4f}")
    logger.info(f"📊 MLflow UI: {MLFLOW_TRACKING_URI}")
    
    # Clean up temporary files
    for file in ["confusion_matrix.png", "feature_importance.png", "feature_importance.csv"]:
        if os.path.exists(file):
            os.remove(file)

if __name__ == "__main__":
    main()
