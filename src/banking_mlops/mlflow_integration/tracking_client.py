"""
MLflow Tracking Client - Centralized MLflow tracking functionality
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class MLflowTrackingClient:
    """Centralized MLflow tracking client for the banking MLOps platform"""
    
    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        """
        Initialize MLflow tracking client
        
        Args:
            tracking_uri: MLflow tracking server URI
        """
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri)
        
        logger.info(f"MLflow tracking client initialized with URI: {tracking_uri}")
    
    def create_experiment(self, name: str, artifact_location: Optional[str] = None, 
                         tags: Optional[Dict[str, str]] = None) -> str:
        """
        Create a new MLflow experiment
        
        Args:
            name: Experiment name
            artifact_location: S3 or local path for artifacts
            tags: Experiment tags
            
        Returns:
            Experiment ID
        """
        try:
            experiment_id = self.client.create_experiment(
                name=name,
                artifact_location=artifact_location,
                tags=tags or {}
            )
            logger.info(f"Created experiment: {name} (ID: {experiment_id})")
            return experiment_id
        except Exception as e:
            if "already exists" in str(e):
                experiment = self.client.get_experiment_by_name(name)
                logger.info(f"Experiment already exists: {name} (ID: {experiment.experiment_id})")
                return experiment.experiment_id
            else:
                logger.error(f"Failed to create experiment {name}: {e}")
                raise
    
    def start_run(self, experiment_name: str, run_name: Optional[str] = None,
                  tags: Optional[Dict[str, str]] = None) -> mlflow.ActiveRun:
        """
        Start a new MLflow run
        
        Args:
            experiment_name: Name of the experiment
            run_name: Optional run name
            tags: Run tags
            
        Returns:
            Active MLflow run
        """
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        # Start run
        run = mlflow.start_run(run_name=run_name, tags=tags or {})
        logger.info(f"Started run: {run.info.run_id} in experiment: {experiment_name}")
        
        return run
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to current run"""
        mlflow.log_params(params)
        logger.debug(f"Logged parameters: {list(params.keys())}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to current run"""
        mlflow.log_metrics(metrics, step=step)
        logger.debug(f"Logged metrics: {list(metrics.keys())}")
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a single metric to current run"""
        mlflow.log_metric(key, value, step=step)
        logger.debug(f"Logged metric: {key} = {value}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log an artifact to current run"""
        mlflow.log_artifact(local_path, artifact_path)
        logger.debug(f"Logged artifact: {local_path}")
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        """Log artifacts directory to current run"""
        mlflow.log_artifacts(local_dir, artifact_path)
        logger.debug(f"Logged artifacts directory: {local_dir}")
    
    def log_model(self, model: Any, artifact_path: str, **kwargs) -> None:
        """
        Log a model to current run
        
        Args:
            model: Model object
            artifact_path: Path within run's artifact directory
            **kwargs: Additional arguments for model logging
        """
        # Determine model type and log accordingly
        model_type = type(model).__name__
        
        if hasattr(model, 'predict'):
            if 'xgboost' in str(type(model)).lower():
                mlflow.xgboost.log_model(model, artifact_path, **kwargs)
            elif 'lightgbm' in str(type(model)).lower():
                mlflow.lightgbm.log_model(model, artifact_path, **kwargs)
            else:
                mlflow.sklearn.log_model(model, artifact_path, **kwargs)
        else:
            # Fallback to pickle
            mlflow.log_artifact(model, artifact_path)
        
        logger.info(f"Logged {model_type} model to: {artifact_path}")
    
    def log_dataset(self, dataset: pd.DataFrame, name: str, 
                   artifact_path: str = "datasets") -> None:
        """
        Log a dataset to current run
        
        Args:
            dataset: Pandas DataFrame
            name: Dataset name
            artifact_path: Artifact path for dataset
        """
        # Create temporary file
        temp_path = f"/tmp/{name}.csv"
        dataset.to_csv(temp_path, index=False)
        
        # Log as artifact
        mlflow.log_artifact(temp_path, artifact_path)
        
        # Log dataset metadata
        mlflow.log_params({
            f"{name}_shape": f"{dataset.shape[0]}x{dataset.shape[1]}",
            f"{name}_columns": len(dataset.columns),
            f"{name}_memory_usage": f"{dataset.memory_usage(deep=True).sum() / 1024**2:.2f}MB"
        })
        
        # Clean up
        os.remove(temp_path)
        
        logger.info(f"Logged dataset: {name} with shape {dataset.shape}")
    
    def end_run(self, status: str = "FINISHED") -> None:
        """End current MLflow run"""
        mlflow.end_run(status=status)
        logger.info(f"Ended run with status: {status}")
    
    def get_experiment_runs(self, experiment_name: str, 
                           max_results: int = 100) -> List[mlflow.entities.Run]:
        """
        Get runs from an experiment
        
        Args:
            experiment_name: Name of the experiment
            max_results: Maximum number of runs to return
            
        Returns:
            List of MLflow runs
        """
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning(f"Experiment not found: {experiment_name}")
            return []
        
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=max_results
        )
        
        logger.info(f"Retrieved {len(runs)} runs from experiment: {experiment_name}")
        return runs
    
    def get_best_run(self, experiment_name: str, metric_name: str, 
                     ascending: bool = False) -> Optional[mlflow.entities.Run]:
        """
        Get the best run from an experiment based on a metric
        
        Args:
            experiment_name: Name of the experiment
            metric_name: Metric to optimize
            ascending: Whether to sort in ascending order (True for loss, False for accuracy)
            
        Returns:
            Best MLflow run or None
        """
        runs = self.get_experiment_runs(experiment_name)
        
        if not runs:
            return None
        
        # Filter runs that have the metric
        runs_with_metric = [run for run in runs if metric_name in run.data.metrics]
        
        if not runs_with_metric:
            logger.warning(f"No runs found with metric: {metric_name}")
            return None
        
        # Sort by metric
        best_run = sorted(
            runs_with_metric,
            key=lambda x: x.data.metrics[metric_name],
            reverse=not ascending
        )[0]
        
        logger.info(f"Best run: {best_run.info.run_id} with {metric_name}={best_run.data.metrics[metric_name]}")
        return best_run
    
    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple runs
        
        Args:
            run_ids: List of run IDs to compare
            
        Returns:
            DataFrame with run comparison
        """
        runs_data = []
        
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            run_data = {
                'run_id': run_id,
                'experiment_id': run.info.experiment_id,
                'status': run.info.status,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
            }
            
            # Add parameters
            run_data.update({f"param_{k}": v for k, v in run.data.params.items()})
            
            # Add metrics
            run_data.update({f"metric_{k}": v for k, v in run.data.metrics.items()})
            
            runs_data.append(run_data)
        
        comparison_df = pd.DataFrame(runs_data)
        logger.info(f"Compared {len(run_ids)} runs")
        
        return comparison_df
