"""
MLflow Model Registry Management
"""

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MLflowModelRegistry:
    """MLflow Model Registry management for banking MLOps platform"""
    
    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        """
        Initialize MLflow Model Registry client
        
        Args:
            tracking_uri: MLflow tracking server URI
        """
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri)
        
        logger.info(f"MLflow Model Registry client initialized with URI: {tracking_uri}")
    
    def create_registered_model(self, name: str, description: Optional[str] = None,
                               tags: Optional[Dict[str, str]] = None) -> None:
        """
        Create a new registered model
        
        Args:
            name: Model name
            description: Model description
            tags: Model tags
        """
        try:
            registered_model = self.client.create_registered_model(
                name=name,
                description=description,
                tags=tags or {}
            )
            logger.info(f"Created registered model: {name}")
            return registered_model
        except MlflowException as e:
            if "already exists" in str(e):
                logger.info(f"Registered model already exists: {name}")
                return self.client.get_registered_model(name)
            else:
                logger.error(f"Failed to create registered model {name}: {e}")
                raise
    
    def register_model(self, model_uri: str, name: str, 
                      description: Optional[str] = None,
                      tags: Optional[Dict[str, str]] = None) -> ModelVersion:
        """
        Register a model version
        
        Args:
            model_uri: URI of the model (e.g., runs:/<run_id>/model)
            name: Registered model name
            description: Version description
            tags: Version tags
            
        Returns:
            ModelVersion object
        """
        # Ensure registered model exists
        self.create_registered_model(name)
        
        # Register model version
        model_version = self.client.create_model_version(
            name=name,
            source=model_uri,
            description=description,
            tags=tags or {}
        )
        
        logger.info(f"Registered model version: {name} v{model_version.version}")
        return model_version
    
    def promote_model(self, name: str, version: str, stage: str,
                     archive_existing: bool = True) -> ModelVersion:
        """
        Promote a model version to a specific stage
        
        Args:
            name: Registered model name
            version: Model version
            stage: Target stage (Staging, Production, Archived)
            archive_existing: Whether to archive existing models in target stage
            
        Returns:
            Updated ModelVersion
        """
        # Archive existing models in the target stage if requested
        if archive_existing and stage in ["Staging", "Production"]:
            existing_versions = self.client.get_latest_versions(name, stages=[stage])
            for existing_version in existing_versions:
                self.client.transition_model_version_stage(
                    name=name,
                    version=existing_version.version,
                    stage="Archived",
                    archive_existing_versions=False
                )
                logger.info(f"Archived existing model: {name} v{existing_version.version}")
        
        # Promote the new version
        model_version = self.client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage,
            archive_existing_versions=False
        )
        
        logger.info(f"Promoted model: {name} v{version} to {stage}")
        return model_version
    
    def get_model_versions(self, name: str, stage: Optional[str] = None) -> List[ModelVersion]:
        """
        Get model versions for a registered model
        
        Args:
            name: Registered model name
            stage: Optional stage filter
            
        Returns:
            List of ModelVersion objects
        """
        if stage:
            versions = self.client.get_latest_versions(name, stages=[stage])
        else:
            registered_model = self.client.get_registered_model(name)
            versions = registered_model.latest_versions
        
        logger.info(f"Retrieved {len(versions)} versions for model: {name}")
        return versions
    
    def get_production_model(self, name: str) -> Optional[ModelVersion]:
        """
        Get the production version of a model
        
        Args:
            name: Registered model name
            
        Returns:
            Production ModelVersion or None
        """
        production_versions = self.client.get_latest_versions(name, stages=["Production"])
        
        if production_versions:
            logger.info(f"Found production model: {name} v{production_versions[0].version}")
            return production_versions[0]
        else:
            logger.warning(f"No production model found for: {name}")
            return None
    
    def load_model(self, name: str, stage: str = "Production") -> Any:
        """
        Load a model from the registry
        
        Args:
            name: Registered model name
            stage: Model stage to load
            
        Returns:
            Loaded model object
        """
        model_uri = f"models:/{name}/{stage}"
        
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Loaded model: {name} from stage: {stage}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {name} from {stage}: {e}")
            raise
    
    def compare_model_versions(self, name: str, versions: List[str]) -> pd.DataFrame:
        """
        Compare multiple versions of a model
        
        Args:
            name: Registered model name
            versions: List of version numbers to compare
            
        Returns:
            DataFrame with version comparison
        """
        comparison_data = []
        
        for version in versions:
            try:
                model_version = self.client.get_model_version(name, version)
                
                # Get run information
                run = self.client.get_run(model_version.run_id)
                
                version_data = {
                    'version': version,
                    'stage': model_version.current_stage,
                    'creation_timestamp': model_version.creation_timestamp,
                    'last_updated_timestamp': model_version.last_updated_timestamp,
                    'run_id': model_version.run_id,
                    'status': model_version.status,
                    'description': model_version.description
                }
                
                # Add run metrics
                version_data.update({f"metric_{k}": v for k, v in run.data.metrics.items()})
                
                # Add run parameters
                version_data.update({f"param_{k}": v for k, v in run.data.params.items()})
                
                comparison_data.append(version_data)
                
            except Exception as e:
                logger.error(f"Failed to get data for version {version}: {e}")
        
        comparison_df = pd.DataFrame(comparison_data)
        logger.info(f"Compared {len(versions)} versions of model: {name}")
        
        return comparison_df
    
    def delete_model_version(self, name: str, version: str) -> None:
        """
        Delete a model version
        
        Args:
            name: Registered model name
            version: Version to delete
        """
        self.client.delete_model_version(name, version)
        logger.info(f"Deleted model version: {name} v{version}")
    
    def update_model_description(self, name: str, version: str, 
                                description: str) -> ModelVersion:
        """
        Update model version description
        
        Args:
            name: Registered model name
            version: Model version
            description: New description
            
        Returns:
            Updated ModelVersion
        """
        model_version = self.client.update_model_version(
            name=name,
            version=version,
            description=description
        )
        
        logger.info(f"Updated description for model: {name} v{version}")
        return model_version
    
    def set_model_tag(self, name: str, version: str, key: str, value: str) -> None:
        """
        Set a tag on a model version
        
        Args:
            name: Registered model name
            version: Model version
            key: Tag key
            value: Tag value
        """
        self.client.set_model_version_tag(name, version, key, value)
        logger.info(f"Set tag {key}={value} on model: {name} v{version}")
    
    def get_model_history(self, name: str) -> pd.DataFrame:
        """
        Get the complete history of a registered model
        
        Args:
            name: Registered model name
            
        Returns:
            DataFrame with model history
        """
        registered_model = self.client.get_registered_model(name)
        
        history_data = []
        for version in registered_model.latest_versions:
            history_data.append({
                'version': version.version,
                'stage': version.current_stage,
                'creation_timestamp': version.creation_timestamp,
                'last_updated_timestamp': version.last_updated_timestamp,
                'run_id': version.run_id,
                'status': version.status,
                'description': version.description
            })
        
        history_df = pd.DataFrame(history_data)
        history_df = history_df.sort_values('creation_timestamp', ascending=False)
        
        logger.info(f"Retrieved history for model: {name}")
        return history_df
    
    def list_registered_models(self) -> List[str]:
        """
        List all registered models
        
        Returns:
            List of registered model names
        """
        registered_models = self.client.list_registered_models()
        model_names = [model.name for model in registered_models]
        
        logger.info(f"Found {len(model_names)} registered models")
        return model_names
