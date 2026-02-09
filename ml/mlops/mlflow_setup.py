"""
MLflow Setup and Configuration
Experiment tracking, model registry, and artifact management
"""

import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging
import os

logger = logging.getLogger(__name__)


class MLflowManager:
    """
    Centralized MLflow management for all ML models
    """
    
    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "sip_brewery_ml",
        artifact_location: Optional[str] = None
    ):
        """
        Initialize MLflow manager
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the experiment
            artifact_location: S3/local path for artifacts
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.artifact_location = artifact_location
        
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location
            )
            logger.info(f"Created experiment: {experiment_name}")
        except:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {experiment_name}")
        
        mlflow.set_experiment(experiment_name)
        
        # Initialize client
        self.client = MlflowClient(tracking_uri=tracking_uri)
    
    def start_run(
        self,
        run_name: str,
        tags: Optional[Dict[str, str]] = None
    ) -> mlflow.ActiveRun:
        """
        Start a new MLflow run
        
        Args:
            run_name: Name for the run
            tags: Optional tags for the run
        
        Returns:
            Active MLflow run
        """
        return mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=tags
        )
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(
        self,
        model: Any,
        artifact_path: str,
        model_type: str = 'pytorch',
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None,
        registered_model_name: Optional[str] = None
    ):
        """
        Log model to MLflow
        
        Args:
            model: Model to log
            artifact_path: Path within run artifacts
            model_type: 'pytorch', 'sklearn', 'tensorflow'
            signature: Model signature
            input_example: Example input
            registered_model_name: Name for model registry
        """
        if model_type == 'pytorch':
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=artifact_path,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name
            )
        elif model_type == 'sklearn':
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        logger.info(f"Model logged to {artifact_path}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact file"""
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_dict(self, dictionary: Dict, artifact_file: str):
        """Log dictionary as JSON artifact"""
        mlflow.log_dict(dictionary, artifact_file)
    
    def register_model(
        self,
        model_uri: str,
        model_name: str,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Register model in MLflow Model Registry
        
        Args:
            model_uri: URI of the model (e.g., runs:/<run_id>/model)
            model_name: Name for registered model
            tags: Optional tags
            description: Model description
        
        Returns:
            Model version
        """
        result = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            tags=tags
        )
        
        # Update description if provided
        if description:
            self.client.update_registered_model(
                name=model_name,
                description=description
            )
        
        logger.info(f"Registered model {model_name} version {result.version}")
        return result.version
    
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing: bool = True
    ):
        """
        Transition model to a different stage
        
        Args:
            model_name: Registered model name
            version: Model version
            stage: Target stage ('Staging', 'Production', 'Archived')
            archive_existing: Archive existing models in target stage
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing
        )
        logger.info(f"Transitioned {model_name} v{version} to {stage}")
    
    def load_model(self, model_uri: str, model_type: str = 'pytorch'):
        """
        Load model from MLflow
        
        Args:
            model_uri: Model URI (e.g., models:/ModelName/Production)
            model_type: Model type
        
        Returns:
            Loaded model
        """
        if model_type == 'pytorch':
            return mlflow.pytorch.load_model(model_uri)
        elif model_type == 'sklearn':
            return mlflow.sklearn.load_model(model_uri)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def get_best_run(
        self,
        metric_name: str,
        order_by: str = 'DESC'
    ) -> Dict:
        """
        Get best run based on metric
        
        Args:
            metric_name: Metric to optimize
            order_by: 'ASC' or 'DESC'
        
        Returns:
            Best run info
        """
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric_name} {order_by}"],
            max_results=1
        )
        
        if runs:
            return runs[0]
        return None
    
    def compare_runs(
        self,
        run_ids: list,
        metric_names: list
    ) -> pd.DataFrame:
        """
        Compare multiple runs
        
        Args:
            run_ids: List of run IDs to compare
            metric_names: Metrics to compare
        
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            row = {
                'run_id': run_id,
                'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
                'start_time': run.info.start_time
            }
            
            for metric in metric_names:
                row[metric] = run.data.metrics.get(metric, None)
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def delete_run(self, run_id: str):
        """Delete a run"""
        self.client.delete_run(run_id)
        logger.info(f"Deleted run {run_id}")
    
    def search_runs(
        self,
        filter_string: str = "",
        max_results: int = 100
    ) -> list:
        """
        Search runs with filter
        
        Args:
            filter_string: Filter expression (e.g., "metrics.accuracy > 0.9")
            max_results: Maximum number of results
        
        Returns:
            List of runs
        """
        return self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            max_results=max_results
        )


class ModelTracker:
    """
    Wrapper for tracking model training with MLflow
    """
    
    def __init__(self, mlflow_manager: MLflowManager):
        self.mlflow_manager = mlflow_manager
        self.current_run = None
    
    def track_training(
        self,
        model_name: str,
        model: Any,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts: Optional[Dict[str, str]] = None,
        register: bool = True,
        stage: str = 'Staging'
    ) -> str:
        """
        Track complete model training
        
        Args:
            model_name: Name of the model
            model: Trained model
            params: Training parameters
            metrics: Training metrics
            artifacts: Additional artifacts to log
            register: Whether to register model
            stage: Initial stage for registered model
        
        Returns:
            Run ID
        """
        with self.mlflow_manager.start_run(run_name=model_name) as run:
            # Log parameters
            self.mlflow_manager.log_params(params)
            
            # Log metrics
            self.mlflow_manager.log_metrics(metrics)
            
            # Log model
            registered_name = f"{model_name}_model" if register else None
            self.mlflow_manager.log_model(
                model=model,
                artifact_path="model",
                registered_model_name=registered_name
            )
            
            # Log additional artifacts
            if artifacts:
                for name, path in artifacts.items():
                    self.mlflow_manager.log_artifact(path, name)
            
            run_id = run.info.run_id
            
            # Transition to stage if registered
            if register:
                # Get latest version
                versions = self.mlflow_manager.client.search_model_versions(
                    f"name='{registered_name}'"
                )
                if versions:
                    latest_version = max([int(v.version) for v in versions])
                    self.mlflow_manager.transition_model_stage(
                        model_name=registered_name,
                        version=str(latest_version),
                        stage=stage
                    )
            
            logger.info(f"Training tracked for {model_name}, run_id: {run_id}")
            return run_id
    
    def track_experiment(
        self,
        experiment_name: str,
        model_variants: list,
        params_grid: Dict[str, list]
    ):
        """
        Track hyperparameter tuning experiment
        
        Args:
            experiment_name: Name of the experiment
            model_variants: List of model configurations
            params_grid: Parameter grid for tuning
        """
        # Implementation for grid search tracking
        pass


def setup_mlflow_server():
    """
    Setup MLflow tracking server
    
    Run this to start MLflow server:
    mlflow server --backend-store-uri postgresql://user:password@localhost/mlflow \
                   --default-artifact-root s3://bucket/mlflow-artifacts \
                   --host 0.0.0.0 \
                   --port 5000
    """
    instructions = """
    # MLflow Server Setup Instructions
    
    ## 1. Install MLflow
    pip install mlflow psycopg2-binary boto3
    
    ## 2. Setup PostgreSQL Backend (Recommended)
    # Install PostgreSQL
    # Create database:
    createdb mlflow
    
    ## 3. Setup S3 Artifact Store (Optional, for production)
    # Configure AWS credentials
    export AWS_ACCESS_KEY_ID=your_key
    export AWS_SECRET_ACCESS_KEY=your_secret
    
    ## 4. Start MLflow Server
    mlflow server \\
        --backend-store-uri postgresql://mlflow:password@localhost/mlflow \\
        --default-artifact-root s3://sip-brewery-ml/artifacts \\
        --host 0.0.0.0 \\
        --port 5000
    
    ## 5. For Development (SQLite backend)
    mlflow server \\
        --backend-store-uri sqlite:///mlflow.db \\
        --default-artifact-root ./mlruns \\
        --host 0.0.0.0 \\
        --port 5000
    
    ## 6. Access UI
    # Open browser: http://localhost:5000
    """
    
    print(instructions)
    return instructions


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Print setup instructions
    setup_mlflow_server()
    
    # Example usage
    print("\n" + "="*50)
    print("Example Usage:")
    print("="*50 + "\n")
    
    # Initialize manager
    manager = MLflowManager(
        tracking_uri="http://localhost:5000",
        experiment_name="portfolio_optimization"
    )
    
    # Example: Track a training run
    with manager.start_run(run_name="rl_portfolio_v1") as run:
        # Log parameters
        manager.log_params({
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon': 1.0,
            'batch_size': 64
        })
        
        # Simulate training metrics
        for epoch in range(10):
            manager.log_metrics({
                'train_loss': 0.5 - epoch * 0.04,
                'val_loss': 0.6 - epoch * 0.03,
                'sharpe_ratio': 0.5 + epoch * 0.1
            }, step=epoch)
        
        # Log final metrics
        manager.log_metrics({
            'final_sharpe_ratio': 1.5,
            'final_return': 0.15,
            'final_volatility': 0.10
        })
        
        print(f"Run ID: {run.info.run_id}")
        print(f"Experiment ID: {run.info.experiment_id}")
    
    print("\nMLflow setup complete!")
    print("Access UI at: http://localhost:5000")
