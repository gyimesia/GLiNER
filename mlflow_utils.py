import os
import mlflow
from mlflow.tracking import MlflowClient
import subprocess
from pathlib import Path


class ModelLogger:
    """
    Handles logging models, parameters, metrics, and artifacts using MLflow with DVC as backend storage.
    """

    def __init__(self, experiment_name, tracking_uri=None, dvc_remote=None):
        """
        Initialize the model logger.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (if None, will use local)
            dvc_remote: DVC remote name (if None, will use default)
        """
        # Set tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Get or create experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = experiment.experiment_id
            
        self.client = MlflowClient()
        self.dvc_remote = dvc_remote
        self.active_run = None
    
    def start_run(self, run_name=None):
        """Start an MLflow run"""
        self.active_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name
        )
        return self.active_run.info.run_id
    
    def end_run(self):
        """End the current MLflow run"""
        if self.active_run:
            mlflow.end_run()
            self.active_run = None
    
    def log_params(self, params):
        """Log parameters to MLflow"""
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics, step=None):
        """Log metrics to MLflow"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_model(self, model, artifact_path, model_path):
        """
        Log model to MLflow and track with DVC
        
        Args:
            model: Model object to log
            artifact_path: Path within the MLflow run artifacts
            model_path: Local path where model will be saved
        """
        # Save model using MLflow
        mlflow.pytorch.log_model(model, artifact_path)
        
        # Track model with DVC
        model_dir = Path(model_path).parent
        if not model_dir.exists():
            model_dir.mkdir(parents=True, exist_ok=True)
            
        # Save model locally
        mlflow.pytorch.save_model(model, model_path)
        
        # Add to DVC
        try:
            subprocess.run(["dvc", "add", model_path], check=True)
            
            # Push to remote if specified
            if self.dvc_remote:
                subprocess.run(["dvc", "push", model_path, "--remote", self.dvc_remote], check=True)
                
            # Log DVC path in MLflow
            mlflow.log_param("dvc_model_path", model_path)
        except subprocess.CalledProcessError as e:
            print(f"DVC operation failed: {e}")
    
    def log_artifact(self, local_path):
        """Log an artifact to MLflow"""
        mlflow.log_artifact(local_path)
    
    def set_tags(self, tags):
        """Set tags for the current run"""
        for key, value in tags.items():
            mlflow.set_tag(key, value)
