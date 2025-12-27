import mlflow
from src.config.settings import Settings

class MLflowManager:
    def __init__(self):
        mlflow.set_tracking_uri(Settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(Settings.EXPERIMENT_NAME)

    def start_run(self, run_name: str):
        return mlflow.start_run(run_name=run_name)

    @staticmethod
    def log_params(params: dict):
        mlflow.log_params(params)

    @staticmethod
    def log_metrics(metrics: dict):
        mlflow.log_metrics(metrics)
