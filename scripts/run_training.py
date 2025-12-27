import mlflow
import subprocess
from src.pipelines.training_pipeline import TrainingPipeline
from src.utils.mlflow_manager import MLflowManager
from src.utils.logger import get_logger
import sys

logger = get_logger(__name__)

def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        return "unknown"

def main(data_path: str):
    mlflow_manager = MLflowManager()

    with mlflow_manager.start_run(run_name="training"):
        mlflow.log_param("git_commit", get_git_commit())

        pipeline = TrainingPipeline()
        f1, output_path = pipeline.run(data_path)

        logger.info(f"Training completed. F1: {f1}")
        logger.info(f"Predictions saved to: {output_path}")
        print(output_path) # For external capture

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_training.py <s3_data_path>")
        sys.exit(1)
    main(sys.argv[1])
