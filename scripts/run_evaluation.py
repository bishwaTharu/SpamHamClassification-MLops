import pandas as pd
import mlflow
from src.pipelines.evaluation_pipeline import EvaluationPipeline
from src.utils.mlflow_manager import MLflowManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main(data_path: str):
    df = pd.read_parquet(data_path)
    y_true = df["label"]
    y_pred = df["prediction"]

    mlflow_manager = MLflowManager()

    with mlflow_manager.start_run(run_name="evaluation"):
        pipeline = EvaluationPipeline()
        f1 = pipeline.evaluate_and_promote(y_true, y_pred)

        mlflow.log_metric("evaluation_f1", f1)

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
