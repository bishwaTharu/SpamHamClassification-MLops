import mlflow
from src.pipelines.etl_pipeline import ETLPipeline
from src.utils.mlflow_manager import MLflowManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    mlflow_manager = MLflowManager()

    with mlflow_manager.start_run(run_name="etl"):
        pipeline = ETLPipeline()
        output_path = pipeline.run()
        logger.info(f"ETL completed. Output: {output_path}")
        print(output_path) # For external capture

if __name__ == "__main__":
    main()
