from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI")
    EXPERIMENT_NAME: str = os.getenv("EXPERIMENT_NAME", "spam-ham-classifier")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "SpamHamClassifier")

    RAW_DATA_PATH: str = os.getenv("RAW_DATA_PATH")
    PROCESSED_DATA_BUCKET: str = os.getenv("PROCESSED_DATA_BUCKET")

    RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", "42"))
    F1_THRESHOLD: float = float(os.getenv("F1_THRESHOLD", "0.85"))

    # SageMaker Deployment Settings
    SAGEMAKER_ROLE_ARN: str = os.getenv("SAGEMAKER_ROLE_ARN")
    REGION_NAME: str = os.getenv("REGION_NAME", "ap-southeast-2")
    ENDPOINT_NAME: str = os.getenv("ENDPOINT_NAME", "spam-ham-classifier-endpoint")
    INSTANCE_TYPE: str = os.getenv("INSTANCE_TYPE", "ml.m5.large")
