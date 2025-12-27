from dataclasses import dataclass
import os

@dataclass(frozen=True)
class Settings:
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI")
    EXPERIMENT_NAME: str = "spam-ham-classifier"
    MODEL_NAME: str = "SpamHamClassifier"

    RAW_DATA_PATH: str = "s3://spam-ham-ml-data-0022/data/spam (1).csv"
    PROCESSED_DATA_BUCKET: str = "s3://spam-ham-ml-data-0022/data/processed/"

    RANDOM_STATE: int = 42
    F1_THRESHOLD: float = 0.85
