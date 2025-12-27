from mlflow.tracking import MlflowClient
from src.config.settings import Settings

class ModelPromoter:
    def __init__(self):
        self.client = MlflowClient()

    def promote_if_valid(self, f1_score: float):
        if f1_score < Settings.F1_THRESHOLD:
            raise ValueError("Model does not meet quality threshold")

        versions = self.client.get_latest_versions(
            Settings.MODEL_NAME, stages=["None"]
        )

        if not versions:
            print(f"No versions of {Settings.MODEL_NAME} found in stage 'None'.")
            return

        latest = versions[0]
        print(f"Promoting model version {latest.version} to Staging...")

        self.client.transition_model_version_stage(
            name=Settings.MODEL_NAME,
            version=latest.version,
            stage="Staging",
            archive_existing_versions=True
        )
