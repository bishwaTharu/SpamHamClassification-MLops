from mlflow.tracking import MlflowClient
from src.config.settings import Settings
import mlflow

def check_registry():
    mlflow.set_tracking_uri(Settings.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    print(f"Checking MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    
    try:
        models = client.search_registered_models(filter_string=f"name='{Settings.MODEL_NAME}'")
        if not models:
            print(f"❌ Error: Model '{Settings.MODEL_NAME}' is NOT registered at all.")
            print("Action: You must run 'python scripts/run_training.py <data_path>' first.")
            return

        print(f"✅ Found model '{Settings.MODEL_NAME}'.")
        
        versions = client.get_latest_versions(Settings.MODEL_NAME)
        if not versions:
            print("❌ Error: No versions found for this model.")
            return

        print("\nVersions found:")
        for v in versions:
            print(f" - Version: {v.version}, Stage: {v.current_stage}, RunID: {v.run_id}")

        staging_versions = [v for v in versions if v.current_stage == "Staging"]
        if not staging_versions:
            print("\n❌ Error: No version is in 'Staging'.")
            print("Action: You must run 'python scripts/run_evaluation.py <pred_path>' to promote it.")
        else:
            print(f"\n✅ Ready for deployment! Version {staging_versions[0].version} is in 'Staging'.")

    except Exception as e:
        print(f"❌ Connection Error: {str(e)}")

if __name__ == "__main__":
    check_registry()
