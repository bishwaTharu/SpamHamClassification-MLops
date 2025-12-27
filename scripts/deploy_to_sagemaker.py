from mlflow.deployments import get_deploy_client
from src.config.settings import Settings
from src.utils.logger import get_logger
import os

logger = get_logger(__name__)

def deploy():
    """
    Deploys the latest Staging model to SageMaker using the modern mlflow.deployments API.
    """
    model_name = Settings.MODEL_NAME
    model_uri = f"models:/{model_name}/Staging"
    
    # SageMaker config
    app_name = Settings.ENDPOINT_NAME
    region_name = Settings.REGION_NAME
    execution_role_arn = Settings.SAGEMAKER_ROLE_ARN
    instance_type = Settings.INSTANCE_TYPE
    
    logger.info(f"Deploying model {model_uri} to SageMaker endpoint {app_name} in {region_name}...")
    
    try:
        # Get the SageMaker deployment client
        client = get_deploy_client("sagemaker")
        
        # Define the configuration for the deployment
        # Note: MLflow's SageMaker deployment client uses this config dictionary
        config = {
            "region_name": region_name,
            "execution_role_arn": execution_role_arn,
            "instance_type": instance_type,
            "mode": "create" # Can also be "replace" or "add"
        }
        
        # Create the deployment
        client.create_deployment(
            name=app_name,
            model_uri=model_uri,
            flavor="python_function",
            config=config
        )
        
        logger.info(f"Successfully deployed model to {app_name}")
        
    except Exception as e:
        logger.error(f"Failed to deploy model: {str(e)}")
        raise e

if __name__ == "__main__":
    deploy()
