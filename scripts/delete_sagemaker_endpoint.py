import boto3
from src.config.settings import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

def delete_endpoint():
    """
    Deletes the SageMaker endpoint and its configuration.
    """
    client = boto3.client("sagemaker", region_name=Settings.REGION_NAME)
    endpoint_name = Settings.ENDPOINT_NAME
    
    logger.info(f"Attempting to delete SageMaker endpoint: {endpoint_name}")
    
    try:
        # First, delete the endpoint
        client.delete_endpoint(EndpointName=endpoint_name)
        logger.info(f"✅ Successfully deleted endpoint: {endpoint_name}")
        
        # Second, delete the endpoint configuration
        # Note: MLflow typically names the config the same as the endpoint
        try:
            client.delete_endpoint_config(EndpointConfigName=endpoint_name)
            logger.info(f"✅ Successfully deleted endpoint configuration: {endpoint_name}")
        except Exception as e:
            logger.warning(f"Could not delete config (it might have a different name): {str(e)}")

        print("\nCleanup Complete: The SageMaker instance has been stopped and deleted.")

    except Exception as e:
        logger.error(f"Failed to delete endpoint: {str(e)}")
        print(f"Error: {e}")

if __name__ == "__main__":
    delete_endpoint()
