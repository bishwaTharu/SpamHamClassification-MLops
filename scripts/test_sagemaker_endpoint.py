import boto3
import json
from src.config.settings import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

def test_endpoint(text: str):
    """
    Sends a sample text to the SageMaker endpoint and prints the response.
    """
    client = boto3.client("sagemaker-runtime", region_name=Settings.REGION_NAME)
    endpoint_name = Settings.ENDPOINT_NAME
    
    # MLflow 2.0+ expected format: {"dataframe_records": [{"column1": value1}, ...]}
    payload = {"dataframe_records": [{"text": text}]}
    
    logger.info(f"Invoking endpoint {endpoint_name} with text: {text}")
    
    try:
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload)
        )
        
        result = json.loads(response["Body"].read().decode())
        logger.info(f"Prediction result: {result}")
        print(f"Response: {result}")
        
    except Exception as e:
        logger.error(f"Failed to invoke endpoint: {str(e)}")
        print(f"Error: {e}")

if __name__ == "__main__":
    test_texts = [
        "I am not doing great, how about you?"
    ]
    
    for text in test_texts:
        test_endpoint(text)
