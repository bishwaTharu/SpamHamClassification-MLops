import subprocess
import sys
from src.utils.logger import get_logger

logger = get_logger(__name__)

def build_and_push():
    """
    Automates the MLflow SageMaker container build and push process.
    Equivalent to: mlflow sagemaker build-and-push-container
    """
    logger.info("Starting MLflow SageMaker container build and push process...")
    
    try:
        # Construct the command
        # You can add --container-env if needed
        command = [
            "mlflow", "sagemaker", "build-and-push-container"
        ]
        
        logger.info(f"Running command: {' '.join(command)}")
        
        # Execute the command
        # Note: This requires Docker to be running and AWS CLI to be configured.
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info("Container build and push successful!")
        logger.info(f"Command output: {result.stdout}")
        
    except subprocess.CalledProcessError as e:
        logger.error("Failed to build/push container.")
        logger.error(f"Error output: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    build_and_push()
