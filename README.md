# Spam-Ham MLOps Project

This project implements an end-to-end MLOps pipeline for classifying SMS messages as Spam or Ham. It includes data extraction from S3, model training with MLflow tracking, model promotion to a registry, drift monitoring, and a Flask API for serving predictions.

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.10+**
- **AWS Credentials**: Configured via `aws configure` (for S3 access).
- **MLflow**: Make sure an MLflow tracking server is accessible or use the local default.

### 2. Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd spam-ham-mlops-demo

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set Environment Variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## 🛠️ Pipeline Execution

### Step 1: Data Processing (ETL)
Extracts raw data from S3, cleans it, checks for **data drift**, and saves versioned Parquet files back to S3.
```bash
python scripts/run_etl.py
```
*Note: Copy the S3 output path printed in the terminal.*

### Step 2: Model Training
Trains a Tfidf + LogisticRegression pipeline and logs artifacts/metrics to MLflow.
```bash
python scripts/run_training.py <PASTE_ETL_S3_PATH>
```
*Note: Copy the `_with_preds.parquet` S3 path.*

### Step 3: Model Evaluation & Promotion
Evaluates the model against a threshold (F1 > 0.85) and promotes it to the `Staging` stage in the MLflow Model Registry.
```bash
python scripts/run_evaluation.py <PASTE_TRAINING_S3_PATH>
```

## 🌐 Prediction API

Run the Flask server to start making real-time predictions using the `Staging` model.
```bash
python src/api/app.py
```

### Test the endpoint
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Urgent! You have won a 1-week holiday to Hawaii. Call 0800-spam now!"}'
```

## ☁️ SageMaker Deployment

### 1. Configuration
Ensure your `.env` file has the following settings:
```env
SAGEMAKER_ROLE_ARN=arn:aws:iam::your-account:role/service-role/AmazonSageMaker-ExecutionRole
REGION_NAME=ap-southeast-2
INSTANCE_TYPE=ml.m5.large  # Smallest reliable general-purpose instance
ENDPOINT_NAME=spam-ham-classifier-endpoint
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

### 2. Prepare the Model
Ensure your model is in the `Staging` stage of the MLflow Model Registry.
```bash
# Verify the registry status
python scripts/check_registry.py

# If version 1 is not in Staging, promote it:
python -c "from mlflow.tracking import MlflowClient; from src.config.settings import Settings; client = MlflowClient(); client.transition_model_version_stage(name=Settings.MODEL_NAME, version=1, stage='Staging')"
```

### 3. Build & Push Container
MLflow requires a Docker container in ECR for SageMaker deployment.
```bash
mlflow sagemaker build-and-push-container
```

### 4. Deploy & Test
```bash
# Run the deployment script
python scripts/deploy_to_sagemaker.py

# Test the remote endpoint (uses MLflow 2.0+ scoring protocol)
python scripts/test_sagemaker_endpoint.py
```

## 🛠️ Troubleshooting

- **Connection Errors**: Ensure the MLflow server is running (`mlflow ui --port 5000`) if using a networked URI.
- **Model Registry Issues**: Use `python scripts/check_registry.py` to diagnose missing versions or stages.
- **Scoring Protocol (400 Bad Request)**: MLflow 2.0+ requires inputs in a specific JSON format (e.g., `dataframe_records`). Ensure your client uses the structure: `{"dataframe_records": [{"text": "your message"}]}`.
- **Dependency Issues**: If you see `Not supported URL scheme http+docker`, ensure you have `requests<2.32` installed.

## 🧹 Cleanup

To avoid ongoing AWS costs when you are done testing, you must delete the SageMaker endpoint:
```bash
python scripts/delete_sagemaker_endpoint.py
```
This script will stop the instance and remove the endpoint from your AWS account.

## 🐳 Docker Support

Build and run the entire pipeline or API inside Docker.

```bash
# Build the image
docker build -t spam-ham-mlops .

# Run a specific script (e.g., ETL)
docker run -e AWS_ACCESS_KEY_ID=xxx -e AWS_SECRET_ACCESS_KEY=xxx spam-ham-mlops scripts/run_etl.py
```

## 📊 Monitoring & Registry
- **Drift Monitoring**: Automated text length drift detection integrated into the ETL pipeline.
- **Model Registry**: Automated promotion of models meeting quality thresholds via `ModelPromoter`.
- **Git Integration**: Graceful handling of environments with or without Git metadata.

## 📁 Project Structure
- `scripts/`: Entry point scripts for pipeline stages.
  - `run_etl.py`, `run_training.py`, `run_evaluation.py` (Core pipeline)
  - `deploy_to_sagemaker.py`, `test_sagemaker_endpoint.py` (SageMaker deployment)
  - `check_registry.py` (Model registry diagnostic)
  - `delete_sagemaker_endpoint.py` (Cleanup/Stop instance)
- `src/api/`: Flask prediction server.
- `src/pipelines/`: Logic for ETL, Training, and Evaluation.
- `src/monitoring/`: Data drift detection tools.
- `src/registry/`: MLflow model promotion logic.
- `src/config/`: Centralized project settings.