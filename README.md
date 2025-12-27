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
- `src/api/`: Flask prediction server.
- `src/pipelines/`: Logic for ETL, Training, and Evaluation.
- `src/monitoring/`: Data drift detection tools.
- `src/registry/`: MLflow model promotion logic.
- `src/config/`: Centralized project settings.