import pandas as pd
import mlflow
from pathlib import Path
from src.data.data_versioning import DataVersioner
from src.config.settings import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ETLPipeline:
    def extract(self) -> pd.DataFrame:
        """extracts data from the source S3 bucket."""
        logger.info(f"Extracting data from {Settings.RAW_DATA_PATH}")
        try:
            # Pandas supports s3:// URLs properly with s3fs installed
            df = pd.read_csv(Settings.RAW_DATA_PATH, encoding='latin-1')
            logger.info(f"Extracted {len(df)} rows.")
            return df
        except Exception as e:
            logger.error(f"Failed to extract data: {e}")
            raise

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans and transforms the raw data."""
        logger.info("Transforming data...")
        
        # Standardize column names
        df = df.rename(columns={'v1': 'label', 'v2': 'text'})
        
        # Check required columns
        required_cols = ['label', 'text']
        if not all(col in df.columns for col in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            raise ValueError(f"Data missing required columns: {missing}. Available: {df.columns}")

        # Basic Cleaning
        df = df[required_cols]
        
        initial_count = len(df)
        df = df.dropna().drop_duplicates()
        dropped_count = initial_count - len(df)
        if dropped_count > 0:
            logger.info(f"Dropped {dropped_count} rows (duplicates/NaNs).")

        # Text normalization
        df['text'] = df['text'].astype(str).str.lower().str.strip()
        
        # Validate labels
        valid_labels = {'ham', 'spam'}
        invalid_mask = ~df['label'].isin(valid_labels)
        if invalid_mask.any():
            logger.warning(f"Found {invalid_mask.sum()} rows with invalid labels. Dropping them.")
            df = df[~invalid_mask]
        
        logger.info(f"Transformed data shape: {df.shape}")
        return df

    def load(self, df: pd.DataFrame) -> str:
        """Loads processed data to the destination S3 bucket (or path)."""
        dataset_hash = DataVersioner.compute_hash(df)
        
        # Construct output path with versioning
        base_path = Settings.PROCESSED_DATA_BUCKET.rstrip('/')
        output_dir = f"{base_path}/{dataset_hash}"
        output_path = f"{output_dir}/data.parquet"
        
        logger.info(f"Loading data to {output_path}")
        try:
            # Ensure directory exists if it's local (not s3://)
            if not output_path.startswith("s3://"):
                Path(output_dir).mkdir(parents=True, exist_ok=True)

            df.to_parquet(output_path, index=False)
            
            # Log to MLflow
            mlflow.log_param("dataset_version", dataset_hash)
            mlflow.log_param("processed_rows", len(df))
            mlflow.log_param("output_path", output_path)
            
            return output_path
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def run(self) -> str:
        """Orchestrates the ETL pipeline."""
        logger.info("Starting ETL Pipeline")
        df = self.extract()
        df = self.transform(df)

        # Monitoring: Check for text length drift
        from src.monitoring.drift import DataDriftMonitor
        # Baseline: 80 characters (arbitrary for ham/spam dataset)
        monitor = DataDriftMonitor(baseline_mean=80.0)
        has_drift = monitor.check_text_length_drift(df['text'].tolist())
        
        mlflow.log_metric("text_length_drift_detected", int(has_drift))

        output_path = self.load(df)
        logger.info(f"ETL Pipeline completed. Output: {output_path}")
        return output_path

if __name__ == "__main__":
    pipeline = ETLPipeline()
    pipeline.run()
