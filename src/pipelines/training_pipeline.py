import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score
from mlflow.models.signature import infer_signature

from src.models.pipeline import SpamHamPipeline
from src.config.settings import Settings

class TrainingPipeline:
    def run(self, data_path: str) -> tuple[float, str]:
        df = pd.read_parquet(data_path)

        X_text = df["text"]         
        y = df["label"]

        pipeline = SpamHamPipeline.build(Settings.RANDOM_STATE)
        pipeline.fit(X_text, y)

        preds = pipeline.predict(X_text)
        df["prediction"] = preds
        
        f1 = f1_score(y, preds, pos_label='spam')
        
        # Save predictions
        output_path = data_path.replace(".parquet", "_with_preds.parquet")
        df.to_parquet(output_path)


        signature = infer_signature(
            model_input=X_text.to_frame(name="text"),
            model_output=preds
        )

        input_example = X_text.iloc[:5].to_frame(name="text")

        mlflow.log_metric("f1_score", f1)
        mlflow.log_param("model_type", "tfidf_logreg")
        mlflow.log_param("predictions_output_path", output_path)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=Settings.MODEL_NAME,
            signature=signature,
            input_example=input_example
        )

        return f1, output_path
