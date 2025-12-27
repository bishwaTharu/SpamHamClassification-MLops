from flask import Flask, request, jsonify
from flask_cors import CORS
import mlflow.sklearn
import pandas as pd
from src.config.settings import Settings
import os

app = Flask(__name__)
CORS(app)


if Settings.MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(Settings.MLFLOW_TRACKING_URI)

MODEL_URI = f"models:/{Settings.MODEL_NAME}/Staging"

print(f"Loading model from {MODEL_URI}...")
try:
    model = mlflow.sklearn.load_model(MODEL_URI)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ready" if model else "model_not_loaded"})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400
    
    text = data["text"]
    input_df = pd.DataFrame([text], columns=["text"])
    
    try:
        prediction = model.predict(input_df["text"])[0]
        return jsonify({
            "text": text,
            "prediction": prediction
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
