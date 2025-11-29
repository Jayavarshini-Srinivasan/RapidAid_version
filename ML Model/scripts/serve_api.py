import os
import json
import numpy as np
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

def _load(path):
    try:
        return joblib.load(path)
    except Exception:
        return None

MODEL_PATHS = {
    "random_forest": os.environ.get("RF_MODEL_PATH", "models/random_forest_model.pkl"),
    "gradient_boosting": os.environ.get("GB_MODEL_PATH", "models/gradient_boosting_model.pkl"),
    "accelerometer": os.environ.get("ACC_MODEL_PATH", "models/accelerometer_accident_detector.pkl"),
}

MODELS = {}
for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        m = _load(path)
        if m is not None:
            MODELS[name] = m

@app.route("/", methods=["GET"]) 
def index():
    return jsonify({
        "status": "ok",
        "message": "COE model API",
        "endpoints": ["/health", "/predict", "/predict_batch", "/predictions"],
        "models": list(MODELS.keys()),
    })

@app.route("/health", methods=["GET"]) 
def health():
    return jsonify({"status": "healthy", "models": list(MODELS.keys())})

@app.route("/predict", methods=["POST"]) 
def predict():
    data = request.get_json() or {}
    model_name = data.get("model")
    features = data.get("features")
    if not model_name or model_name not in MODELS:
        return jsonify({"error": "model not available"}), 400
    if not isinstance(features, list):
        return jsonify({"error": "features must be a list"}), 400
    X = np.array([features], dtype=float)
    m = MODELS[model_name]
    y = m.predict(X)
    resp = {"prediction": y[0]}
    if hasattr(m, "predict_proba"):
        p = m.predict_proba(X)
        resp["probabilities"] = p[0].tolist()
    return jsonify(resp)

@app.route("/predict_batch", methods=["POST"]) 
def predict_batch():
    data = request.get_json() or {}
    model_name = data.get("model")
    records = data.get("records")
    if not model_name or model_name not in MODELS:
        return jsonify({"error": "model not available"}), 400
    if not isinstance(records, list):
        return jsonify({"error": "records must be a list"}), 400
    X = np.array(records, dtype=float)
    m = MODELS[model_name]
    y = m.predict(X).tolist()
    resp = {"predictions": y}
    if hasattr(m, "predict_proba"):
        p = m.predict_proba(X).tolist()
        resp["probabilities"] = p
    return jsonify(resp)

@app.route("/predictions", methods=["GET"]) 
def get_predictions():
    """Serve predictions.json data from the output directory"""
    try:
        predictions_path = os.path.join(os.path.dirname(__file__), "..", "output", "predictions.json")
        with open(predictions_path, 'r') as f:
            predictions_data = json.load(f)
        return jsonify(predictions_data)
    except FileNotFoundError:
        return jsonify({"error": "predictions.json file not found"}), 404
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON format in predictions.json"}), 500
    except Exception as e:
        return jsonify({"error": f"Error reading predictions file: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, threaded=True)