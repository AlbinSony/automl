from flask import Flask, request, jsonify, render_template
from google.cloud import storage, aiplatform
from google.cloud.aiplatform import AutoMLTabularTrainingJob  # Import the correct class
import os
import time
import pandas as pd

app = Flask(__name__)

# Google Cloud configurations
PROJECT_ID = "easyml-448708"
REGION = "us-central1"
BUCKET_NAME = "easyml_bucket"

# Set environment credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/ALBIN SONY/Downloads/easyml-448708-5b7d2498af17.json"

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION)

# Clients
storage_client = storage.Client()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if file and file.filename.endswith('.csv'):
            os.makedirs("uploads", exist_ok=True)
            file_path = os.path.join("uploads", file.filename)
            file.save(file_path)

            # Upload file to Google Cloud Storage
            bucket = storage_client.bucket(BUCKET_NAME)
            blob = bucket.blob(file.filename)
            blob.upload_from_filename(file_path)
            gcs_uri = f"gs://{BUCKET_NAME}/{file.filename}"

            return jsonify({"message": "File uploaded successfully", "gcs_uri": gcs_uri}), 200
        else:
            return jsonify({"error": "Invalid file type"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/train_model", methods=["POST"])
def train_model():
    try:
        data = request.json
        gcs_uri = data.get("gcs_uri")
        target_column = data.get("target_column")

        if not gcs_uri or not target_column:
            return jsonify({"error": "GCS URI and target column are required"}), 400

        prediction_type = "classification"  # or "regression"

        # Create a training job
        training_job = AutoMLTabularTrainingJob(
            display_name=f"model_{int(time.time())}",
            optimization_prediction_type=prediction_type,
            optimization_objective='maximize-au-prc'
        )

        # Run training
        model = training_job.run(
            dataset=aiplatform.TabularDataset.create(
                display_name=f"dataset_{int(time.time())}",
                gcs_source=[gcs_uri]
            ),
            target_column=target_column,
            model_display_name=f"trained_model_{int(time.time())}",
            training_fraction_split=0.8,
            validation_fraction_split=0.1,
            test_fraction_split=0.1
        )

        return jsonify({
            "message": "Model training started", 
            "model_name": model.resource_name
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    model_id = data.get("model_id")
    input_data = data.get("input_data")

    if not model_id or not input_data:
        return jsonify({"error": "Model ID and input data are required"}), 400

    try:
        # Predict using the model
        model_full_id = f"projects/{PROJECT_ID}/locations/{COMPUTE_REGION}/models/{model_id}"
        prediction_client = automl.PredictionServiceClient()
        inputs = {"feature_1": input_data["feature_1"], "feature_2": input_data["feature_2"]}  # Replace with actual fields

        # Format input as a JSON row
        payload = {"row": inputs}
        response = prediction_client.predict(name=model_full_id, payload=payload)

        predictions = [result.payload[0].display_name for result in response.payload]
        return jsonify({"predictions": predictions}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
