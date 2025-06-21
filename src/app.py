from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Define the labels directly in the app
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
import os
import time
from prometheus_client import Counter, Histogram, generate_latest
import logging
import json

# Set up a logger for query logging
query_logger = logging.getLogger('query_logger')
query_logger.setLevel(logging.INFO)
# Use a file handler to log to a file. Each log entry will be a new line.
handler = logging.FileHandler('queries.jsonl')
query_logger.addHandler(handler)

# Prometheus Metrics
REQS = Counter("tox_api_requests_total", "Total number of /predict calls")
LATENCY = Histogram("tox_api_latency_seconds", "Latency of /predict calls")

app = Flask(__name__)

# Determine if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
# Assuming the model is saved in a directory named 'model' relative to this script
# or accessible via a path defined in an environment variable.
MODEL_PATH = os.environ.get("MODEL_DIR", "model") 

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device).eval()
except Exception as e:
    # Fallback or error handling if model loading fails
    # For now, we'll print an error and exit if the model can't be loaded.
    # In a production app, you might want more sophisticated error handling.
    print(f"Error loading model from {MODEL_PATH}: {e}")
    # You could raise the exception or use a dummy model for development
    # For this example, we'll re-raise to make it clear the app can't start.
    raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    try:
        data = request.get_json()
        text = data.get('text')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            outputs = model(**inputs)
            scores = torch.sigmoid(outputs.logits)[0].cpu().tolist()
        
        predictions = dict(zip(LABELS, scores))

        # Log request details as JSON
        log_entry = {
            'timestamp': time.time(),
            'ip_address': request.remote_addr,
            'query': text,
            'prediction': predictions
        }
        query_logger.info(json.dumps(log_entry))

        # Record latency and increment request counter
        LATENCY.observe(time.time() - start_time)
        REQS.inc()
        return jsonify(predictions)
    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return jsonify({'error': 'Error processing request', 'details': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    start_time = time.time()
    try:
        data = request.get_json()
        text1 = data.get('text1')
        text2 = data.get('text2')

        if not text1 or not text2:
            return jsonify({'error': 'Two text fields are required'}), 400

        with torch.no_grad():
            inputs1 = tokenizer(text1, return_tensors="pt", truncation=True, max_length=512).to(device)
            outputs1 = model(**inputs1)
            scores1 = torch.sigmoid(outputs1.logits)[0].cpu().tolist()
            prediction1 = dict(zip(LABELS, scores1))

            inputs2 = tokenizer(text2, return_tensors="pt", truncation=True, max_length=512).to(device)
            outputs2 = model(**inputs2)
            scores2 = torch.sigmoid(outputs2.logits)[0].cpu().tolist()
            prediction2 = dict(zip(LABELS, scores2))

        # Log comparison request
        log_entry = {
            'timestamp': time.time(),
            'ip_address': request.remote_addr,
            'request_type': 'comparison',
            'queries': [text1, text2],
            'predictions': [prediction1, prediction2]
        }
        query_logger.info(json.dumps(log_entry))

        # Record latency and increment request counter for batch endpoint
        LATENCY.observe(time.time() - start_time)
        REQS.inc(2) # Increment by 2 for two predictions

        return jsonify({'prediction1': prediction1, 'prediction2': prediction2})

    except Exception as e:
        app.logger.error(f"Error during batch prediction: {e}")
        return jsonify({'error': 'Error processing request', 'details': str(e)}), 500

@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': 'text/plain; version=0.0.4'}

if __name__ == '__main__':
    # Make sure to create a 'templates' folder in the same directory as app.py
    # and place index.html inside it.
    # The MODEL_DIR environment variable can be set to point to your model directory.
    # Example: MODEL_DIR=../model_output python app.py
    # Ensure the model directory 'model' (or the one specified by MODEL_DIR) 
    # contains the necessary model files (pytorch_model.bin, config.json, etc.)
    app.run(debug=True, host='0.0.0.0', port=5000)
