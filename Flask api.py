from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('xgb_stress_model.pkl')
scaler = joblib.load('stress_scaler.pkl')

app = Flask(__name__)
CORS(app)  # Enable CORS

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        # Print incoming data for debugging
        print("Received:", data)

        # Extract features
        features = [
            data['AcademicWorkload'],
            data['SleepQuality'],
            data['FinancialStrain'],
            data['SocialSupport'],
            data['AnxietyLevel']
        ]

        # Preprocess and predict
        scaled = scaler.transform([features])
        prediction = model.predict(scaled)
        stress_level = int(prediction[0])

        # Stress level messages
        messages = {
            0: "Low Stress 🌿 – Keep it up!",
            1: "Moderate Stress ⚖️ – Stay balanced!",
            2: "High Stress 🔥 – Take care, consider relaxing activities."
        }

        message = f"Stress level is: {messages.get(stress_level, 'Unknown')}"
        print("Prediction:", message)

        return jsonify({
            "stress_level": stress_level,
            "message": message
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/')
def home():
    return "Stress Predictor API Running"

if __name__ == '__main__':
    app.run(debug=True, port=5000)
