from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import logging

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Load model and scaler
model = joblib.load('Xgb_HeartDisease_model.pkl')


# Feature mapping
FEATURE_MAPPING = {
    'Sex': {'M': 0, 'F': 1},
    'ChestPainType': {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3},
    'RestingECG': {'Normal': 0, 'ST': 1, 'LVH': 2},
    'ExerciseAngina': {'N': 0, 'Y': 1},
    'ST_Slope': {'Up': 0, 'Flat': 1, 'Down': 2}
}

@app.route('/')
def home():
    return render_template("index1.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Prepare input array in correct order
        input_data = np.array([
            float(data['Age']),
            FEATURE_MAPPING['Sex'].get(data['Sex'], 0),
            FEATURE_MAPPING['ChestPainType'].get(data['ChestPainType'], 0),
            float(data['RestingBP']),
            float(data['Cholesterol']),
            float(data['FastingBS']),
            FEATURE_MAPPING['RestingECG'].get(data['RestingECG'], 0),
            float(data['MaxHR']),
            FEATURE_MAPPING['ExerciseAngina'].get(data['ExerciseAngina'], 0),
            float(data['Oldpeak']),
            FEATURE_MAPPING['ST_Slope'].get(data['ST_Slope'], 0)
        ]).reshape(1, -1)
        
        # Scale numerical features
        numerical_indices = [0, 3, 4, 7, 9]
        input_data[:, numerical_indices] = scaler.transform(input_data[:, numerical_indices])
        
        # Make prediction
        proba = model.predict_proba(input_data)[0]
        prediction = model.predict(input_data)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(proba[1]),
            'status': 'success'
        })
    
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e), 'status': 'error'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)