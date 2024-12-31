from flask import Flask, render_template, request
import numpy as np
import xgboost as xgb
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("best_xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Retrieve input values from the form
            inputs = [
                float(request.form['power_generation']),
                float(request.form['gdp']),
                float(request.form['industrial_production']),
                float(request.form['agriculture_production']),
                float(request.form['month']),
                float(request.form['renewable_energy'])
            ]

            # Preprocess inputs
            inputs = np.array(inputs).reshape(1, -1)
            inputs_scaled = scaler.transform(inputs)

            # Make prediction
            prediction = model.predict(inputs_scaled)[0]

            # Render the prediction
            return render_template('index.html', prediction=f"Predicted Demand: {prediction:.2f} Million Metric Tons")
        except Exception as e:
            return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)