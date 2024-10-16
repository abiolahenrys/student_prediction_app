import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Absolute paths to the model and transformations
model_path = '/Users/mac/Desktop/my_flask_app/models/rf_reg_model.pkl'
scaler_path = '/Users/mac/Desktop/my_flask_app/models/scaler.pkl'
poly_path = '/Users/mac/Desktop/my_flask_app/models/poly_features.pkl'

# Load the model, scaler, and polynomial features
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    poly = joblib.load(poly_path)
except Exception as e:
    print(f"Error loading model, scaler, or polynomial features: {e}")
    raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/student_info')
def student_info():
    return render_template('student_info.html')

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/guideline')
def guideline():
    return render_template('guideline.html')

@app.route('/study_mat')
def study_mat():
    return render_template('study_mat.html')

@app.route('/tips')
def tips():
    return render_template('tips.html')

@app.route('/subject')
def subject():
    subject = request.args.get('subject')
    if subject:
        return render_template('subject.html', subject=subject)
    else:
        return "Subject not specified", 400


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract and convert form data
        try:
            grade_avg = float(request.form.get('grade_avg', 0))
            past_failures = int(request.form.get('failures', 0))
            romantic_relationship = int(request.form.get('romantic_relationship', 0))
            family_relationship = int(request.form.get('family_relationship', 0))
            studytime = float(request.form.get('studytime', 0))
            health = float(request.form.get('health', 0))
            fathers_education = float(request.form.get('fathers_education', 0))
            mothers_education = float(request.form.get('mothers_education', 0))
            free_time = float(request.form.get('free_time', 0))
            travel_time = float(request.form.get('travel_time', 0))
            go_out = float(request.form.get('go_out', 0))
        except ValueError as e:
            print(f"Error converting form data: {e}")
            return "Invalid input data."

        # Prepare feature array
        features = pd.DataFrame({
            'grade_avg': [grade_avg],
            'failures': [past_failures],
            'romantic_relationship': [romantic_relationship],
            'family_relationship': [family_relationship],
            'studytime': [studytime],
            'health': [health],
            'fathers_education': [fathers_education],
            'mothers_education': [mothers_education],
            'free_time': [free_time],
            'travel_time': [travel_time],
            'go_out': [go_out]
        })

        # Polynomial features transformation
        try:
            features_poly = poly.transform(features)
            feature_names = poly.get_feature_names_out(features.columns)
            features_poly_df = pd.DataFrame(features_poly, columns=feature_names)
        except Exception as e:
            print(f"Error during polynomial transformation or DataFrame creation: {e}")
            return "Error in polynomial feature transformation."

        # Apply scaling
        try:
            features_scaled = scaler.transform(features_poly_df)
        except Exception as e:
            print(f"Error during scaling: {e}")
            return "Error in feature scaling."

        # Make prediction
        try:
            prediction = model.predict(features_scaled)
            prediction_value = prediction[0]
        except Exception as e:
            print(f"Error during prediction: {e}")
            return "Error in prediction."

        # Output prediction
        print(f"Prediction: {prediction_value}")
        return render_template('result.html', prediction=prediction_value)

    except Exception as e:
        print(f"Overall Error: {e}")
        return "An error occurred during prediction."

if __name__ == '__main__':
    print("Starting Flask application...")
    try:
        app.run(debug=True)
    except Exception as e:
        print(f"Error starting Flask application: {e}")
