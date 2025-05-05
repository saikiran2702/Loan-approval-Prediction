from flask import Flask, request, render_template
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the trained Decision Tree model
model_path = os.path.join('model', 'decision_tree_model.pkl')

# Check if the model file exists
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("Model loaded successfully!")
else:
    print("Model file not found!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect data from form
    education = request.form['education']
    self_employed = request.form['self_employed']
    no_of_dependents = int(request.form['no_of_dependents'])
    income_annum = float(request.form['income_annum'])
    loan_amount = float(request.form['loan_amount'])
    loan_term = float(request.form['loan_term'])
    cibil_score = float(request.form['cibil_score'])
    assets = float(request.form['assets'])

    # Log the form data for debugging
    print(f"Education: {education}, Self Employed: {self_employed}, No. of Dependents: {no_of_dependents}, "
          f"Income: {income_annum}, Loan Amount: {loan_amount}, Loan Term: {loan_term}, "
          f"CIBIL Score: {cibil_score}, Assets: {assets}")

    # Encode categorical variables
    education_val = 1 if education == 'Graduate' else 0
    self_employed_val = 1 if self_employed.strip() == 'Yes' else 0

    # Prepare input features - reshape to 2D array
    features = np.array([[no_of_dependents, education_val, self_employed_val, income_annum, loan_amount,
                          loan_term, cibil_score, assets]])

    # Make prediction
    prediction = model.predict(features)[0]
    result = "Approved" if prediction == 1 else "Rejected"

    # Log prediction result for debugging
    print(f"Prediction: {result}")

    # Return result to HTML template
    return render_template('index.html', prediction_text=f'Loan Status: {result}')

if __name__ == '__main__':
    app.run(debug=True)
