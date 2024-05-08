from flask import Flask, render_template, request
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input data from the form
        air_temperature = request.form['Air_temperature_K']
        process_temperature = request.form['Process_temperature_K']
        rotational_speed = request.form['Rotational_speed_rpm']
        torque = request.form['Torque_Nm']
        tool_wear = request.form['Tool_wear_min']
        
        # Create a DataFrame from the input data
        input_features = pd.DataFrame({
            'Air temperature [K]': [air_temperature],
            'Process temperature [K]': [process_temperature],
            'Rotational speed [rpm]': [rotational_speed],
            'Torque [Nm]': [torque],
            'Tool wear [min]': [tool_wear]
        })

        # Load the saved model
        model = joblib.load('predictive_maintenance_model.pkl')
         
        # Predict the failure type
        prediction = model.predict(input_features)
        # Pass prediction value back to index.html
        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
