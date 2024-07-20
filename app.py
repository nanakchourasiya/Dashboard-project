

from flask import Flask, render_template, request
import numpy as np
import pickle
import wine
import pandas as pd
import air
import water


app = Flask(__name__)
app.config['DEBUG'] = True 

# with open('model.pkl', 'rb') as f:
#     model_data = pickle.load(f)

with open('air_model.pkl', 'rb') as f:
    air_model = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/wine', methods=['POST','GET'])
def wine_page():
    return render_template('wine.html')

@app.route('/water', methods=['POST','GET'])
def water_page():
    return render_template('water.html')

@app.route('/air', methods=['POST','GET'])
def air_page():
    return render_template('air.html')


@app.route('/wine_predict', methods=['POST','GET'])
def wine_predict():
    if request.method == 'POST':
    # Extract input data from the form
         input_data = [float(x) for x in request.form.values()]

    # Make prediction using the function from wine.py
         prediction = wine.predict_wine_quality(input_data)

    # Return prediction result to the user
         return render_template('wine_result.html', prediction=prediction)
    return render_template('wine_result.html', prediction=None)

@app.route('/water_predict', methods=['POST', 'GET'])
def water_predict():
    if request.method == 'POST':
        # Extract input data from the form
        input_data = [float(x) for x in request.form.values()]

        # Make prediction using the pre-trained model
        result = water.water_quality_prediction(input_data)

        # Return prediction result to the user
        return render_template('water_result.html', result=result)
    return render_template('water_result.html', result=None)

@app.route('/air_predict', methods=['POST', 'GET'])
def air_predict():
    if request.method == 'POST':
        # Extract input data from the form
        input_data = [float(x) for x in request.form.values()]
        input_data_reshape_air = np.array(input_data).reshape(1, -1)

        # Make prediction using the air.py module
        air_prediction = air.model_lr.predict(input_data_reshape_air)

        if air_prediction[0] == 'Good':
            predict = "Quality of air is good"
        else:
            predict = "Quality of air is polluted"

        # Return prediction result to the user
        return render_template('air_result.html', predict = predict)
    return render_template('air_result.html', predict=None)


if __name__ == '__main__':
    app.run()
