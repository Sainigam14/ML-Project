from flask import Flask, render_template, request
import numpy as np
import joblib


app = Flask(__name__)

model = joblib.load('bank_campaign_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    age = int(request.form['age'])
    job = int(request.form['job'])
    marital = int(request.form['marital'])
    education = int(request.form['education'])
    default = int(request.form['default'])
    balance = int(request.form['balance'])
    house = int(request.form['house'])
    loan = int(request.form['loan'])
    contact  = int(request.form['contact'])
    day = int(request.form['day'])
    month = int(request.form['month'])
    duration = int(request.form['duration'])
    campaign = int(request.form['campaign'])
    pdays = int(request.form['pdays'])
    previous = int(request.form['previous'])
    poutcome = int(request.form['poutcome'])

    new_arr = np.array([[age,job,marital,education,default,balance,house,loan,contact,day,month,duration,campaign,pdays,previous,poutcome]])

    new_output = model.predict(new_arr)

    if(new_output[0]==1):
        predicted_result = "Customer is likely to subscribe!"
    else:
        predicted_result = "Customer is unlikely to subscribe"
    
    return render_template('prediction.html',predicted_result = predicted_result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)