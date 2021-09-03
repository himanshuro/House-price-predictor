from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv("models/Cleaned_Bengaluru_House_Data.csv")
pipe = pickle.load(open("models/Ridge.pkl", "rb"))

@app.route("/")
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations = locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('locality')
    bhk = float(request.form.get('bhk'))
    bath = float(request.form.get('bathroom'))
    area = request.form.get('sqft')
    input = pd.DataFrame([[location, area, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = pipe.predict(input)[0] 
    
    if(prediction < 100):
        return "---> Rs." + str(round(prediction,4)) + " lakhs <---"
    return "---> Rs. " + str(round((round(prediction,2)/100),4)) + " Cr <---" 

if __name__ == "__main__":
    app.run(debug = True, host='0.0.0.0', port = 3030)