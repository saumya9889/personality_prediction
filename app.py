# import numpy as np
# from flask import Flask, render_template, request
# from sklearn.preprocessing import StandardScaler
# import joblib
# app = Flask(__name__)
# model = joblib.load("train_model.pkl")
# scaler = StandardScaler()

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/submit',methods = ['POST', 'GET'])
# def result():
#    if request.method == 'POST':
#       gender = request.form['gender']
#       if(gender == "Female"):
#         gender_no = 1
#       else:
#         gender_no = 2
#       age = request.form['age']
#       openness = request.form['openness']
#       neuroticism = request.form['neuroticism']
#       conscientiousness = request.form['conscientiousness']
#       agreeableness = request.form['agreeableness']
#       extraversion = request.form['extraversion']
#       result = np.array([gender_no, age, openness,neuroticism, conscientiousness, agreeableness, extraversion], ndmin = 2)
#       final = scaler.fit_transform(result)
#       personality = str(model.predict(final)[0])
#       return render_template("submit.html",answer = personality)

# if __name__ == '__main__':
#     app.run()

import numpy as np
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)
model = joblib.load("train_model.pkl")
scaler = StandardScaler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        try:
            gender = request.form['gender']
            gender_no = 1 if gender.lower() == "female" else 2
            
            age = float(request.form['age'])
            openness = float(request.form['openness'])
            neuroticism = float(request.form['neuroticism'])
            conscientiousness = float(request.form['conscientiousness'])
            agreeableness = float(request.form['agreeableness'])
            extraversion = float(request.form['extraversion'])
            
            # Logging input values for debugging
            print("Input Values:")
            print(f"Gender: {gender}")
            print(f"Age: {age}")
            print(f"Openness: {openness}")
            print(f"Neuroticism: {neuroticism}")
            print(f"Conscientiousness: {conscientiousness}")
            print(f"Agreeableness: {agreeableness}")
            print(f"Extraversion: {extraversion}")
            
            # Create numpy array for prediction
            result = np.array([[gender_no, age, openness, neuroticism, conscientiousness, agreeableness, extraversion]])
            
            # Scale the input features
            final = scaler.transform(result)
            
            # Predict personality
            personality = str(model.predict(final)[0])
            
            return render_template("submit.html", answer=personality)
        except Exception as e:
            # Log any exceptions for debugging
            print(f"An error occurred: {str(e)}")
            return "An error occurred. Please try again."
    else:
        return "Invalid request method. Please use POST."

if __name__ == '__main__':
    app.run(debug=True)
