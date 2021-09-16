# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import numpy as np
from flask import Flask, request, render_template
import pickle

#create flask app
app = Flask(__name__)

#load pickle model
model = pickle.load(open("model.pkl", "rb"))

#home route
@app.route('/')
def Home():
   return render_template("index.html")

#predict post route
@app.route('/predict', methods = ['POST'])
def predict():
    float_features = [x for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    text = prediction[0]
    string = "Benign" if (text == 2) else "Malignant"    
    
    
    return render_template("index.html", prediction_text = "The Class is : {}".format(string))

if __name__ == '__main__':
   app.run(debug=True)

