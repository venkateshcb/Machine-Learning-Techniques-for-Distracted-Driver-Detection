from flask import Flask, render_template, request
import requests
from keras.models import load_model
import tensorflow as tf
import numpy as np

app = Flask(__name__)

dic = {
    0: "safe driving",
    1: "texting - right",
    2: "talking on the phone - right",
    3: "texting - left",
    4: "talking on the phone - left",
    5: "operating the radio",
    6: "drinking",
    7: "reaching behind",
    8: "hair and makeup",
    9: "talking to passenger",
}    

model = load_model("model.h5")

def predict_class(img_path):
    data = []
    img = tf.keras.preprocessing.image.load_img(img_path,color_mode="grayscale",target_size=(64,64))
    img = np.array(img)
    data.append(img)
    data = np.array(data)
    p = model.predict(data)
    p = [np.argmax(i) for i in p]
    return dic[p[0]]

@app.route('/', methods = ["GET"])
def index():
    return render_template("index.html")

@app.route('/submit', methods = ['GET', 'POST'])
def output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/"+img.filename
        img.save(img_path)

        p = predict_class(img_path)
    
    return render_template('index.html', prediction = p)

app.run(debug=True)