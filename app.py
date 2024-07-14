import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import imutils
from imutils.contours import sort_contours
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
model = tf.keras.models.load_model('maths_symbol_and_digit_recognition.h5')

def process_image(image_path):
    chars = []
    img = cv2.imread(image_path)
    img = cv2.resize(img, (800, 800))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(img_gray, 30, 150)
    contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    
    if len(contours) == 0:
        return "No contours found", None
    
    contours = sort_contours(contours, method="left-to-right")[0]
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'add', 'div', 'mul', 'sub']

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if 20 <= w and 30 <= h:
            roi = img_gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            (th, tw) = thresh.shape
            if tw > th:
                thresh = imutils.resize(thresh, width=32)
            if th > tw:
                thresh = imutils.resize(thresh, height=32)
            (th, tw) = thresh.shape
            dx = int(max(0, 32 - tw) / 2.0)
            dy = int(max(0, 32 - th) / 2.0)
            padded = cv2.copyMakeBorder(thresh, top=dy, bottom=dy, left=dx, right=dx, borderType=cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0))
            padded = cv2.resize(padded, (32, 32))
            padded = np.array(padded)
            padded = padded / 255.
            padded = np.expand_dims(padded, axis=0)
            padded = np.expand_dims(padded, axis=-1)
            pred = model.predict(padded)
            pred = np.argmax(pred, axis=1)
            label = labels[pred[0]]
            chars.append(label)

    expression = ''
    for char in chars:
        if char == 'add':
            expression += '+'
        elif char == 'sub':
            expression += '-'
        elif char == 'mul':
            expression += '*'
        elif char == 'div':
            expression += '/'
        else:
            expression += char

    try:
        result = eval(expression)
    except:
        result = "Error in evaluating expression"
    return expression, result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = data['image']
    image_data = image_data.split(",")[1]
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image_path = 'temp.png'
    image.save(image_path)

    expression, result = process_image(image_path)
    if result is None:
        return jsonify({'error': expression})
    return jsonify({'expression': expression, 'result': result})

if __name__ == '__main__':
    app.run(debug=True)


