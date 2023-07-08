# Keras
from __future__ import division, print_function

import os

import numpy as np
# Flask utils
from flask import Flask, request, render_template
from keras.models import load_model
from keras.preprocessing import image
from werkzeug.utils import secure_filename

# from gevent.pywsgi import WSGIServer

app = Flask(__name__)
# Model saved with keras model.save()
model_path = 'Image_Classification.h5'

# Loading model
model = load_model(model_path)


def model_predict(img_path, model1):
    img = image.load_img(img_path, target_size=(32, 32, 3))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    # Scaling
    x = x / 255
    x = np.expand_dims(x, axis=0)

    preds = model1.predict(x)
    class_names = ['Airplane', 'Automobile', 'Birds', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    preds = class_names[np.argmax(preds)]

    return preds


@app.route('/')
def home():
    return render_template('index.html')


# get user input, predict output and then return to user
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./pics
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'pics', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result = preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
