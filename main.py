from flask import Flask, jsonify, abort, make_response, request

import numpy as np
import tensorflow as tf
from PIL import Image

from text_classifier import imdb_sentiment
from image_classifier import number_mnist, fashion_mnist

# load models
IMDB = imdb_sentiment()
MNIST = number_mnist()
FASHION = fashion_mnist()

G = tf.Graph()
with G.as_default():
    IMDB.load_model()
    MNIST.load_model()
    FASHION.load_model()

app = Flask(__name__)



@app.route('/', methods=['GET'])
def get_services ():
    services = ['mnist', 'fashion', 'imdb']
    return jsonify({'services': services, 'author':'Tianrun Wang'})


@app.route('/imdb', methods=['POST'])
def process_imdb ():
    if 'file' not in request.files:
        return jsonify({'error':'use curl -F file=@<file_path> <url>'})

    f = request.files['file']
    if not f.filename.endswith('.txt'):
        return jsonify({'error':'wrong file type, upload only .txt files'})
    
    text = f.read().decode()
    with G.as_default():
        try:
            result = IMDB.predict(text)
        except:
            result = -1

    if round(result) == 1:
        sentiment = 'positive'
    elif round(result) == 0:
        sentiment = 'negative'
    else:
        sentiment = 'NA'

    return jsonify({'numeric':str(result), 'sentiment':sentiment})


@app.route('/mnist', methods=['POST'])
def process_mnist ():
    if 'file' not in request.files:
        return jsonify({'error':'use curl -F file=@<file_path> <url>'})

    f = request.files['file']
    image = Image.open(f)
    image = np.array(image)
    if image.shape != (28, 28):
        return jsonify({'error':'input shape must be (28, 28)'})

    with G.as_default():
        try:
            index = MNIST.predict_index(image)
            label = MNIST.labels[index]
        except:
            index = 'NA'
            label = 'NA'

    return jsonify({'index':str(index), 'label':label})


@app.route('/fashion', methods=['POST'])
def process_fashion ():
    if 'file' not in request.files:
        return jsonify({'error':'use curl -F file=@<file_path> <url>'})

    f = request.files['file']
    image = Image.open(f)
    image = np.array(image)
    if image.shape != (28, 28):
        return jsonify({'error':'input shape must be (28, 28)'})

    with G.as_default():
        try:
            index = FASHION.predict_index(image)
            label = FASHION.labels[index]
        except:
            index = 'NA'
            label = 'NA'

    return jsonify({'index':str(index), 'label':label})





if __name__ == '__main__':
    app.run(debug=True)
