from flask import Flask, jsonify, request
from database import create_keyspace, query_all, insert_row

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

# services
app = Flask(__name__)
counter = 1



def register_row (model, result, label):
    global counter
    row = {'post_id':counter, 'model':model, 'result':result, 'label':label}
    insert_row(row)
    counter0 = counter
    counter += 1
    return counter0


@app.route('/', methods=['GET'])
def get_services ():
    services = ['/mnist', '/fashion', '/imdb', '/database']
    return jsonify({'services': services, 'author':'Tianrun Wang'})


@app.route('/database', methods=['GET'])
def get_database ():
    rows = query_all()
    return jsonify(rows)


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

    pid = register_row('imdb', result, sentiment)
    return jsonify({'numeric':str(result), 'sentiment':sentiment, 
                    'id':str(pid)})


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
            pid = register_row('mnist', index, label)
        except:
            index = 'NA'
            label = 'NA'
            pid = 'NA'

    return jsonify({'index':str(index), 'label':label, 'id':str(pid)})


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
            pid = register_row('fashion', index, label)
        except:
            index = 'NA'
            label = 'NA'
            pid = 'NA'

    return jsonify({'index':str(index), 'label':label, 'id':str(pid)})





if __name__ == '__main__':
    # docker run -p 9042:9042 cassandra
    create_keyspace()
    app.run(host='0.0.0.0', port=80, debug=True)
