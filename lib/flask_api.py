import datetime as dt
from flask import Flask, jsonify, abort, make_response, request
from text_classifier import imdb_sentiment

app = Flask(__name__)



@app.errorhandler(400)
def wrong_file_type (error):
    return make_response(jsonify({'error': 'Wrong file type'}), 400)


@app.route('/services', methods=['GET'])
def get_services ():
    services = ['mnist', 'fashion', 'imdb']
    return jsonify({'services': services})


@app.route('/imdb', methods=['POST'])
def process_imdb ():
    f = request.files['file']
    if not f.filename.endswith('.txt'):
        abort(400)

    text = f.read().decode()
    imdb = imdb_sentiment()
    imdb.load_model()
    result = imdb.predict(text)

    if round(result) == 1:
        sentiment = 'positive'
    elif round(result) == 0:
        sentiment = 'negative'
    else:
        raise ValueError

    return jsonify({'numeric':str(result), 'sentiment':sentiment})






if __name__ == '__main__':
    app.run(debug=True)
