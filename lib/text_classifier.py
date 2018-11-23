import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

from imdb_dict import decode_review, encode_review, word_index



def pad_sequence (seq, maxlen):
    padded = keras.preprocessing.sequence.\
                pad_sequences(seq,
                              value=word_index["<PAD>"],
                              padding='post',
                              maxlen=maxlen)
    return padded



class imdb_sentiment (object):

    def __init__ (self):
        self.maxlen = 256
        self.vocab_size = 10000
        self.model = None
        self.model_path = 'saved_models/imdb.h5'
        self.loaded = False


    def load_data (self):
        if not self.loaded:
            imdb = keras.datasets.imdb
            (train_data, train_labels), (test_data, test_labels) = \
                imdb.load_data(num_words=self.vocab_size)

            self.train_data = pad_sequence(train_data, self.maxlen)
            self.test_data = pad_sequence(test_data, self.maxlen)
            self.train_labels = train_labels
            self.test_labels = test_labels
            self.loaded = True


    def _compile (self):
        assert(self.model is not None)
        self.model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])


    def train (self, epochs=40):
        self.load_data()
        model = keras.Sequential()
        model.add(keras.layers.Embedding(self.vocab_size, 16))
        model.add(keras.layers.GlobalAveragePooling1D())
        model.add(keras.layers.Dense(16, activation=tf.nn.relu))
        model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
        self.model = model
        self._compile()

        x_val = self.train_data[:10000]
        partial_x_train = self.train_data[10000:]
        y_val = self.train_labels[:10000]
        partial_y_train = self.train_labels[10000:]

        history = self.model.fit(partial_x_train,
                                 partial_y_train,
                                 epochs=epochs,
                                 batch_size=512,
                                 validation_data=(x_val, y_val),
                                 verbose=1)


    def evaluate (self):
        assert(self.model is not None)
        self.load_data()
        loss, acc = self.model.evaluate(self.test_data, self.test_labels)
        print("Test loss:", round(loss, 4))
        print("Test accuracy:", round(acc, 4))


    def predict (self, text):
        assert(self.model is not None)
        assert(type(text) is str)
        codes = encode_review(text)
        pred = self.model.predict(np.array([codes])).round(4)
        return pred[0][0]


    def save_model (self):
        assert(self.model is not None)
        self.model.save(self.model_path)


    def load_model (self):
        assert(os.path.exists(self.model_path))
        self.model = keras.models.load_model(self.model_path)
        self._compile()





if __name__ == '__main__':
    text1 = "This movie sucks! What a waste of time."
    text2 = "This is the greatest film I have ever seen."
    text3 = "Although not for me, I can see why people love this flick."
    m = imdb_sentiment()
