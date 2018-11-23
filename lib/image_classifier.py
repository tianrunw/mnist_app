import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt



class _image_classifier (object):
    """
    Image classification model for keras fashion and number MNIST
    """
    def __init__ (self, dataset):
        self.dataset = dataset
        self.model = None
        self.loaded = False


    def load_data (self):
        if not self.loaded:
            (train_images, train_labels), (test_images, test_labels) = \
                self.dataset.load_data()
            self.train_images = train_images/255.0
            self.train_labels = train_labels
            self.test_images = test_images/255.0
            self.test_labels = test_labels
            self.loaded = True


    def _compile (self):
        assert(self.model is not None)
        self.model.compile(optimizer=tf.train.AdamOptimizer(), 
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])


    def train (self, epochs=5):
        self.load_data()

        # set up layers
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

        # compile the model
        self._compile()

        # train the model
        self.model.fit(self.train_images, self.train_labels, epochs=epochs)


    def evaluate (self):
        assert(self.model is not None)
        self.load_data()
        loss, acc = self.model.evaluate(self.test_images, self.test_labels)
        print("Test loss:", round(loss, 4))
        print("Test accuracy:", round(acc, 4))


    # single image only, image has dimension 2
    def predict_index (self, image):
        assert(self.model is not None)
        pred = self.model.predict(np.array([image]))
        return np.argmax(pred)


    # single image only, image has dimension 2
    def predict_label (self, image):
        assert(hasattr(self, 'labels'))
        return self.labels[self.predict_index(image)]


    def save_model (self):
        assert(self.model is not None)
        assert(hasattr(self, 'model_path'))
        self.model.save(self.model_path)


    def load_model (self):
        assert(hasattr(self, 'model_path'))
        assert(os.path.exists(self.model_path))
        self.model = keras.models.load_model(self.model_path)
        self._compile()


    def plot (self, image):
        plt.figure()
        plt.imshow(image)
        plt.colorbar()
        plt.grid(False)
        plt.show()



class fashion_mnist (_image_classifier):

    def __init__ (self):
        super().__init__(keras.datasets.fashion_mnist)
        self.labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        self.model_path = 'saved_models/fashion.h5'



class number_mnist (_image_classifier):

    def __init__ (self):
        super().__init__(keras.datasets.mnist)
        self.labels = ['Zero', 'One', 'Two', 'Three', 'Four', 
                       'Five', 'Six', 'Seven', 'Eight', 'Nine']
        self.model_path = 'saved_models/number.h5'





if __name__ == '__main__':
    f = fashion_mnist()
    n = number_mnist()
