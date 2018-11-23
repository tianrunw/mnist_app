import string
import numpy as np
from tensorflow import keras


# dictionary mapping words to an integer index
word_index = keras.datasets.imdb.get_word_index()

# the first indices are reserved
word_index = {k:(v + 3) for (k, v) in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

# dictionary mapping integer index to words
reverse_word_index = {value:key for (key, value) in word_index.items()}



def decode_review (text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


def encode_review (text):
    table = str.maketrans({key: None for key in string.punctuation})
    text = text.lower().translate(table)
    words = text.split(' ')

    def translate (word):
        if len(word) == 0:
            return None
        elif word in word_index:
            return word_index[word]
        else:
            return 2

    words = map(translate, words)
    words = filter(lambda x: x is not None, words)
    return list(words)




if __name__ == '__main__':
    text = "This movie sucks! What a waste of time. iPhone is cool."
    codes = encode_review(text)
    words = decode_review(codes)
