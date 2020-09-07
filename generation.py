import sys
print("Python Executable at: ", sys.executable)
print("Python version: ", sys.version)

from pathlib import Path
import pickle
import re
import string

import tensorflow as tf
from tensorflow import keras
import numpy as np
print("TF version: ", tf.__version__)
print("Keras version:", keras.__version__)
print("CWD: ", Path().absolute())


#%% Load File
def load_file(file_path):
    # print("file_path: ", Path().absolute(file_path))
    with open(file_path, 'r') as file:
        text = file.read()

    return text

#%% Data Preparation
def create_sequence(raw_text, sequence_length):
    sequences = list()
    for idx in range(0, len(raw_text)+1):
        if idx == len(raw_text)+1-sequence_length:
            break
        sequences.append(raw_text[idx:idx+sequence_length])
    return sequences

def save_sequence(sequences, filename):
    with open(filename, 'w') as file:
        data = "\n".join(sequences)
        file.write(data)
    return Path(filename).is_file()

#%% Create the model and compile
def define_model(X):
    model = keras.models.Sequential([
        keras.layers.LSTM(128, input_shape=(X.shape[1], X.shape[2])),
        keras.layers.Dense(64),
        keras.layers.Dense(vocab_size, activation='softmax')
    ])
    optimizer = keras.optimizers.Adam()
    loss = keras.losses.CategoricalCrossentropy()
    model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=["accuracy"]
    )
    model.summary()
    return model

#%% Predict character
# TODO


#%%
if __name__ == "__main__":
    # raw_text_unformatted = load_file("language_generation/yapay_sinir.txt")
    # tokens = raw_text_unformatted.split()
    # raw_text_in_a_string = " ".join(tokens)
    # sequences = create_sequence(raw_text_in_a_string, 20)
    # print(len(sequences))
    # print(sequences[:10])
    # status = save_sequence(sequences, "language_generation/yapay_sinir_sequence.txt")
    # print(status)

    raw_text_sequence = load_file("language_generation/yapay_sinir_sequence.txt")
    lines = raw_text_sequence.split("\n")
    # print(lines[:10])
    characters = sorted(list(set(raw_text_sequence)))
    # print(characters)
    mapping = dict((v,k) for k,v in enumerate(characters))
    vocab_size = len(mapping)
    # print(mapping)
    encoded_lines = []
    for line in lines:
        encoded = [mapping.get(character) for character in line]
        encoded_lines.append(encoded)
    # print(encoded_lines)
    encoded_lines = np.array(encoded_lines)
    # print(encoded_lines.shape)
    X, y = encoded_lines[:,:-1], encoded_lines[:,-1]
    # print(X.shape, y.shape)
    X = np.array([keras.utils.to_categorical(x, num_classes=vocab_size) for x in X])
    y = keras.utils.to_categorical(y, num_classes=vocab_size)
    # print(X.shape, y.shape)
    model = define_model(X)
    # model.fit(X,y, epochs=100, validation_split=0.2)
    # model.save("language_generation/char_based.h5")
    # with open("language_generation/char_based_mapping.pkl", "wb") as file:
        # pickle.dump(mapping, file, protocol=pickle.HIGHEST_PROTOCOL)
