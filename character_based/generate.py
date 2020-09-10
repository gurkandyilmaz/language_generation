#%% import necessary libraries
import sys
print("Python Executable at: ", sys.executable)
print("Python version: ", sys.version)

from pathlib import Path, PurePath
import pickle
import re
import string

import tensorflow as tf
from tensorflow import keras
import numpy as np
print("TF version: ", tf.__version__)
print("Keras version:", keras.__version__)
print("CWD: ", Path().absolute())

from prepare import PrepareModel, GenerateText


#%%
char2idx = PrepareModel().load_pickle("char2idx.pkl")
idx2char = PrepareModel().load_pickle("idx2char.pkl")
vocab_size = len(char2idx)
# print(idx2char[char2idx.get('ö')])

embedding_dim = 256
rnn_units = 1024
batch_size = 1

model = PrepareModel().build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
print(GenerateText().generate_text(model, start_string=u"akşam olunca "))
