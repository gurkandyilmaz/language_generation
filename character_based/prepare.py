#%%
import pickle
from pathlib import Path, PurePath

import tensorflow as tf

#%%
class PrepareModel():

    def __init__(self):
        pass

    def load_pickle(self, pickle_file):
        # print(Path(pickle_file).resolve())
        cwd = Path.cwd()
        if cwd.parts[-1] == 'language_model':
            pkl_path = PurePath(cwd).joinpath('language_generation/character_based')
            pickle_file = PurePath(pkl_path).joinpath(pickle_file)
            with open(pickle_file, 'rb') as filename:
                enc_dec = pickle.load(filename)
        elif cwd.parts[-1] == 'character_based':
            with open(pickle_file, 'rb') as filename:
                enc_dec = pickle.load(filename)
        return enc_dec

    def build_model(self, vocab_size, embedding_dim, rnn_units, batch_size):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
            tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(vocab_size)
        ])
        ckpt_best_path = Path.cwd().resolve()
        # print(ckpt_best_path)
        model.load_weights(ckpt_best_path.joinpath('ckpt_best'))
        model.build(tf.TensorShape([1,None]))

        return model

class GenerateText():
    def __init__(self):
        pass

    def generate_text(self, model, start_string, num_generate=500, temperature= 1.0):
      # num_generate = 500
      input_eval = [char2idx[s] for s in start_string]
      # print(len(input_eval))
      input_eval = tf.expand_dims(input_eval, 0)
      # print(input_eval.shape)
      text_generated = []
      # Low temperatures results in more predictable text.
      # Higher temperatures results in more surprising text.
      # temperature = 0.10

      model.reset_states()
      for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

      return (start_string + ''.join(text_generated))


#%%
char2idx = PrepareModel().load_pickle("char2idx.pkl")
idx2char = PrepareModel().load_pickle("idx2char.pkl")
vocab_size = len(char2idx)
# print(idx2char[char2idx.get('ö')])
embedding_dim = 256
rnn_units = 1024
batch_size = 1

#%%
if __name__ == '__main__':
    # print(vocab_size, embedding_dim, rnn_units, batch_size)
    model = PrepareModel().build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
    # model.summary()
    print(GenerateText().generate_text(model, start_string=u"abimi görünce "))
