#%% import necessary libraries
import sys
print("Python Executable at: ", sys.executable)
print("Python version: ", sys.version)
from pathlib import Path
from prepare import PrepareModel, GenerateText

print("CWD: ", Path().absolute())

#%%
def main(text, num_generate, temperature):
    char2idx = PrepareModel().load_pickle("char2idx.pkl")
    idx2char = PrepareModel().load_pickle("idx2char.pkl")
    vocab_size = len(char2idx)
    # print(idx2char[char2idx.get('ö')])
    # Model building parameters
    embedding_dim = 256
    rnn_units = 1024
    batch_size = 1
    # Model predict parameters
    model = PrepareModel().build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
    generated_text = GenerateText().generate_text(model, start_string=u"{0}".format(text), num_generate=num_generate, temperature=temperature)

    return generated_text

#%%
num_generate = 100
temperature = 0.10
text = "her zaman oduğu gibi "
generated_text = main(text, num_generate, temperature)
print(generated_text)
