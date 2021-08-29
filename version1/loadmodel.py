import tensorflow as tf
import numpy as np
import pandas as pd


vocab = "abcdefghijklmnopqrstuvwxyz0123456789 -,;.!?:’’’/\|_@#$%ˆ&*˜‘+-=()[]{}' ABCDEFGHIJKLMNOPQRSTUVWXYZ"
chars = set(vocab)
chars = sorted(chars)

EOS = '<EOS>'
UNK = "<UNK>"
PAD = "<PAD>" 

chars.append(EOS)
chars.append(UNK)
chars.insert(0, PAD)

idx2char = np.array(chars)
char2idx = {c:i for i,c in enumerate(chars)}
print(char2idx[UNK])

def char_idx(c):
  if c in chars:
    return char2idx[c]
  return char2idx[UNK]


def build_gen_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model


def build_gen_model2(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True, input_length=99, batch_size= batch_size),
                              tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
                              tf.keras.layers.Dense(vocab_size)
                              ])
  return model


vocab_size = len(vocab)
embedding_dim = 256  
rnn_units = 1024
batch_size = 1 #128

gen_model = build_gen_model2(vocab_size, embedding_dim, rnn_units, batch_size)

checkpoint_dir = './version1/training_checkpoints/2021-Aug-28-18-25-45/' 

gen_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

gen_model.build(tf.TensorShape([1, None]))


def generate_text(model, start_string, temperature=0.7, num_generate=99):
  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Here batch size == 1
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the word returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)
        
      text_generated.append(idx2char[predicted_id])
      # lets break is <EOS> token is generated
      #if idx2char[predicted_id] == EOS:
      #  break #end of a sentence reached, lets stop

  return (start_string + ''.join(text_generated))

print(generate_text(gen_model, start_string=u"Seattle"))
  

