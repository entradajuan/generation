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
  model = tf.keras.Sequential([tf.keras.layers.Embedding(vocab_size, embedding_dim,batch_input_shape=[batch_size, None]),
                              tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
                              tf.keras.layers.Dense(vocab_size)
                              ])
  return model


vocab_size = len(vocab)
embedding_dim = 256  
rnn_units = 1024
batch_size = 1 #128

gen_model = build_gen_model2(vocab_size, embedding_dim, rnn_units, batch_size)

checkpoint_dir = './training_checkpoints/2021-Sep-07-17-19-19' 

gen_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

gen_model.build(tf.TensorShape([1, None]))


def generate_text(model, start_string, temperature=0.05, num_generate=50):
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  text_generated = []

  # Here batch size == 1
  for i in range(num_generate):
      predictions = model(input_eval)
      print(type(predictions))
#      print(predictions)
      predictions = tf.squeeze(predictions, 0)
#      print(predictions)
      predictions = predictions / temperature
#      print(predictions)
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
      print(predicted_id)
      input_eval = tf.expand_dims([predicted_id], 0)        
      text_generated.append(idx2char[predicted_id])


  return (start_string + ''.join(text_generated))

print(generate_text(gen_model, start_string=u"Appl"))
  