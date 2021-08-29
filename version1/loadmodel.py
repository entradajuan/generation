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


  

