import tensorflow as tf
import numpy as np
import pandas as pd
import csv
import os
import datetime
import time
import matplotlib.pyplot as plt

print(tf.version)
print(tf.__version__)

tf.keras.backend.clear_session() 
tf.config.list_physical_devices('GPU')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    print(e)

# REMEMBER IMPORT THE DATASET FROM LOCAL FILESYSTEM!!
# __________________________________________-
%cd version1/
!head -3 news-headlines.tsv

# REMEMBER IMPORT THE DATASET FROM LOCAL FILESYSTEM!!
# __________________________________________-

vocab = "abcdefghijklmnopqrstuvwxyz0123456789 -,;.!?:’’’/\|_@#$%ˆ&*˜‘+-=()[]{}' ABCDEFGHIJKLMNOPQRSTUVWXYZ"
chars = set(vocab)
chars = sorted(chars)

EOS = '<EOS>'
UNK = "<UNK>"
PAD = "<PAD>" 

chars.append(EOS)
chars.append(UNK)
chars.insert(0, PAD)

char2idx = {c:i for i,c in enumerate(chars)}
print(char2idx[UNK])

def char_idx(c):
  if c in chars:
    return char2idx[c]
  return char2idx[UNK]

data = []
MAX_LEN = 100
with open('news-headlines.tsv', 'r') as file:
  lines = csv.reader(file, delimiter='\t')
  for line in lines:
    headline = line[0]
    tokenized =  [char_idx(c) for c in headline]
    if len(tokenized) >= MAX_LEN:
      tokenized = tokenized[:MAX_LEN-1]
      tokenized.append(char_idx(EOS))
    else:
      tokenized.append(char_idx(EOS))
      remain = MAX_LEN - len(tokenized)  
      if remain>0 :
        for i in range(remain):
          tokenized.append(char_idx(PAD))  
    
    data.append(tokenized)

data = np.asarray(data)
data_in = data[:, :-1]
data_out = data[:, 1:]
print(data_in[0])
print(data_out[0])
print(len(data_in[0]) , '  ', len(data_out[0]))

x = tf.data.Dataset.from_tensor_slices((data_in, data_out))

print(data_in.shape)
print(data_in[0, :])


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([#tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True, batch_input_shape=[batch_size, None]),
                              #tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True, input_length=len(data_in), batch_size= batch_size),
                              tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True, input_length=data_in.shape[1], batch_size= batch_size),
                              tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
                              tf.keras.layers.Dense(vocab_size)
                              ])
  return model


dropout = 0.2
def build_model2(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([#Embedding(vocab_size, embedding_dim, mask_zero=True, batch_input_shape=[BATCH_SIZE, None] ),
                               tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True, input_length=data_in.shape[1] ),
                               tf.keras.layers.LSTM(units=rnn_units, return_sequences=True, dropout=dropout, kernel_initializer=tf.keras.initializers.he_normal() ),
                               tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(rnn_units, activation= 'relu')),
                               tf.keras.layers.Dense(vocab_size)
                               ])
  return model

def build_model3(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([Embedding(vocab_size, embedding_dim, mask_zero=True, batch_input_shape=[batch_size, None] ),
                              tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
                              tf.keras.layers.dropout(0.1),
                              tf.keras.layers.Dense(vocab_size)
                               ])
  return model


vocab_size = len(vocab)
embedding_dim = 256  
rnn_units = 1024
batch_size = 128

model = build_model3(vocab_size, embedding_dim, rnn_units, batch_size)

print(model.summary())
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer = 'adam', loss = loss)


dt = datetime.datetime.today().strftime("%Y-%b-%d-%H-%M-%S")
checkpoint_dir = './training_checkpoints/'+ dt

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS=25
x_train = x.shuffle(100000, reshuffle_each_iteration=True
                   ).batch(batch_size, drop_remainder=True)
#lr_decay = LearningRateScheduler(0.001, 4., EPOCHS, 10)
start = time.time()
#history = model.fit(x_train, epochs=EPOCHS, 
#                    callbacks=[checkpoint_callback, lr_decay])
history = model.fit(x_train, epochs=EPOCHS, 
                    callbacks=[checkpoint_callback])
print("Training time: ", time.time()- start)

# Plot accuracies
lossplot = "loss-" + dt + ".png"
plt.plot(history.history['loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig(lossplot)

print("Saved loss to: ", lossplot)

