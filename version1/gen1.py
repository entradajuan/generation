import tensorflow as tf
import numpy as np
import pandas as pd
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

import csv

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

