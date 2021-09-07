import tensorflow as tf
import numpy as np

print(tf.__version__)

tf.keras.backend.clear_session() #- for easy reset of notebook state

# chck if GPU can be seen by TF
tf.config.list_physical_devices('GPU')
#tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)


!pip install transformers
from transformers import TFOpenAIGPTLMHeadModel, OpenAIGPTTokenizer

gpttokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
gpt = TFOpenAIGPTLMHeadModel.from_pretrained('openai-gpt')

# ___________________________________________________________________________

input_ids = gpttokenizer.encode('Robotics is the ', return_tensors='tf')
print(input_ids)
greedy_output = gpt.generate(input_ids, max_length=100)

print("Output:\n" + 100 * '-')
print(gpttokenizer.decode(greedy_output[0], skip_special_tokens=True))

# ___________________________________________________________________________
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

gpt2tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

gpt2 = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=gpt2tokenizer.eos_token_id)

input_ids = gpt2tokenizer.encode('Robotics is the ', return_tensors='tf')
greedy_output = gpt2.generate(input_ids, max_length=50)

print("Output:\n" + 50 * '-')
print(gpt2tokenizer.decode(greedy_output[0], skip_special_tokens=True))


# ___________________________________________________________________________
tf.random.set_seed(42)  
# BEAM SEARCH

beam_output = gpt2.generate(
    input_ids, 
    max_length=51, 
    num_beams=20, 
    early_stopping=True
)

print("Output:\n" + 50 * '-')
print(gpt2tokenizer.decode(beam_output[0], skip_special_tokens=True))

# ___________________________________________________________________________
beam_output = gpt2.generate(
    input_ids, 
    max_length=50, 
    num_beams=5, 
    no_repeat_ngram_size=3, 
    early_stopping=True
)

print("Output:\n" + 50 * '-')
print(gpt2tokenizer.decode(beam_output[0], skip_special_tokens=True))

# ___________________________________________________________________________
tf.random.set_seed(42)  # for reproducible results
beam_outputs = gpt2.generate(
    input_ids, 
    max_length=50, 
    num_beams=7, 
    no_repeat_ngram_size=3, 
    num_return_sequences=3,  
    early_stopping=True,
    temperature=0.7
)

print("Output:\n" + 50 * '-')
for i, beam_output in enumerate(beam_outputs):
  print("\n{}: {}".format(i, 
                        gpt2tokenizer.decode(beam_output, 
                                             skip_special_tokens=True)))

# ___________________________________________________________________________
tf.random.set_seed(42)  # for reproducible results
beam_output = gpt2.generate(
    input_ids, 
    max_length=50, 
    do_sample=True, 
    top_k=25,
    temperature=2
)

print("Output:\n" + 50 * '-')
print(gpt2tokenizer.decode(beam_output[0], skip_special_tokens=True))

# ___________________________________________________________________________
input_ids = gpt2tokenizer.encode('In the dark of the night, there was a ', return_tensors='tf')
# Top-K sampling
tf.random.set_seed(42)  # for reproducible results
beam_output = gpt2.generate(
    input_ids, 
    max_length=200, 
    do_sample=True, 
    top_k=50
)

print("Output:\n" + 50 * '-')
print(gpt2tokenizer.decode(beam_output[0], skip_special_tokens=True))

# ___________________________________________________________________________
gpt2tok_l = GPT2Tokenizer.from_pretrained("gpt2-large")

# add the EOS token as PAD token to avoid warnings
gpt2_l = TFGPT2LMHeadModel.from_pretrained("gpt2-large", pad_token_id=gpt2tokenizer.eos_token_id)


# ___________________________________________________________________________

#input_ids = gpt2tok_l.encode('In the dark of the night, there was a ', return_tensors='tf')
input_ids = gpt2tok_l.encode('London is the capital of ', return_tensors='tf')

# Top-K sampling
tf.random.set_seed(42)  # for reproducible results
beam_output = gpt2_l.generate(
    input_ids, 
    max_length=200, 
    do_sample=True, 
    top_k=25
)

print("Output:\n" + 50 * '-')
print(gpt2tok_l.decode(beam_output[0], skip_special_tokens=True))


