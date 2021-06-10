!git clone https://github.com/openai/gpt-2.git

import os
os.chdir("/content/generation/gpt-2/")
!pip3 install -r requirements.txt
!pip install toposort
%tensorflow_version 1.x
import tensorflow as tf
print(tf.__version__)

!python3 download_model.py '117M'

!python3 download_model.py '117M'

## Pay attention to upload some files from local PC in this moment

!cp /content/dset.txt /content/generation/gpt-2/src/
!cp -r /content/generation/gpt-2/models/ /content/generation/gpt-2/src/
!cp /content/train.py /content/generation/gpt-2/src/
!cp /content/load_dataset.py /content/generation/gpt-2/src/
!cp /content/encode.py /content/generation/gpt-2/src/
!cp /content/accumulate.py /content/generation/gpt-2/src/
!cp /content/memory_saving_gradients.py /content/generation/gpt-2/src/

os.chdir("/content/generation/gpt-2/src/")
model_name = "117M"

# update encode.py
!python /content/generation/gpt-2/src/encode.py dset.txt out.npz

# update train.py
!python /content/generation/gpt-2/src/train.py --dataset out.npz

run_dir = '/content/generation/gpt-2/models/tgmodel'
if not os.path.exists(run_dir):
  os.makedirs(run_dir)

!cp /content/generation/gpt-2/src/checkpoint/run1/model-1000.data-00000-of-00001 /content/generation/gpt-2/models/tgmodel
!cp /content/generation/gpt-2/src/checkpoint/run1/checkpoint /content/generation/gpt-2/models/tgmodel
!cp /content/generation/gpt-2/src/checkpoint/run1/model-1000.index /content/generation/gpt-2/models/tgmodel
!cp /content/generation/gpt-2/src/checkpoint/run1/model-1000.meta /content/generation/gpt-2/models/tgmodel
!cp /content/generation/gpt-2/models/117M/encoder.json /content/generation/gpt-2/models/tgmodel
!cp /content/generation/gpt-2/models/117M/hparams.json /content/generation/gpt-2/models/tgmodel
!cp /content/generation/gpt-2/models/117M/vocab.bpe /content/generation/gpt-2/models/tgmodel
!mv /content/generation/gpt-2/models/117M  /content/generation/gpt-2/models/117M_OpenAI
!mv /content/generation/gpt-2/models/tgmodel  /content/generation/gpt-2/models/117M






