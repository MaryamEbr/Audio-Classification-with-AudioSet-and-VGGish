import torch
import urllib
from scipy.io import wavfile
import soundfile as sf
import os
import numpy as np
from torchsummary import summary
from torch import nn, Tensor
import torch.optim as optim
from torch import hub
import csv
import sys
import vggish_input, vggish_params

if torch.cuda.is_available():
    device = torch.device("cuda") 
    print("Using GPUs ", device)
else:
    device = torch.device("cpu")


### loading dataset and labels
valid_folder = 'audioset_datasest/validset/valid_wav/'

label_displayname_dict = {}
for i, key in enumerate(csv.DictReader(open(os.path.join('audioset_datasest/trainset/class_labels_indices.csv')))):
    label_displayname_dict[key['mid']] = key['display_name']

sample_label_dict_valid = {}
for key in csv.DictReader(open(os.path.join('audioset_datasest/validset/valid.csv'))):
    sample_label_dict_valid[key['YTID']] = key['positive_labels']

class_num = len(list(label_displayname_dict.keys()))


for ind, file in enumerate(os.listdir(valid_folder)):
    
    # getting wav files
    wav_data, sr = sf.read(os.path.join(valid_folder, file), dtype='int16')
    wav_data = wav_data / 32768.0  # Convert to [-1.0, +1.0]

    # preprocessing wav to mel spectogram 
    # (shape: 10, 1, 96, 64 (10 may be different, depending on audio length, usually it's 10s so: 10))
    if(len(wav_data))<100000:
        print("XXXXX skipped, len: ", len(wav_data))
        continue
    mel_data = vggish_input.waveform_to_examples(wav_data, sr)
    
    # preprocessing labels to onehot version
    wav_labels = sample_label_dict_valid[file.split('.wav')[0]].split(',')
    wav_labels_indexes = [list(label_displayname_dict.keys()).index(a) for a in  wav_labels]
    wav_labels_onehot = [1 if a in wav_labels_indexes  else 0 for a in range(class_num)]
    label_data = torch.Tensor([wav_labels_onehot for a in range(mel_data.shape[0])])


    if ind == 0 or (ind-1)%batch_size == 0:
        mel_data_batch = mel_data
        label_data_batch = label_data
        

    elif ind%batch_size==0 and ind!=0:
        print("........, batch", ind/batch_size,  "wav_len", len(wav_data), "sr", sr, "mel_data ", mel_data_batch.shape, " label ", label_data_batch.shape, flush=True )
        torch.save(mel_data_batch, f"{file_name}_mel_{int(ind/batch_size)}.pt")
        torch.save(label_data_batch, f"{file_name}_label_{int(ind/batch_size)}.pt")
        # mel_data_batch_total.append(mel_data_batch)
        # label_data_batch_total.append(label_data_batch)

    else:
        ### making batch
        mel_data_batch = torch.cat((mel_data_batch, mel_data), dim=0)
        label_data_batch = torch.cat((label_data_batch, label_data), dim=0)
