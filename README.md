# training-VGG-model-and-AudioSet-dataset
This repository contains scripts for training VGG model on Google's AudioSet dataset (https://research.google.com/audioset/)  
using help repository:
https://github.com/tensorflow/models/tree/master/research/audioset/vggish
and the VGGish paper:
https://research.google/pubs/pub45611/

This is done as a part of a research project about early exiting and partitioning. So it also contains a multi-exit version of the VGGish model. 
In the multi-exit case, easier samples that do not need the entire model for classification, can take the earlier exits and save time in the inference process.
This way we accelerate the inference process without losing much accuracy.
The codes are in PyTorch.
