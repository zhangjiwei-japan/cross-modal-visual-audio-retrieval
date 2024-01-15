# visual-audio cross modal retrieval task

Code for Paper: DCLMA: Deep Correlation Learning with Multi-modal Attention for Visual-Audio Retrieval.

## AVE Dataset 
AVE dataset can be downloaded from https://drive.google.com/open?id=1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK.
Audio feature and visual feature (7.7GB) are also released. 
Scripts for generating audio and visual features: https://drive.google.com/file/d/1TJL3cIpZsPHGVAdMgyr43u_vlsxcghKY/view?usp=sharing.
- Original Dataset homepage: https://sites.google.com/view/audiovisualresearch and https://github.com/YapengTian/AVE-ECCV18
## VEGAS Dataset 
The Raw dataset from: https://arxiv.org/abs/1712.01393.

## Requirements
Python-3.6, Pytorch-0.3.0, Keras, ffmpeg.

## Feature Extraction
ffmpeg:
It is a tool to edit the video or audio, more detail seen: http://ffmpeg.org/. Here, I use the tool to extract audio track from video.

## Testing:
DCLMA model in the paper: python test.py

## Training:
DCLMA model in the paper: python train.py

## Evelation: 
we use MAP as metrics to evaluate our architecture, when the system generates a ranked list in one modality for a query in another modality. Those documents in the ranked list with the same class are regarded as relevant or correct.

## Contact
If you have any questions, please email s210068@wakayama-u.ac.jp
## Other Related or Follow-up works
Zhang, Jiwei, et al. "Variational Autoencoder with CCA for Audio-Visual Cross-Modal Retrieval." ACM Transactions on Multimedia Computing, Communications and Applications (2022).
