# visual-audio cross modal retrieval task

Code for Paper: DCLMA: Deep Correlation Learning with Multi-modal Attention for Visual-Audio Retrieval.

AVE Dataset & Features
AVE dataset can be downloaded from https://drive.google.com/open?id=1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK.
Audio feature and visual feature (7.7GB) are also released. Please put videos of AVE dataset into /data/AVE folder and features into /data folder before running the code.
Scripts for generating audio and visual features: https://drive.google.com/file/d/1TJL3cIpZsPHGVAdMgyr43u_vlsxcghKY/view?usp=sharing (Feel free to modify and use it to precess your audio and visual data).

VEGAS Dataset & Features
Requirements
Python-3.6, Pytorch-0.3.0, Keras, ffmpeg.

Visualize attention maps
## Data
Download routines for the datasets are not provided in the repository. Please download and prepare the datasets yourself according to our paper:
- [VEGAS Dataset](https://drive.google.com/file/d/1EjRDkgiXzAR8thouBVJrj7hQg2WBUZ88/view?usp=share_link)
- [AVE Dataset](https://drive.google.com/file/d/1EjsbGoFZ2mCHNeVYmf45Kb4tNwTLV86o/view?usp=share_link)
- Original Dataset homepage: https://sites.google.com/view/audiovisualresearch and https://github.com/YapengTian/AVE-ECCV18

## Contact
If you have any questions, please email s210068@wakayama-u.ac.jp
## Reference
Zhang, Jiwei, et al. "Variational Autoencoder with CCA for Audio-Visual Cross-Modal Retrieval." ACM Transactions on Multimedia Computing, Communications and Applications (2022).
