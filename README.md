# visual-audio cross modal retrieval task

Code for Paper: DCLMA: Deep Correlation Learning with Multi-modal Attention for Visual-Audio Retrieval. Our code runs in the Windows 11 environment.
# Installation
## 1. Clone the repository
```bash
$ git clone https://github.com/zhangjiwei-japan/cross-modal-visual-audio-retrieval.git
```
## 2.Requirements
```bash
conda install --file requirements.txt
```
## 3. Prepare the datasets
### (1) AVE Dataset 
- AVE dataset can be downloaded from https://drive.google.com/open?id=1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK.
Audio feature and visual feature (7.7GB) are also released. Scripts for generating audio and visual features: https://drive.google.com/file/d/1TJL3cIpZsPHGVAdMgyr43u_vlsxcghKY/view?usp=sharing
  Please put the extracted data into the directory "datasets/ave/". 
### (2) VEGAS Dataset 
- The Raw dataset from: https://arxiv.org/abs/1712.01393.
  Please put the extracted data into the directory "datasets/vegas/". 
## 4. Execute train_model.py to train and evaluate the model as follows:
```bash
python train_model.py
```
Model parameters：
- `load_path`: which dataset "VEGAS" or "AVE" path.
- `lr`: initial learning rate.
- `batch_size`: train batch size.
- `optim`: optimizer "ADAM" or "SGD".
- `num_epochs`: set training epoch.
- `beta`: hyper-parameters of cross-modal correlation loss.
# Example
## Testing:
The DCLMA model in the paper can be tested as follows:
```bash
python test_model.py
```
- `save_path`: trained model path
- `test_size`: batch size of the test set
## Evelation: 
we use mAP as metrics to evaluate our architecture, when the system generates a ranked list in one modality for a query in another modality. Those documents in the ranked list with the same class are regarded as relevant or correct.
|Datasets    | Audio2Visual| Visual2Audio  | mAP |
| --------   | -----    | -----  |  -----  |
|#AVE      | ~0.450  | ~0.454 | ~0.452| 
|#VEGAS  | ~0.906 | ~ 0.888  | ~0.897 | 
## Contact
If you have any questions, please email s210068@wakayama-u.ac.jp
## Other Related or Follow-up works
Zhang, Jiwei, et al. "Variational Autoencoder with CCA for Audio-Visual Cross-Modal Retrieval." ACM Transactions on Multimedia Computing, Communications and Applications (2022).
