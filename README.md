# visual-audio cross modal retrieval task

Code for Paper: DCLMA: Deep Correlation Learning with Multi-modal Attention for Visual-Audio Retrieval. Our code runs in the Windows 11 environment.
# Installation
## 1. Clone the repository
```bash
$ git clone https://github.com/zhangjiwei-japan/cross-modal-visual-audio-retrieval.git
```
## 2. Requirements
#### （1） Install python from website ：https://www.python.org/downloads/windows/
#### （2） Our program runs on the GPU. Please install the cuda, cudnn, etc as follows : 
- CUDA Toolkit Archive ：https://developer.nvidia.com/cuda-toolkit-archive
- cuDNN Download | NVIDIA Developer ：https://developer.nvidia.com/login
- PYTORCH : https://pytorch.org/
#### （3） Libraries required to install the program ：
```bash
pip install -r requirements.txt
```
## 3. Prepare the datasets
### (1) AVE Dataset 
- AVE dataset can be downloaded from https://drive.google.com/open?id=1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK.
- Scripts for generating audio and visual features: https://drive.google.com/file/d/1TJL3cIpZsPHGVAdMgyr43u_vlsxcghKY/view?usp=sharing
#### You can also download our prepared [AVE](https://drive.google.com/file/d/14Qdprd8_9cdih3QDN726kJTzaoo9Y8Y-/view?usp=sharing) dataset.
- Please create a folder named 'ave' to place the downloaded dataset and set the dataset base path in the code (train_model.py and test_model.py): `base_dir` = "./datasets/ave/"
- Place the downloaded dataset in the 'ave' file and load a dataset path in the code (train_model.py and test_model.py): `load_path` = `base_dir` + "your downloaded dataset"
### (2) VEGAS Dataset 
- The Raw dataset from: https://arxiv.org/abs/1712.01393.
#### You can also download our prepared [VEGAS](https://drive.google.com/file/d/142VXU9-3P2HcaCWCQVlezRGJguGnHeHD/view?usp=sharing) dataset. 
- Please create a folder named 'veags' to place the downloaded dataset and set the dataset base path in the code (train_model.py and test_model.py): `base_dir` = "./datasets/vegas/"
- Place the downloaded dataset in the 'veags' file and load a dataset path in the code (train_model.py and test_model.py): `load_path` = `base_dir` + "your downloaded dataset"
## 4. Execute train_model.py to train and evaluate the model as follows :
```bash
python train_model.py
```
#### Only the following parameters need to be modified when running train_model.py : 
- `Lr`: initial learning rate.
- `batch_size`: train batch size.
- `dataset`: dataset name "vegas or ave".
- `num_epochs`: set training epoch.
- `class_dim`: vegas dataset class_dim = 10, ave dataset class_dim = 15. 
- `beta`: hyper-parameters of cross-modal correlation loss.
# Example
## 1. Testing :
The DCLMA model in the paper can be tested as follows :
```bash
python test_model.py
```
#### Only the following parameters need to be modified when running test_model.py :
- `save_path`: load trained model path.
- `dataset`: dataset name "vegas or ave".
- `test_size`: batch size of the test set.
- `class_dim`: vegas dataset class_dim = 10, ave dataset class_dim = 15. 
## 2. Evelation : 
we use mAP as metrics to evaluate our architecture, when the system generates a ranked list in one modality for a query in another modality. Those documents in the ranked list with the same class are regarded as relevant or correct.
|Datasets    | Audio2Visual| Visual2Audio  | mAP |
| --------   | -----    | -----  |  -----  |
|#AVE      | ~0.507  | ~0.508 | ~0.508| 
|#VEGAS  | ~0.933 | ~0.931  | ~0.932 | 
## Contact
If you have any questions, please email s210068@wakayama-u.ac.jp
## Other Related or Follow-up works
Zhang, Jiwei, et al. "Variational Autoencoder with CCA for Audio-Visual Cross-Modal Retrieval." ACM Transactions on Multimedia Computing, Communications and Applications (2022).
