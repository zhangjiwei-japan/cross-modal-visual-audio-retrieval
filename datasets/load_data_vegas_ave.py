import torch
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import torchvision.transforms as transforms  
import pandas as pd  
import h5py
import os  

def load_dataset(load_path):
    f = h5py.File(load_path,'r')
    f.keys()
    lab_test = f["lab_test"]
    visual_test = f["visual_test"]
    audio_test = f["audio_test"]
    visual_train = f["visual_train"]
    audio_train = f["audio_train"]
    lab_train = f["lab_train"] 

    return visual_train,audio_train,visual_test,audio_test,lab_train,lab_test           


def load_dataset_train(load_path,train_size):
    f = h5py.File(load_path,'r')
    f.keys()
    lab_test = f["lab_test"]
    visual_test = f["visual_test"]
    audio_test = f["audio_test"]
    visual_train = f["visual_train"]
    audio_train = f["audio_train"]
    lab_train = f["lab_train"] 

    lab_train = torch.tensor(lab_train)
    lab_train = lab_train.view(lab_train.size(0),1)
    lab_train = lab_train.long()
    train_visual = TensorDataset(torch.tensor(visual_train).float(), lab_train)
    train_audio = TensorDataset(torch.tensor(audio_train).float(), lab_train)

    data_loader_visual = DataLoader(dataset=train_visual, batch_size=train_size, shuffle=False,drop_last=True)
    data_loader_audio = DataLoader(dataset=train_audio, batch_size=train_size, shuffle=False,drop_last=True)

    return data_loader_visual,data_loader_audio


def load_dataset_test(load_path,test_size):
    f = h5py.File(load_path,'r')
    f.keys()
    lab_test = f["lab_test"]
    visual_test = f["visual_test"]
    audio_test = f["audio_test"]
    visual_train = f["visual_train"]
    audio_train = f["audio_train"]
    lab_train = f["lab_train"] 


    lab_test = torch.tensor(lab_test)
    lab_test = lab_test.view(lab_test.size(0),1)
    lab_test = lab_test.long()
    test_visual = TensorDataset(torch.tensor(visual_test).float(), lab_test)
    test_audio = TensorDataset(torch.tensor(audio_test).float(), lab_test)

    data_loader_visual = DataLoader(dataset=test_visual, batch_size=test_size, shuffle=False,drop_last=True)
    data_loader_audio = DataLoader(dataset=test_audio, batch_size=test_size, shuffle=False,drop_last=True)

    return data_loader_visual,data_loader_audio


