import torch
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import torchvision.transforms as transforms  
from keras.utils import to_categorical
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

    # 标签onehot编码
    # y_train_onehot, y_test_onehot = to_categorical(lab_train, num_classes=10),to_categorical(lab_test, num_classes=10)
    # # scale data visual
    # t_visual_ave = MinMaxScaler()
    # # t_visual.fit(visual_train)
    # visual_train = t_visual_ave.fit_transform(visual_train)
    # visual_test = t_visual_ave.transform(visual_test)

    # t_audio_ave = MinMaxScaler()
    # # t_audio.fit(audio_train)
    # audio_train = t_audio_ave.fit_transform(audio_train)
    # audio_test = t_audio_ave.transform(audio_test)

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

    # scale data visual
    # t_visual_ave = MinMaxScaler()
    # # t_visual.fit(visual_train)
    # visual_train_scaled = t_visual_ave.fit_transform(visual_train)
    # visual_test_scaled = t_visual_ave.transform(visual_test)

    # t_audio_ave = MinMaxScaler()
    # # t_audio.fit(audio_train)
    # audio_train_scaled = t_audio_ave.fit_transform(audio_train)
    # audio_test_scaled = t_audio_ave.transform(audio_test)

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

    # scale data visual
    # t_visual_ave = MinMaxScaler()
    # # t_visual.fit(visual_train)
    # visual_train_scaled = t_visual_ave.fit_transform(visual_train)
    # visual_test_scaled = t_visual_ave.transform(visual_test)

    # t_audio_ave = MinMaxScaler()
    # # t_audio.fit(audio_train)
    # audio_train_scaled = t_audio_ave.fit_transform(audio_train)
    # audio_test_scaled = t_audio_ave.transform(audio_test)


    lab_test = torch.tensor(lab_test)
    lab_test = lab_test.view(lab_test.size(0),1)
    lab_test = lab_test.long()
    test_visual = TensorDataset(torch.tensor(visual_test).float(), lab_test)
    test_audio = TensorDataset(torch.tensor(audio_test).float(), lab_test)

    data_loader_visual = DataLoader(dataset=test_visual, batch_size=test_size, shuffle=False,drop_last=True)
    data_loader_audio = DataLoader(dataset=test_audio, batch_size=test_size, shuffle=False,drop_last=True)

    return data_loader_visual,data_loader_audio

if __name__ == "__main__":
    base_dir = "E:/Doctor-coder/multi-level-attention-2023/datasets/"
    # load_path =  base_dir +"vegas_feature.h5"
    load_path =  base_dir +"AVE_feature_updated_squence.h5"
    test_size = 128
    train_size = 128
    visual_test,audio_test = load_dataset_test(load_path,test_size)
    data_loader_visual,data_loader_audio = load_dataset_train(load_path,train_size)
    for epoch in range(2):
        for i, data in enumerate(zip(data_loader_visual, data_loader_audio)):
            inputs_visual = data[0][0].cuda()
            labels_visual = data[0][1].cuda()
            inputs_audio = data[1][0].cuda()
            labels_audio = data[1][1].cuda()
            
            print("epoch", epoch, "的第" , i, "个inputs", inputs_visual.shape, "labels", labels_audio.shape)

