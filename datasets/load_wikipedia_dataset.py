import pickle as cPickle
import torch
import os, time
import sys
import numpy as np
from keras.utils import to_categorical
from base_model import BaseModel, BaseModelParams, BaseDataIter
# coding: utf-8


DATA_DIR = 'E:/Doctor-coder/cross-modal-dataset/'
class DataIter(BaseDataIter):
    def __init__(self, batch_size):
        BaseDataIter.__init__(self, batch_size)
        self.batch_size = batch_size
        self.num_train_batch = 0
        self.num_test_batch = 0

        with open(DATA_DIR+'wikipedia_dataset/train_img_feats.pkl', 'rb') as f:
            self.train_img_feats = cPickle.load(f,encoding='bytes')
        with open(DATA_DIR+'wikipedia_dataset/train_txt_vecs.pkl', 'rb') as f:
            self.train_txt_vecs = cPickle.load(f,encoding='bytes')
        with open(DATA_DIR+'wikipedia_dataset/train_labels.pkl', 'rb') as f:
            self.train_labels = cPickle.load(f,encoding='bytes')
        with open(DATA_DIR+'wikipedia_dataset/test_img_feats.pkl', 'rb') as f:
            self.test_img_feats = cPickle.load(f,encoding='bytes')
        with open(DATA_DIR+'wikipedia_dataset/test_txt_vecs.pkl', 'rb') as f:
            self.test_txt_vecs = cPickle.load(f,encoding='bytes')
        with open(DATA_DIR+'wikipedia_dataset/test_labels.pkl', 'rb') as f:
            self.test_labels = cPickle.load(f,encoding='bytes')
        
   
        self.num_train_batch = len(self.train_img_feats) // self.batch_size
        self.num_test_batch = len(self.test_img_feats) // self.batch_size

    def train_data(self):
        for i in range(self.num_train_batch):
            batch_img_feats = self.train_img_feats[i*self.batch_size : (i+1)*self.batch_size]
            batch_txt_vecs = self.train_txt_vecs[i*self.batch_size : (i+1)*self.batch_size]
            batch_labels = self.train_labels[i*self.batch_size : (i+1)*self.batch_size]
            # train_onehot = to_categorical(batch_labels-1, num_classes=10)
            yield batch_img_feats, batch_txt_vecs, batch_labels, i
            # yield batch_img_feats, batch_txt_vecs, train_onehot, i

    def test_data(self):
        for i in range(self.num_test_batch):
            batch_img_feats = self.test_img_feats[i*self.batch_size : (i+1)*self.batch_size]
            batch_txt_vecs = self.test_txt_vecs[i*self.batch_size : (i+1)*self.batch_size]
            batch_labels = self.test_labels[i*self.batch_size : (i+1)*self.batch_size]
            # 标签onehot编码
            # test_onehot = to_categorical(batch_labels-1, num_classes=10)
            yield batch_img_feats, batch_txt_vecs, batch_labels, i
            # yield batch_img_feats, batch_txt_vecs, test_onehot, i

def list_2_tensor(input):
    target = np.array(input)
    output = torch.from_numpy(target)
    return output  # list转换为tensors列表格式的数据类型。

def train():
    num_epoch = 1
    batch_size = 64
    visual_feat_dim = 4096
    word_vec_dim = 5000
    data_iter = DataIter(batch_size)
    for epoch in range(num_epoch):
        for feats, vecs, labels, i in data_iter.train_data():
            if torch.cuda.is_available():
                    feats = list_2_tensor(feats)
                    feats = feats.cuda()
                    vecs= list_2_tensor(vecs)
                    vecs = vecs.cuda()
                    labels = list_2_tensor(labels)
                    labels = labels.cuda()
                    print("image shape",feats.shape)
                    print("txts shape",vecs.shape)

def eval():
    batch_size = 64
    data_iter = DataIter(batch_size)
    for feats, vecs, labels, i in data_iter.test_data():
        if torch.cuda.is_available():
                    feats = list_2_tensor(feats)
                    feats = feats.cuda()
                    vecs= list_2_tensor(vecs)
                    vecs = vecs.cuda()
                    labels = list_2_tensor(labels)
                    labels = labels.cuda()
                    print("image shape",feats.shape)
                    print("txts shape",vecs.shape)
        # print("feature shape:",len(feats))

# 2,173 462 10 4,096 300
# if __name__ == '__main__':
#     # train()
#     eval()
