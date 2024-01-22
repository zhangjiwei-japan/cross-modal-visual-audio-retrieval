import pickle as cPickle
import torch
import os, time
import sys
import numpy as np
from base_model import BaseModel, BaseModelParams, BaseDataIter
# coding: utf-8

DATA_DIR = 'E:/Doctor-coder/cross-modal-dataset/' 

class DataIter(BaseDataIter):
    def __init__(self, batch_size):
        BaseDataIter.__init__(self, batch_size)
        self.batch_size = batch_size
        self.num_train_batch = 0
        self.num_test_batch = 0

        with open(DATA_DIR+'nuswide/img_train_id_feats.pkl', 'rb') as f:
            self.train_img_feats = cPickle.load(f,encoding='bytes')
        with open(DATA_DIR+'nuswide/train_id_bow.pkl', 'rb') as f:
            self.train_txt_vecs = cPickle.load(f,encoding='bytes')
        with open(DATA_DIR+'nuswide/train_id_label_map.pkl', 'rb') as f:
            self.train_labels = cPickle.load(f,encoding='bytes')
        with open(DATA_DIR+'nuswide/img_test_id_feats.pkl', 'rb') as f:
            self.test_img_feats = cPickle.load(f,encoding='bytes')
        with open(DATA_DIR+'nuswide/test_id_bow.pkl', 'rb') as f:
            self.test_txt_vecs = cPickle.load(f,encoding='bytes')
        with open(DATA_DIR+'nuswide/test_id_label_map.pkl', 'rb') as f:
            self.test_labels = cPickle.load(f,encoding='bytes')
        with open(DATA_DIR+'nuswide/train_ids.pkl', 'rb') as f:
            self.train_ids = cPickle.load(f,encoding='bytes')
        with open(DATA_DIR+'nuswide/test_ids.pkl', 'rb') as f:
            self.test_ids = cPickle.load(f,encoding='bytes')  
        with open(DATA_DIR+'nuswide/train_id_label_single.pkl', 'rb') as f:
            self.train_labels_single = cPickle.load(f,encoding='bytes')
        with open(DATA_DIR+'nuswide/test_id_label_single.pkl', 'rb') as f:
            self.test_labels_single = cPickle.load(f,encoding='bytes')              
                
        np.random.shuffle(self.train_ids)
        np.random.shuffle(self.test_ids)          
        self.num_train_batch = len(self.train_ids) // self.batch_size
        self.num_test_batch = len(self.test_ids) // self.batch_size


    def train_data(self):
        # print(self.num_train_batch)
        for i in range(self.num_train_batch):
            batch_img_ids = self.train_ids[i*self.batch_size : (i+1)*self.batch_size]
            batch_img_feats = [self.train_img_feats[n] for n in batch_img_ids]
            batch_txt_vecs = [self.train_txt_vecs[n] for n in batch_img_ids]
            batch_labels = [self.train_labels[n] for n in batch_img_ids]
            batch_labels_single = np.array([self.train_labels_single[n] for n in batch_img_ids])
            yield batch_img_feats, batch_txt_vecs, batch_labels, batch_labels_single, i

    def test_data(self):
        for i in range(self.num_test_batch):
            batch_img_ids = self.test_ids[i*self.batch_size : (i+1)*self.batch_size]
            batch_img_feats = [self.test_img_feats[n] for n in batch_img_ids]
            batch_txt_vecs = [self.test_txt_vecs[n] for n in batch_img_ids]
            batch_labels = [self.test_labels[n] for n in batch_img_ids]
            batch_labels_single = [self.test_labels_single[n] for n in batch_img_ids]
            yield batch_img_feats, batch_txt_vecs, batch_labels, batch_labels_single, i
def list_2_tensor(input):
    target = np.array(input)
    output = torch.from_numpy(target)
    return output  # list转换为tensors列表格式的数据类型。
def train():
    batch_size = 64
    num_epochs = 1
    data_iter = DataIter(batch_size)
    for epoch in range(num_epochs):  
        for batch_feat, batch_vec, batch_labels, batch_labels_single, idx in data_iter.train_data():
            if torch.cuda.is_available():
                batch_feat = list_2_tensor(batch_feat).cuda()  # TENSORS ARE CUDA TYPES. 这里需要将它们传递到GPU上。
                batch_vec = list_2_tensor(batch_vec).cuda()  # 也需要将它们传递到GPU上。 这里需要
                batch_labels = 	list_2_tensor(batch_labels).cuda()  # 传递到GPU上。  这也需要传递
                batch_labels_single = list_2_tensor(batch_labels_single).cuda()  # 到CPU上。  这也
                print("idx:", idx)
                print("Feature shapes: ", batch_feat.shape)
                print("Vec shapes: ", batch_vec.shape)
                print("Label shapes: ", batch_labels.shape)
                print("Label single shapes: ", batch_labels_single.shape)

def eval():
    batch_size = 64
    data_iter = DataIter(batch_size)
    for feats, vecs, _, labels, i in data_iter.test_data():
        if torch.cuda.is_available():
                    feats = list_2_tensor(feats)
                    feats = feats.cuda()
                    vecs= list_2_tensor(vecs)
                    vecs = vecs.cuda()
                    labels = list_2_tensor(labels)
                    labels = labels.cuda()
                    print("image shape",feats.shape)
                    print("txts shape",vecs.shape)
        # print("Image feature shapes: ", len(feats))


# #8,000 1,000 10 4,096 300
# if __name__ == '__main__':
#     train()
    # eval()
     

