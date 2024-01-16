import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import models
from models.multi_scale_net import MultiScale_Modal_Net
from models.multi_modal_attention_models import *

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            nn.init.zeros_(m.bias.data)

class Classifier(nn.Module):
    def __init__(self,latent_dim=1024,out_label=10):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(64,out_label, bias =False))

    def forward(self, x):
        x = self.classifier(x)
        return x #nn.CrossEntropyLoss()
        # return F.softmax(x, dim=1)
        # return F.log_softmax(x, dim = 1) #F.nll_loss

class ImgNN(nn.Module):
    """Network to learn image representations"""
    def __init__(self, input_dim=4096, output_dim=2048):
        super(ImgNN, self).__init__()
        self.visual_encoder = nn.Linear(input_dim, output_dim)
        # self.visual_encoder = nn.Sequential(
        #     nn.Linear(input_dim, output_dim),   # 1024 512
        #     # nn.LayerNorm(mid_dim),
        #     # nn.BatchNorm1d(mid_dim, affine=False,track_running_stats=False), 
        #     # nn.BatchNorm1d(output_dim, affine=False),
        #     # nn.Dropout(0.5),
        #     nn.Linear(output_dim, output_dim)# 512 128
        # )

    def forward(self, x):
        out = F.relu(self.visual_encoder(x))

        return out

class AudioNN(nn.Module):
    """Network to learn audio representations"""
    def __init__(self, input_dim=1024, output_dim=2048):
        super(AudioNN, self).__init__()
        self.audio_encoder =  nn.Linear(input_dim, output_dim) 
        # self.audio_encoder = nn.Sequential(
        #     nn.Linear(input_dim, input_dim),   # 128 128
        #     # nn.LayerNorm(output_dim),
        #     # nn.BatchNorm1d(output_dim, affine=False,track_running_stats=False), 
        #     # nn.BatchNorm1d(input_dim, affine=False), 
        #     # nn.Dropout(0.5),
        #     nn.Linear(input_dim, output_dim)    # 128 128
        # )

    def forward(self, x):
        out = F.relu(self.audio_encoder(x))
        return out
 

class CrossModal_NN(nn.Module):
    """Network to learn audio representations"""
    
    def __init__(self, img_input_dim=1024, img_output_dim=1024,
                 audio_input_dim=128, audio_output_dim=1024, minus_one_dim=128, output_dim=10):
        super(CrossModal_NN, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) 
        self.visual_layer = ImgNN(input_dim= img_input_dim, output_dim= img_output_dim)
        self.audio_layer = AudioNN(input_dim= audio_input_dim, output_dim= audio_output_dim)
        # self.visual_global_feature = nn.AdaptiveMaxPool1d(output_size = img_output_dim, return_indices=False)
        # self.audio_global_feature = nn.AdaptiveMaxPool1d(output_size = audio_output_dim, return_indices=False)
        self.v_att = Visual_Modality_Attention(256,256)
        self.a_att = Audio_Modality_Attention(256,256)
        self.MmFA = Multi_modal_Fusion_Attention(256,256)
        self.CmJA = Multi_modal_Joint_Attention(256,256)
        self.CmA = Cross_modal_Attention(img_output_dim,audio_output_dim)
        self.asy_att = Asymmetic_Attention(img_input_dim,768)
        self.visual_global_feature = nn.AdaptiveAvgPool1d(output_size = img_output_dim)
        self.audio_global_feature = nn.AdaptiveAvgPool1d(output_size = audio_output_dim)
        self.visual_fine_grained_feature =  MultiScale_Modal_Net(num_features=1)  #  torch.Size([32, 3, 256])
        self.audio_fine_grained_feature =  MultiScale_Modal_Net(num_features=1) #  torch.Size([32, 3, 256])

        self.out_layer = nn.Sequential(
            nn.Linear(in_features=1792, out_features = 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features = minus_one_dim)
            )
        self.classifier_audio = Classifier(latent_dim= minus_one_dim,out_label= output_dim)
        self.classifier_visual = Classifier(latent_dim= minus_one_dim,out_label= output_dim)

    def forward(self, img, audio):
        batch_size = img.size(0)
        dimanal = img.size(1)
        # visual-audio encoder 
        visual_feature = self.visual_layer(img)
        audio_feature = self.audio_layer(audio)
        # GLR
        visual_global_feature = self.visual_global_feature(visual_feature)
        audio_global_feature = self.audio_global_feature(audio_feature)
        # FGR
        visual_multi_scale_feature = self.visual_fine_grained_feature(visual_feature)
        audio_multi_scale_feature = self.audio_fine_grained_feature(audio_feature)
        # visual/audio modality attention 
        visual_multi_scale_fine = self.v_att(visual_multi_scale_feature,audio_multi_scale_feature)
        audio_multi_scale_fine = self.a_att(visual_multi_scale_feature,audio_multi_scale_feature)
        # two-stage fusion, multi-modal fusion attention
        Z_1_k = self.MmFA(visual_multi_scale_fine,audio_multi_scale_fine)
        I_k_v = Z_1_k*visual_multi_scale_fine
        I_k_a = Z_1_k*audio_multi_scale_fine
        Z_2_k = self.MmFA(I_k_v,I_k_a)
        # obtain fusion feature: visual_audio_fusion_feature (f_k_c)
        visual_audio_fusion_feature = torch.add(Z_2_k*visual_multi_scale_fine, Z_2_k*audio_multi_scale_fine)
        # multi-modak joint attention
        F_visual_,F_audio_ = self.CmJA(visual_multi_scale_fine,audio_multi_scale_fine,visual_audio_fusion_feature)

        # multi-stage cross-modal attention for visual-audio modality global-level feature 
        Z_1_g = self.CmA(visual_global_feature,audio_global_feature)
        I_g_v = Z_1_g*visual_global_feature
        I_g_a = Z_1_g*audio_global_feature
        Z_2_g = self.CmA(I_g_v,I_g_a)
        G_v_ = Z_2_g*torch.add(visual_global_feature,I_g_v)
        G_a_ = Z_2_g*torch.add(audio_global_feature,I_g_a)
        
        # asymmetric attention module: global-level feature and fine-grained feature
        S_v = self.asy_att(G_v_,F_visual_.view(batch_size, -1))
        S_a = self.asy_att(G_a_,F_audio_.view(batch_size, -1))
        # S_v = torch.cat((visual_global_feature,M_visual.view(batch_size, -1)),1)
        # S_a = torch.cat((audio_global_feature,M_audio.view(batch_size, -1)),1)
        visual_global_fine_feature = S_v
        audio_global_fine_feature = S_a
        
        # output feature
        out_visual_feature = self.out_layer(visual_global_fine_feature)
        out_audio_feature = self.out_layer(audio_global_fine_feature)
        # out_visual_feature = F.normalize(out_visual_feature, dim=-1)
        # out_audio_feature = F.normalize(out_audio_feature, dim=-1)

        visual_predict = self.classifier_visual(out_visual_feature)
        audio_predict = self.classifier_visual(out_audio_feature)

        return out_visual_feature, out_audio_feature, visual_predict, audio_predict,self.logit_scale.exp()

