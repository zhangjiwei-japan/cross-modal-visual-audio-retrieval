import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from attention_module import Attention_Layer,Cross_Attention_Layer
class Classifier(nn.Module):
    def __init__(self,latent_dim=1024,out_label=10,kaiming_init=False):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128, bias=False),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(128,out_label, bias=False))
        if kaiming_init:
            self._init_weights_classifier()

    def _init_weights_classifier(self):
        for  m in self._modules:
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.classifier(x)
        return x #nn.CrossEntropyLoss()
        # return F.softmax(x, dim=1)
        # return F.log_softmax(x, dim = 1) #F.nll_loss

class ImgNN(nn.Module):
    """Network to learn image representations"""
    def __init__(self, input_dim=4096, output_dim=2048,kaiming_init=False):
        super(ImgNN, self).__init__()
        self.visual_encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=False),   # 1024 512
            # nn.LayerNorm(mid_dim),
            # nn.BatchNorm1d(mid_dim, affine=False,track_running_stats=False), #使移动均值和移动方差不起作用
            nn.BatchNorm1d(output_dim, affine=False),
            nn.Dropout(0.5),
            nn.Linear(output_dim, output_dim, bias=False)# 512 128
        )
        if kaiming_init:
            self._init_weights_img()
    def _init_weights_img(self):
        for  m in self._modules:
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                # nn.init.constant_(m.bias, 0.0)
            elif isinstance(m,nn.BatchNorm1d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
    def forward(self, x):
        out = self.visual_encoder(x)

        return out

class TextNN(nn.Module):
    """Network to learn text representations"""
    def __init__(self, input_dim=1024, output_dim=2048,kaiming_init=False):
        super(TextNN, self).__init__()
        self.text_encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),   # 128 128
            # nn.LayerNorm(output_dim),
            # nn.BatchNorm1d(output_dim, affine=False,track_running_stats=False), #使移动均值和移动方差不起作用
            nn.BatchNorm1d(input_dim, affine=False), 
            nn.Dropout(0.5),
            nn.Linear(input_dim, output_dim, bias=False)    # 128 128
        )
        if kaiming_init:
            self._init_weights_text()
    def _init_weights_text(self):
        for  m in self._modules:
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                # nn.init.constant_(m.bias, 0.0)
            elif isinstance(m,nn.BatchNorm1d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
    def forward(self, x):
        out = self.text_encoder(x)
        return out

class CrossModal_NN(nn.Module):
    """Network to learn text representations"""
    def __init__(self, img_input_dim=4096, img_output_dim=2048,
                 text_input_dim=1024, text_output_dim=2048, minus_one_dim=1024, output_dim=10, kaiming_init=False):
        super(CrossModal_NN, self).__init__()
        mid_dim =512
        self.kaiming_init = kaiming_init # 模型初始化
        self.visual_layer = ImgNN(input_dim=img_input_dim, output_dim=img_output_dim,kaiming_init=self.kaiming_init)
        self.text_layer = TextNN(input_dim=text_input_dim, output_dim=text_output_dim,kaiming_init=self.kaiming_init)
        self.linearLayer = nn.Linear(img_output_dim, minus_one_dim)
        self.cross_attention = Cross_Attention_Layer(minus_one_dim)
        self.co_attention = Attention_Layer(mid_dim)
        self.linearLayer2 = nn.Linear( minus_one_dim,mid_dim)

        self.classifier_t = Classifier(latent_dim=mid_dim,out_label=output_dim,kaiming_init=self.kaiming_init)
        self.classifier_v = Classifier(latent_dim=mid_dim,out_label=output_dim,kaiming_init=self.kaiming_init)

    def forward(self, img, text):
        visual_feature = self.visual_layer(img)
        text_feature = self.text_layer(text)
        visual_feature = self.linearLayer(visual_feature)
        
        text_feature = self.linearLayer(text_feature)
        
        _,visual_feature = self.cross_attention(visual_feature,text_feature)
        _,text_feature = self.cross_attention(text_feature,visual_feature)
        visual_feature = self.linearLayer2(visual_feature)
        text_feature = self.linearLayer2(text_feature)

        fusion_feat = torch.add(visual_feature, text_feature)
        co_attention_map,_ = self.co_attention(fusion_feat)
        visual_feature = torch.add(co_attention_map,visual_feature)  
        text_feature = torch.add(co_attention_map,text_feature)


        visual_predict = self.classifier_v(visual_feature)
        text_predict = self.classifier_t(text_feature)
        return visual_feature, text_feature, visual_predict, text_predict
