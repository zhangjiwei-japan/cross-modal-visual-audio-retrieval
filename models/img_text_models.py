import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_module import Attention_Layer,Cross_Attention_Layer
     
classifier_criterion = nn.CrossEntropyLoss().cuda()
class Classifier(nn.Module):
    def __init__(self,latent_dim,num_classes=2):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            #全链接
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.classifier(x)
        # return F.softmax(x,dim=1)
        return x
        # return F.log_softmax(x,dim=1)

class Visual_Encoder(nn.Module):
    def __init__(self,img_input_dim, img_output_dim):
        super(Visual_Encoder, self).__init__()
        visual_mid_dim = 2048
        visual_final_dim = 512
        self.visual_encoder = nn.Sequential(
            nn.Linear(img_input_dim, visual_mid_dim),   # 4096 2048
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(visual_mid_dim,visual_final_dim), # 2048 512
            nn.BatchNorm1d(visual_final_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(visual_final_dim, img_output_dim),   # 512 256
        )
        
    def forward(self, img):
        visual_feature = self.visual_encoder(img)

        return visual_feature 

class Text_Encoder(nn.Module):
    def __init__(self,text_input_dim, text_output_dim):
        super(Text_Encoder, self).__init__()
        text_mid_dim = 256
        self.text_Encoder = nn.Sequential(
            nn.Linear(text_input_dim, text_mid_dim),   # 300 256
            nn.BatchNorm1d(text_mid_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(text_mid_dim, text_output_dim),    # 256 256
        )
        
    def forward(self,text):
        text_feature = self.text_Encoder(text)

        return text_feature 
    
class Cross_Modal_Net(nn.Module):
    def __init__(self, img_input_dim=4096, img_output_dim=256,
                 text_input_dim=300, text_output_dim=256, \
                 minus_one_dim=256,minus_two_dim=128,minus_three_dim=64, output_class_dim=10):
        super(Cross_Modal_Net, self).__init__()
        self.visual_encoder = Visual_Encoder(img_input_dim,img_output_dim)
        self.text_encoder = Text_Encoder(text_input_dim,text_output_dim)
        self.cross_attention_one = Cross_Attention_Layer(minus_one_dim)
        self.cross_attention_two = Cross_Attention_Layer(minus_two_dim)
        self.co_attention = Attention_Layer(minus_three_dim)
        self.linear_Layer_one = nn.Linear(text_output_dim,minus_one_dim)
        self.linear_Layer_two = nn.Linear(minus_one_dim,minus_two_dim)
        self.linear_Layer_three = nn.Linear(minus_two_dim,minus_three_dim)
        self.classifier_visual = Classifier(minus_three_dim,num_classes=output_class_dim)
        self.classifier_text = Classifier(minus_three_dim,num_classes=output_class_dim)

    def forward(self, img, text):
        visual_feature = self.visual_encoder(img)
        text_feature = self.text_encoder(text)
        visual_feature_one = self.linear_Layer_one(visual_feature)
        text_feature_one = self.linear_Layer_one(text_feature)
        _,visual_feature_one = self.cross_attention_one(visual_feature_one,text_feature_one)
        _,text_feature_one = self.cross_attention_one(text_feature_one,visual_feature_one)
        visual_feature_two = self.linear_Layer_two(visual_feature_one)
        text_feature_two = self.linear_Layer_two(text_feature_one)
        _,visual_feature_two = self.cross_attention_two(visual_feature_two,text_feature_two)
        _,text_feature_two = self.cross_attention_two(text_feature_two,visual_feature_two)
        visual_feature_three = self.linear_Layer_three(visual_feature_two)
        text_feature_three = self.linear_Layer_three(text_feature_two)
        fusion_feat = torch.add(visual_feature_three, text_feature_three)
        co_attention_map,_ = self.co_attention(fusion_feat)
        visual_feature_out = torch.add(co_attention_map,visual_feature_three)  
        text_feature_out = torch.add(co_attention_map,text_feature_three)

        visual_predict = self.classifier_visual(visual_feature_out)
        text_predict = self.classifier_text(text_feature_out)

        return visual_feature_out,text_feature_out,visual_predict,text_predict


class Cross_Modal_Base_Net(nn.Module):
    def __init__(self, img_input_dim=4096, img_output_dim=256,
                 text_input_dim=300, text_output_dim=256, \
                 minus_one_dim=128,minus_two_dim=64, output_class_dim=10):
        super(Cross_Modal_Base_Net, self).__init__()
        self.visual_encoder = Visual_Encoder(img_input_dim,img_output_dim)
        self.text_encoder = Text_Encoder(text_input_dim,text_output_dim)
        self.linear_Layer_one = nn.Linear(text_output_dim,minus_one_dim)
        self.linear_Layer_two = nn.Linear(minus_one_dim,minus_two_dim)
        self.classifier_visual = Classifier(minus_two_dim,num_classes=output_class_dim)
        self.classifier_text = Classifier(minus_two_dim,num_classes=output_class_dim)

    def forward(self, img, text):
        visual_feature = self.visual_encoder(img)
        text_feature = self.text_encoder(text)
        visual_feature_one = self.linear_Layer_one(visual_feature)
        text_feature_one = self.linear_Layer_one(text_feature)
        visual_feature_two = self.linear_Layer_two(visual_feature_one)
        text_feature_two = self.linear_Layer_two(text_feature_one)

        visual_predict = self.classifier_visual(visual_feature_two)
        text_predict = self.classifier_text(text_feature_two)

        return visual_feature_two,text_feature_two,visual_predict,text_predict

    
