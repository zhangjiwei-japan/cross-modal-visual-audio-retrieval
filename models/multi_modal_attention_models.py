import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from math import sqrt
class Visual_Modality_Attention(nn.Module):
    # input : batch_size * seq_len * input_dim
    def __init__(self,input_dim,dim_k):
        super(Visual_Modality_Attention,self).__init__()
        self.q_v = nn.Linear(input_dim,dim_k)
        self.q_a = nn.Linear(input_dim,dim_k)
        self.k_v = nn.Linear(input_dim,dim_k)
        self.k_a = nn.Linear(input_dim,dim_k)

        self.joint_attention_map = nn.Bilinear(256, 256, 256, bias = False)
        self._norm_fact = 1 / sqrt(dim_k) 
        self.t_v = nn.Parameter(torch.ones([])) 
    
    def forward(self, img, audio):
        W_q_a = self.q_a(audio)  # Q: batch_size * seq_len * dim_k
        W_q_v = self.q_v(img)  # Q: batch_size * seq_len * dim_k
        W_k_a = self.k_a(audio)
        W_k_v = self.k_v(img) # Q: batch_size * seq_len * dim_k
         
        atten_vv = nn.Softmax(dim=-1)(torch.bmm(W_q_v,W_k_v.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
        atten_va = nn.Softmax(dim=-1)(torch.bmm(W_q_v,W_k_a.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len

        atten_B_kv = self.joint_attention_map(torch.bmm(atten_va,audio),torch.bmm(atten_vv,img))
        # output_i = torch.bmm(atten_c,img)
        atten_v = nn.Softmax(dim=-1)(torch.bmm(atten_B_kv,img.permute(0,2,1)))*self._norm_fact
        output_v = torch.add((self.t_v* torch.bmm(atten_v,img)), img)

        return output_v
    
class Audio_Modality_Attention(nn.Module):
    # input : batch_size * seq_len * input_dim
    def __init__(self,input_dim,dim_k):
        super(Audio_Modality_Attention,self).__init__()
        self.q_v = nn.Linear(input_dim,dim_k)
        self.q_a = nn.Linear(input_dim,dim_k)
        self.k_v = nn.Linear(input_dim,dim_k)
        self.k_a = nn.Linear(input_dim,dim_k)

        self.joint_attention_map = nn.Bilinear(256, 256, 256, bias = False)
        self._norm_fact = 1 / sqrt(dim_k) 
        self.t_a = nn.Parameter(torch.ones([]))
    
    def forward(self, img, audio):
        W_q_a = self.q_a(audio)  # Q: batch_size * seq_len * dim_k
        W_q_v = self.q_v(img)  # Q: batch_size * seq_len * dim_k
        w_k_a = self.k_a(audio)
        W_k_v = self.k_v(img) # Q: batch_size * seq_len * dim_k
         
        atten_aa = nn.Softmax(dim=-1)(torch.bmm(W_q_a,w_k_a.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
        atten_av = nn.Softmax(dim=-1)(torch.bmm(W_q_a,W_k_v.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len

        atten_B_ka = self.joint_attention_map(torch.bmm(atten_av,img),torch.bmm(atten_aa,audio))
        # output_i = torch.bmm(atten_c,img)
        atten_a = nn.Softmax(dim=-1)(torch.bmm(atten_B_ka,audio.permute(0,2,1)))*self._norm_fact
        output_a = torch.add((self.t_a* torch.bmm(atten_a,audio)), audio)

        return output_a
# class Multi_modal_Fusion_Attention(nn.Module):
#     # input : batch_size * seq_len * input_dim
#     def __init__(self,input_dim,dim_k):
#         super(Multi_modal_Fusion_Attention,self).__init__()
#         self.k_i = nn.Linear(input_dim,dim_k)
#         self.k_t = nn.Linear(input_dim,dim_k)

#         self.joint_attention_map = nn.Bilinear(256, 256, 256, bias = False)
#         self._norm_fact = 1 / sqrt(dim_k) 
    
#     def forward(self, img, audio):
#         W_i = self.k_i(img)  # Q: batch_size * seq_len * dim_k
#         W_t = self.k_t(audio) # Q: batch_size * seq_len * dim_k
         
#         atten_ti = nn.Softmax(dim=-1)(torch.bmm(audio,W_i.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
#         atten_ii = nn.Softmax(dim=-1)(torch.bmm(img,W_i.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len

#         atten_it = nn.Softmax(dim=-1)(torch.bmm(img,W_t.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
#         atten_tt = nn.Softmax(dim=-1)(torch.bmm(audio,W_t.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len

#         hadamard_product_v = atten_ti*atten_ii
#         hadamard_product_a = atten_it*atten_tt

#         common_feat = torch.add(img, audio)

#         feature_visual = torch.bmm(hadamard_product_v,common_feat)
#         feature_audio = torch.bmm(hadamard_product_a,common_feat)

#         atten_c = self.joint_attention_map(feature_visual,feature_audio)
#         j_c = F.sigmoid(atten_c)
#         Z_c = j_c*img + (1-j_c)*audio

#         return Z_c 
class Multi_modal_Fusion_Attention(nn.Module):
    # input : batch_size * seq_len * input_dim
    def __init__(self,input_dim,dim_k):
        super(Multi_modal_Fusion_Attention,self).__init__()
        self.q_v = nn.Linear(input_dim,dim_k)
        self.q_a = nn.Linear(input_dim,dim_k)
        self.k_v = nn.Linear(input_dim,dim_k)
        self.k_a = nn.Linear(input_dim,dim_k)

        self.w_1 = nn.Linear(input_dim,dim_k)
        self.w_2 = nn.Linear(input_dim,dim_k)
        self.w_3 = nn.Linear(input_dim,dim_k)
        self.w_4 = nn.Linear(input_dim,dim_k)

        self.joint_attention_map = nn.Bilinear(256, 256, 256, bias = False)
        self._norm_fact = 1 / sqrt(dim_k) 
        self.t_c = nn.Parameter(torch.ones([]))
    
    def forward(self, img, audio):
        W_q_v = self.q_v(img)  # Q: batch_size * seq_len * dim_k
        W_q_a = self.q_a(audio)  # Q: batch_size * seq_len * dim_k
        w_k_a = self.k_a(audio)
        W_k_v = self.k_v(img) # Q: batch_size * seq_len * dim_k
         
        atten_va = nn.Softmax(dim=-1)(torch.bmm(W_q_v, w_k_a.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
        atten_vv = nn.Softmax(dim=-1)(torch.bmm(W_q_v,W_k_v.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len

        atten_av = nn.Softmax(dim=-1)(torch.bmm(W_q_a,W_k_v.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
        atten_aa = nn.Softmax(dim=-1)(torch.bmm(W_q_a,w_k_a.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len

        alpha_va = torch.bmm(atten_va,audio)
        alpha_v = torch.bmm(atten_vv,img) 

        alpha_av = torch.bmm(atten_av,img)
        alpha_a = torch.bmm(atten_aa, audio) 

        c_k_va = F.sigmoid(torch.add(self.w_1(alpha_va),self.w_2(alpha_v))*alpha_v)
        c_k_av = F.sigmoid(torch.add(self.w_3(alpha_av),self.w_4(alpha_a))*alpha_a)
        atten_c = self.joint_attention_map(c_k_va,c_k_av)
        j_c = F.sigmoid(atten_c)
        Z_k = (self.t_c*j_c)*img + (1-j_c)*audio

        return Z_k 
    
class Multi_modal_Joint_Attention(nn.Module):
    # input : batch_size * seq_len * input_dim
    def __init__(self,input_dim,dim_k):
        super(Multi_modal_Joint_Attention,self).__init__()
        self.q_v = nn.Linear(input_dim,dim_k)
        self.q_a = nn.Linear(input_dim,dim_k)
        self.k_v = nn.Linear(input_dim,dim_k)
        self.k_a = nn.Linear(input_dim,dim_k)
        self.c_c = nn.Linear(input_dim,dim_k)

        self.w_5 = nn.Linear(input_dim,dim_k)
        self.w_6 = nn.Linear(input_dim,dim_k)
        self.w_7 = nn.Linear(input_dim,dim_k)
        self.w_8 = nn.Linear(input_dim,dim_k)

        self.joint_attention_map = nn.Bilinear(256, 256, 256, bias = False)
        self._norm_fact = 1 / sqrt(dim_k) 
        self.t_o = nn.Parameter(torch.ones([]))
    
    def forward(self, img, audio, common):
        W_q_v = self.q_v(img)  # Q: batch_size * seq_len * dim_k
        W_q_a = self.q_a(audio) # Q: batch_size * seq_len * dim_k
        w_k_a = self.k_a(audio)
        W_k_v = self.k_v(img) # Q: batch_size * seq_len * dim_k
        C_c = self.c_c(common)
  
        atten_vv = nn.Softmax(dim=-1)(torch.bmm(W_q_v,W_k_v.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
        atten_vc = nn.Softmax(dim=-1)(torch.bmm(W_q_v,C_c.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len

        atten_aa = nn.Softmax(dim=-1)(torch.bmm(W_q_a, w_k_a.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
        atten_ac = nn.Softmax(dim=-1)(torch.bmm(W_q_a,C_c.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
        
        common_feat_v = (1+ self.w_5(torch.bmm(atten_vc,common)))*torch.bmm(atten_vv,img) + self.w_6(torch.bmm(atten_vc,common))
        common_feat_a = (1+ self.w_7(torch.bmm(atten_ac,common)))*torch.bmm(atten_aa,audio) + self.w_8(torch.bmm(atten_ac,common))
        # common_feat_v = torch.add(torch.bmm(atten_vv,img), torch.bmm(atten_vc,common))
        # common_feat_a = torch.add(torch.bmm(atten_aa,audio), torch.bmm(atten_ac,common))

        atten_m = self.joint_attention_map(common_feat_v,common_feat_a)
        j_o = F.sigmoid(atten_m)
        Z_k_o = (self.t_o*j_o)*img + (1-j_o)*audio
        f_k_v_ = torch.add(Z_k_o, img)
        f_k_a_ = torch.add(Z_k_o, audio)

        return f_k_v_,f_k_a_
 
# class Cross_modal_Attention(nn.Module):
#     # input : batch_size * seq_len * input_dim
#     def __init__(self,input_dim,dim_k):
#         super(Cross_modal_Attention,self).__init__()
#         self.k_i = nn.Linear(input_dim,dim_k)
#         self.k_t = nn.Linear(input_dim,dim_k)

#         self.joint_attention_map = nn.Bilinear(1024, 1024, 1024, bias = False)
#         self._norm_fact = 1 / sqrt(dim_k) 
    
#     def forward(self, img, audio):
#         # K_i = self.k_i(img)  # Q: batch_size * seq_len * dim_k
#         # K_t = self.k_t(audio) # Q: batch_size * seq_len * dim_k 
#         K_i = torch.transpose(self.k_i(img),1,0)
#         K_t = torch.transpose(self.k_t(audio),1,0)
#         #
#         alpha_av = torch.matmul(audio, K_i)
#         alpha_av_norm= F.softmax(alpha_av, dim = 1)* self._norm_fact 

#         alpha_va = torch.matmul(img, K_t)
#         alpha_va_norm= F.softmax(alpha_va, dim = 1)* self._norm_fact

#         beta_a = torch.matmul(alpha_av_norm,self.k_t(audio))
#         beta_v = torch.matmul(alpha_va_norm,self.k_i(img))

#         atten_m = torch.add(beta_a,beta_v)
#         j_m = F.sigmoid(atten_m)
#         Z_g = j_m*img + (1-j_m)*audio

#         return Z_g

class Cross_modal_Attention(nn.Module):
    # input : batch_size *  input_dim
    def __init__(self,input_dim,dim_k):
        super(Cross_modal_Attention,self).__init__()
        self.q_v = nn.Linear(input_dim,dim_k)
        self.q_a = nn.Linear(input_dim,dim_k)
        self.k_v = nn.Linear(input_dim,dim_k)
        self.k_a = nn.Linear(input_dim,dim_k)
        self.w_9 = nn.Linear(input_dim,dim_k)
        self.w_10 = nn.Linear(input_dim,dim_k)
        self.w_11 = nn.Linear(input_dim,dim_k)
        self.w_12 = nn.Linear(input_dim,dim_k)

        self.joint_attention_map = nn.Bilinear(1024, 1024, 1024, bias = False)
        self._norm_fact = 1 / sqrt(dim_k)
        self.t_m = nn.Parameter(torch.ones([])) 
    
    def forward(self, img, audio):
        W_q_v = self.q_v(img)  # Q: batch_size *  dim_k
        W_q_a = self.q_a(audio)  # Q: batch_size *  dim_k
        w_k_a = self.k_a(audio)
        W_k_v = self.k_v(img) # Q: batch_size *  dim_k
        w_k_a_ = torch.transpose(w_k_a,1,0)#
        W_k_v_ = torch.transpose(W_k_v,1,0)#
        #
        alpha_va = torch.matmul(W_q_v, w_k_a_)
        alpha_va_norm= F.softmax(alpha_va, dim = 1)* self._norm_fact 

        alpha_av = torch.matmul(W_q_a, W_k_v_)
        alpha_av_norm= F.softmax(alpha_av, dim = 1)* self._norm_fact

        beta_va = torch.matmul(alpha_va_norm,audio)
        beta_av = torch.matmul(alpha_av_norm,img)

        beta_va = F.sigmoid(beta_va)
        beta_av = F.sigmoid(beta_av)

        feature_v = (1-beta_va)*img + beta_va*torch.add(img,audio)
        feature_a = (1-beta_av)*audio + beta_av*torch.add(audio,img)

        # atten_m = torch.add(feature_v,feature_a)
        C_g_v = F.sigmoid(torch.add(self.w_9(feature_v),self.w_10(feature_a)))*feature_a
        C_g_a = F.sigmoid(torch.add(self.w_11(feature_a),self.w_12(feature_v)))*feature_v
        j_m = F.sigmoid(self.joint_attention_map(C_g_v,C_g_a))
        Z_g = (self.t_m*j_m)*img + (1-j_m)*audio

        return Z_g
    
class Asymmetic_Attention(nn.Module):
    # input : batch_size * seq_len * input_dim
    def __init__(self,input_dim,dim_k):
        super(Asymmetic_Attention,self).__init__()
        self.k_i = nn.Linear(input_dim,dim_k)

    def forward(self, G_v, F_v):
        G_v_ = self.k_i(G_v)  # Q: batch_size * seq_len * dim_k

        alpha = torch.add(F_v, G_v_)
        j_m = F.sigmoid(alpha)
        M_v = j_m* F_v
        out_feature = torch.cat((M_v,G_v),1)

        return out_feature
