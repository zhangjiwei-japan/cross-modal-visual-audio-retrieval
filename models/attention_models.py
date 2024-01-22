import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from math import sqrt
class Attention_Layer(nn.Module):
    #用来实现mask-attention layer
    def __init__(self, input_dim,dim_k,dim_v):
        super(Attention_Layer,self).__init__()
        
        # self.hidden_dim = hidden_dim
        self.Q_linear = nn.Linear(input_dim,dim_k, bias = False)
        self.K_linear = nn.Linear(input_dim,dim_k, bias = False)
        self.V_linear = nn.Linear(input_dim,dim_v, bias = False)
                 
    def forward(self, inputs):
        
        #计算生成QKV矩阵
        Q = self.Q_linear(inputs) 
        K = self.K_linear(inputs).permute(0, 2, 1)#先进行一次转置
        # K = torch.transpose(self.K_linear(inputs),1,0)#先进行一次转置
        V = self.V_linear(inputs) 
        #下面开始计算啦
        alpha = torch.matmul(Q, K)
        #下面开始softmax
        alpha = F.softmax(alpha, dim = 1)
        #print('\nalpha is :', alpha)
        out = torch.matmul(alpha, V)
        feature = torch.add(out,inputs)
        # return feature_map
        return out,feature
    
class Joint_Attention_Layer(nn.Module):
    #用来实现mask-attention layer
    def __init__(self, hidden_dim):
        super(Joint_Attention_Layer,self).__init__()
        self.hidden_dim = hidden_dim
        self.Q_linear = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.K_linear = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self._norm_fact = 1 / sqrt(hidden_dim) 

    def forward(self, img, text):
        #计算生成QKV矩阵
        Q = self.Q_linear(img) 
        K = torch.transpose(self.K_linear(text),1,0)#先进行一次转
        #下面开始计算啦
        alpha_qk = torch.matmul(Q, K)
        alpha_qk_norm= F.softmax(alpha_qk, dim = 1)* self._norm_fact

        out_img = torch.matmul(alpha_qk_norm,img)
        out_text = torch.matmul(alpha_qk_norm,text)
        return out_img, out_text
       
class Cross_Attention_Layer(nn.Module):
    #用来实现mask-attention layer
    def __init__(self, input_dim,dim_k,dim_v):
        super(Cross_Attention_Layer,self).__init__()
        
        # self.hidden_dim = hidden_dim
        self.Q_linear = nn.Linear(input_dim,dim_k, bias = False)
        self.K_linear = nn.Linear(input_dim,dim_k, bias = False)
        self.V_linear = nn.Linear(input_dim,dim_v, bias = False)
                 
    def forward(self, img, text):
        
        #计算生成QKV矩阵
        Q = self.Q_linear(img) 
        # K = torch.transpose(self.K_linear(text),1,0)#先进行一次转置
        K = self.K_linear(text).permute(0,2,1)
        V = self.V_linear(text)

        #下面开始计算啦
        alpha_qk = torch.matmul(Q, K)
        #下面开始softmax
        alpha = F.softmax(alpha_qk, dim = 1)
        #print('\nalpha is :', alpha)
        out = torch.matmul(alpha, V)
        feature = torch.add(out,text)
        # common_feature = torch.add(feature,img)
         
        # return feature_map
        return out,feature
    
class Self_Attention(nn.Module):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self,input_dim,dim_k,dim_v):
        super(Self_Attention,self).__init__()
        self.q = nn.Linear(input_dim,dim_k)
        self.k = nn.Linear(input_dim,dim_k)
        self.v = nn.Linear(input_dim,dim_v)
        self._norm_fact = 1 / sqrt(dim_k)  
    
    def forward(self,x):
        Q = self.q(x) # Q: batch_size * seq_len * dim_k
        K = self.k(x) # K: batch_size * seq_len * dim_k
        V = self.v(x) # V: batch_size * seq_len * dim_v
         
        atten = nn.Softmax(dim=-1)(torch.bmm(Q,K.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
        
        output = torch.bmm(atten,V) # Q * K.T() * V # batch_size * seq_len * dim_v
        feature = torch.add(output,x)
        
        return output,feature

class AV_Attention(nn.Module):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self,input_dim,dim_k,dim_v):
        super(AV_Attention,self).__init__()
        self.q = nn.Linear(input_dim,dim_k)
        self.k = nn.Linear(input_dim,dim_k)
        self.v = nn.Linear(input_dim,dim_v)
        self._norm_fact = 1 / sqrt(dim_k) 
    
    def forward(self, img, text):
        Q = self.q(img) # Q: batch_size * seq_len * dim_k
        K = self.k(text) # K: batch_size * seq_len * dim_k
        V = self.v(text) # V: batch_size * seq_len * dim_v
        
        atten = nn.Softmax(dim=-1)(torch.bmm(Q,K.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
        
        output = torch.bmm(atten,V) # Q * K.T() * V # batch_size * seq_len * dim_v
        feature = torch.add(output,text)
        
        return output,feature
    
class Interaction_Attention(nn.Module):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self,input_dim,dim_k):
        super(Interaction_Attention,self).__init__()
        self.q = nn.Linear(input_dim,dim_k) # 256 128
        self.k = nn.Linear(input_dim,dim_k) # 256 128

        self.j = nn.Linear(3,3) # 256 128

        self.joint_attention_map = nn.Bilinear(3, 3, 3, bias = False)
        self._norm_fact = 1 / sqrt(dim_k) 
    
    def forward(self, img, text):
        Q_i = self.q(img) # Q: batch_size * seq_len * dim_k
        K_i = self.k(img) # K: batch_size * seq_len * dim_k

        Q_t = self.q(text) # Q: batch_size * seq_len * dim_k
        K_t = self.k(text) # K: batch_size * seq_len * dim_k

        atten_it = nn.Softmax(dim=-1)(torch.bmm(Q_i,K_t.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
        atten_ii = nn.Softmax(dim=-1)(torch.bmm(Q_i,K_i.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
        atten_tt = nn.Softmax(dim=-1)(torch.bmm(Q_t,K_t.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len

        atten_i = self.joint_attention_map(self.j(atten_it),self.j(atten_ii))
        atten_t = self.joint_attention_map(self.j(atten_it),self.j(atten_tt))
        output_i = torch.bmm(atten_i,img)
        output_t = torch.bmm(atten_t,text)
        
        return output_i,output_t

class Integration_Attention(nn.Module):
    # input : batch_size * seq_len * input_dim
    def __init__(self,input_dim,dim_k):
        super(Integration_Attention,self).__init__()
        self.q = nn.Linear(input_dim,dim_k)
        self.k = nn.Linear(input_dim,dim_k)
        self.j = nn.Linear(3,3) # 256 128
        self.joint_attention_map = nn.Bilinear(3, 3, 3, bias = False)
        self._norm_fact = 1 / sqrt(dim_k) 
    
    def forward(self, img, text, comm):
        W_i = self.q(img)  # Q: batch_size * seq_len * dim_k
        W_t = self.q(text) # Q: batch_size * seq_len * dim_k
        W_c = self.k(comm) # K: batch_size * seq_len * dim_k
         
        atten_ic = nn.Softmax(dim=-1)(torch.bmm(W_i,W_c.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
        atten_tc = nn.Softmax(dim=-1)(torch.bmm(W_t,W_c.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len

        atten_c = self.joint_attention_map(self.j(atten_ic),self.j(atten_tc))
        output_i = torch.bmm(atten_c,img)
        output_t = torch.bmm(atten_c,text) 

        return output_i,output_t
    

