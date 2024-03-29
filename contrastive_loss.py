import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F

class IntraModalLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.2, hyper_gamma=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))			
        self.register_buffer("hyper_gamma", torch.tensor(hyper_gamma).to(device))			 
        self.register_buffer("negatives_mask", (~torch.eye(batch_size, batch_size, dtype=bool).to(device)).float())		
        
    def forward(self, emb):		# emb_i, emb_j 
        z = F.normalize(emb, dim=1)     # (bs, dim)  --->  (bs, dim)
        # representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        # print(similarity_matrix)
        sim_positive = torch.diag(similarity_matrix)
        # print(sim_positive)
        positives_exp = torch.exp(-sim_positive + self.temperature)
        negatives_exp = self.negatives_mask * torch.exp(similarity_matrix - self.temperature)
        # print(positives_exp,negatives_exp)
        positives_similarity = torch.log(1 + positives_exp)
        negatives_similarity = torch.log(1 + negatives_exp)
        positives_sum = torch.sum(positives_similarity) / (self.batch_size)
        negatives_sum = torch.sum(negatives_similarity) / (self.batch_size*(self.batch_size-1))
        loss = self.hyper_gamma*positives_sum + (1-self.hyper_gamma)*negatives_sum
        # print(positives_sum, negatives_sum)
        return loss

class TotalIntraModalLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', hyper_lambda=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.intra_modal_a = IntraModalLoss(self.batch_size)
        self.intra_modal_b = IntraModalLoss(self.batch_size)
        self.register_buffer("hyper_lambda", torch.tensor(hyper_lambda).to(device))			
        
    def forward(self, emb_i,emb_j):		# emb_i, emb_j 
        intra_i_loss = self.intra_modal_a(emb_i)
        intra_j_loss = self.intra_modal_b(emb_j)
        loss = self.hyper_lambda * intra_i_loss + (1-self.hyper_lambda) *intra_j_loss
        # print(intra_i_loss, intra_j_loss)
        return loss


class InterModalLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.2, hyper_gamma=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))			 
        self.register_buffer("hyper_gamma", torch.tensor(hyper_gamma).to(device))			
        self.register_buffer("negatives_mask", (~torch.eye(batch_size, batch_size, dtype=bool).to(device)).float())		
        
    def forward(self, emb_i, emb_j):		# emb_i, emb_j 
        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)

        # representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(z_j.unsqueeze(1), z_i.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        # print(similarity_matrix)
        sim_positive = torch.diag(similarity_matrix)
        # print(sim_positive)
        positives_exp = torch.exp(-sim_positive + self.temperature)
        negatives_exp = self.negatives_mask * torch.exp(similarity_matrix - self.temperature)
        # print(positives_exp,negatives_exp)
        positives_similarity = torch.log(1 + positives_exp)
        negatives_similarity = torch.log(1 + negatives_exp)
        positives_sum = torch.sum(positives_similarity) / (self.batch_size)
        negatives_sum = torch.sum(negatives_similarity) / (self.batch_size*(self.batch_size-1))
        loss = self.hyper_gamma*positives_sum + (1-self.hyper_gamma)*negatives_sum
        return loss

class TotalInterModalLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', hyper_lambda=0.5):
        super().__init__() 
        self.batch_size = batch_size
        self.inter_modal_a = InterModalLoss(self.batch_size)
        self.inter_modal_b = InterModalLoss(self.batch_size)
        self.register_buffer("hyper_lambda", torch.tensor(hyper_lambda).to(device))		
        
    def forward(self, emb_i,emb_j):		# emb_i, emb_j 
        intra_i_loss = self.inter_modal_b (emb_i,emb_j)
        intra_j_loss = self.inter_modal_b (emb_j,emb_i)
        loss = self.hyper_lambda * intra_i_loss + (1-self.hyper_lambda) *intra_j_loss
        # print(intra_i_loss, intra_j_loss)
        return loss

