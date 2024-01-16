import torch
import time
import copy
import argparse
import torch.nn as nn
from loss_functions import *
from contrastive_loss import *
from datasets.load_data_vegas_ave import *
from evaluate import fx_calc_map_label
import numpy as np
import torch.optim as optim
# from models.img_text_models import Cross_Modal_Net
from cross_model_net_base import CrossModal_NN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate, vegas 0.01 for ave 0.001')
parser.add_argument('--batch_size', default=128, type=int, help='train batch size')
parser.add_argument('--dataset', default='vegas', help='dataset name: vegas or ave')
parser.add_argument('--optim', default='adam', type=str, help='optimizer')
parser.add_argument('--l_id', default=1, type=float,help='loss parameter')
parser.add_argument('--l_corr', default=0.01, type=float,help='loss parameter')
parser.add_argument("--load_vegas_data", type=str, default= "vegas_feature.h5" , help="data_path")
args = parser.parse_args()

print('...Data loading is beginning...')
# load dataset path
base_dir = "./datasets/"
load_path =  base_dir + args.load_vegas_data

def adjust_learning_rate(optimizer, epoch,num_epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 20: 
        lr = args.lr * (epoch + 1) / 20
    elif epoch >= 20 and epoch < 0.25*num_epoch:
        lr = args.lr
    elif epoch >=  0.25*num_epoch and epoch < 0.50*num_epoch:
        lr = args.lr * 0.1
    elif epoch >= 0.50*num_epoch and epoch < 0.75*num_epoch:
        lr = args.lr * 0.01
    elif epoch >= 0.75*num_epoch:
        lr = args.lr * 0.001

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr

def train_model(Lr, beta, batch_size, test_size, num_epochs):
    print("....train the model on vegas dataset....")
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')
    # batch_size = 64
    best_acc = 0.0
    best_audio_2_img = 0.0
    best_img_2_audio = 0.0
    visual_feat_dim = 1024
    audio_fea_dim = 128
    mid_dim = 128
    class_dim = 10
    net = CrossModal_NN(img_input_dim=visual_feat_dim, img_output_dim=visual_feat_dim,
                        audio_input_dim=audio_fea_dim, audio_output_dim=visual_feat_dim, minus_one_dim= mid_dim, output_dim=class_dim).to(device)
    
    nllloss = nn.CrossEntropyLoss().to(device)
    intramodal_loss = TotalIntraModalLoss(batch_size=batch_size)
    intermodal_loss = TotalInterModalLoss(batch_size=batch_size)
    best_model_wts = copy.deepcopy(net.state_dict())
    params_to_update = list(net.parameters())
    betas = (0.5, 0.999)
    optimizer = optim.Adam(params_to_update, Lr, betas=betas)
    data_loader_visual,data_loader_audio = load_dataset_train(load_path,batch_size)
    for epoch in range(num_epochs):
        net.train()
        train_loss,train_inter,train_nll,train_intra,train_dis,train_center,train_contra = 0,0,0,0,0,0,0
        for i, data in enumerate(zip(data_loader_visual, data_loader_audio)):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                inputs_visual = data[0][0].to(device)
                labels_visual = data[0][1].to(device)
                labels_visual = labels_visual.squeeze(1)
                inputs_audio = data[1][0].to(device)
                labels_audio = data[1][1].to(device)
                labels_audio  = labels_audio.squeeze(1)
            view1_feature, view2_feature, view1_predict, view2_predict,logit_scale = net(inputs_visual,inputs_audio)
            loss_id = nllloss(view1_predict,labels_visual.long()) + nllloss(view2_predict,labels_audio.long())
            loss_intra = intramodal_loss(view1_feature,view2_feature)
            loss_inter = intermodal_loss(view1_feature,view2_feature)
            loss = loss_id + beta * (loss_intra+ loss_inter)

            train_loss += loss.item()
            train_inter += loss_inter.item()
            train_intra += loss_intra.item()
            train_nll += loss_id.item()

            loss.backward()
            optimizer.step()
        print("Epoch:{}/{} Loss:{:.2f} Nll:{:.2f} Inter:{:.2f} Intra:{:.2f} Lr:{:.6f}".format(epoch,num_epochs, train_loss,
                 train_nll,train_inter,train_intra,optimizer.param_groups[0]['lr']))
        
        if epoch > 0 and epoch%5==0:
             img_to_txt,txt_to_img,MAP = eval_model(net, test_size)
             if MAP > 0.89:
            #  if epoch > 50 and MAP > best_acc:
                best_acc = MAP
                best_audio_2_img = txt_to_img
                best_img_2_audio = img_to_txt
                print("Best Acc: {}".format(best_acc))
                torch.save(net.state_dict(), 'save_models/vegas-models/audio_image_best_vegas_{}_revise_alignment.pth'.format(best_acc))
    return round(best_img_2_audio,4),round(best_audio_2_img,4),round(best_acc,4)

def eval_model(model, test_size):
    local_time =  time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    model.eval()
    t_imgs, t_txts, t_labels,p_img,p_txt = [], [], [], [], []
    visual_test,audio_test = load_dataset_test(load_path,test_size)
    with torch.no_grad():
        for i, data in enumerate(zip(visual_test, audio_test)):
            if torch.cuda.is_available():
                    inputs_visual = data[0][0].to(device)
                    labels_visual = data[0][1].to(device)
                    labels_visual = labels_visual.squeeze(1)
                    inputs_audio = data[1][0].to(device)
                    labels_audio = data[1][1].to(device)
            t_view1_feature, t_view2_feature, predict_view1, predict_view2,logit_scale= model(inputs_visual,inputs_audio)
            labels_view1 = torch.argmax(predict_view1,dim=1).long()
            labels_view2 = torch.argmax(predict_view2,dim=1).long()

            t_imgs.append(t_view1_feature.cpu().detach().numpy())
            t_txts.append(t_view2_feature.cpu().detach().numpy())
            t_labels.append(labels_visual.cpu().detach().numpy())
            p_img.append(labels_view1.cpu().detach().numpy())
            p_txt.append(labels_view2.cpu().detach().numpy())

    t_imgs = np.concatenate(t_imgs)
    t_txts = np.concatenate(t_txts)
    t_labels = np.concatenate(t_labels)
    p_img =  np.concatenate(p_img)
    p_txt =  np.concatenate(p_txt)
    
    img2audio = fx_calc_map_label(t_imgs, t_txts, t_labels)
    print('...Image to audio MAP = {}'.format(img2audio))
    txt2img = fx_calc_map_label(t_txts, t_imgs, t_labels)
    print('...audio to Image MAP = {}'.format(txt2img))
    Acc = (img2audio + txt2img) / 2.
    print('...Average MAP = {}'.format(Acc))

    return round(img2audio,4),round(txt2img,4),round(Acc,4)
