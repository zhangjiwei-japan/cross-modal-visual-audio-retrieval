import torch
import time
import copy
import argparse
import torch.nn as nn
from loss_functions import *
from center_loss import CenterLoss
from evaluate import fx_calc_map_label
import numpy as np
import torch.optim as optim
# from models.img_text_models import Cross_Modal_Net
from models.basic_cross_models import CrossModal_NN
from models.basic_model import IDCM_NN
# from train_model import train_model
from datasets.load_data import get_loader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate, vegas 0.01 for ave 0.001')
parser.add_argument('--batch_size', default=100, type=int, help='train batch size')
parser.add_argument('--Center_lr', default=0.05, type=float, help='learning rate, vegas 0.5 ave 0.05')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--fig_attention_1', default= True, type=bool, help='fig_attention')
parser.add_argument('--fig_attention_2', default= True, type=bool, help='fig_attention')
parser.add_argument('--dataset', default='pascal', help='dataset name: vegas or ave]')
parser.add_argument('--l_id', default=1, type=float,help='loss paraerta')
parser.add_argument('--l_center', default=0.01, type=float,help='loss paraerta')
parser.add_argument('--l_dis', default=1, type=float,help='loss paraerta')
parser.add_argument("--load_ave_data", type=str, default= "dataset/AVE_feature_updated_squence.h5" , help="data_path")
parser.add_argument("--load_vegas_data", type=str, default= "dataset/vegas_feature.h5" , help="data_path")
args = parser.parse_args()

print('...Data loading is beginning...')
DATA_DIR = 'E:/Doctor-coder/cross-modal-dataset/' + args.dataset + '/'
data_loader, input_data_par = get_loader(DATA_DIR, args.batch_size)
net = CrossModal_NN(img_input_dim=input_data_par['img_dim'],text_input_dim=input_data_par['text_dim'],output_class_dim=input_data_par['num_class']).to(device)
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
if args.optim == 'sgd':
    ignored_params =  list(map(id, net.visual_layer.parameters())) \
                    + list(map(id, net.text_layer.parameters())) \
                    + list(map(id, net.linearLayer.parameters())) \
                    + list(map(id, net.classifier_t.parameters()))\
                    + list(map(id, net.classifier_v.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.visual_layer.parameters(), 'lr': args.lr},
        {'params': net.text_layer.parameters(), 'lr': args.lr},
        {'params': net.linearLayer.parameters(), 'lr': args.lr},
        {'params': net.classifier_t.parameters(), 'lr': args.lr},
        {'params': net.classifier_v.parameters(), 'lr': args.lr}
        ],
        weight_decay=5e-4, momentum=0.9, nesterov=True)
    optmizercenter = optim.SGD(center_loss.parameters(), lr=args.Center_lr)  # 0.05 设置weight_decay=5e-3，即设置较大的L2正则来降低过拟合。

def train_model(optimizer, alpha, beta, num_epochs=500):
    best_acc = 0.0
    since = time.time()
    test_img_acc_history = []
    test_txt_acc_history = []
    epoch_loss_history =[]
    center_loss = CenterLoss(20,64).to(device)
    nllloss = nn.CrossEntropyLoss().to(device)
    best_model_wts = copy.deepcopy(net.state_dict())
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 20)
        current_lr = adjust_learning_rate(optimizer, epoch,num_epochs)
        train_loss,train_center,train_nll,train_dis = 0,0,0,0
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                # Set model to training mode
                net.train()
            else:
                # Set model to evaluate mode
                net.eval()

            running_loss = 0.0
            running_corrects_img = 0
            running_corrects_txt = 0
            # Iterate over data.
            for imgs, txts, labels in data_loader[phase]:
                if torch.sum(imgs != imgs)>1 or torch.sum(txts != txts)>1:
                    print("Data contains Nan.")

                # zero the parameter gradients
                optimizer.zero_grad()
                optmizercenter.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if torch.cuda.is_available():
                        
                        imgs = imgs.to(device)
                        txts = txts.to(device)
                        labels = labels.to(device)
                        # print(imgs.shape,txts.shape)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    optmizercenter.zero_grad()

                    # Forward
                    view1_feature, view2_feature, view1_predict, view2_predict = net(imgs, txts)
                    # term1 = ((view1_predict-labels.float())**2).sum(1).sqrt().mean() + ((view2_predict-labels.float())**2).sum(1).sqrt().mean()
                    labels_1 = torch.argmax(labels,dim=1).long()
                    labels_2 = torch.argmax(labels,dim=1).long()
                    loss_id = nllloss(view1_predict,labels_1) + nllloss(view2_predict,labels_2)
                    loss_cent = center_loss(view1_feature,labels_1) + center_loss(view2_feature,labels_2)
                    loss_dis = ((view1_feature - view2_feature)**2).sum(1).sqrt().mean()
                    loss = loss_id + alpha * loss_cent + beta * loss_dis

                    train_loss += loss.item()
                    train_center += loss_cent.item()
                    train_nll += loss_id.item()
                    train_dis += loss_dis.item()

                    # loss = calc_loss(view1_feature, view2_feature, view1_predict,
                    #                  view2_predict, labels, labels, alpha, beta)

                    img_preds = view1_predict
                    txt_preds = view2_predict

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects_img += torch.sum(torch.argmax(img_preds, dim=1) == torch.argmax(labels, dim=1))
                running_corrects_txt += torch.sum(torch.argmax(txt_preds, dim=1) == torch.argmax(labels, dim=1))

            print("Epoch:{}/{} Loss:{:.2f} Nll:{:.2f} Dis:{:.2f} Center:{:.2f} Lr:{:.6f}/{:.6f}".format(epoch,num_epochs, train_loss,
                 train_nll,train_dis,train_center,current_lr,optmizercenter.param_groups[0]['lr']))
            epoch_loss = running_loss / len(data_loader[phase].dataset)
            # epoch_img_acc = running_corrects_img.double() / len(data_loaders[phase].dataset)
            # epoch_txt_acc = running_corrects_txt.double() / len(data_loaders[phase].dataset)
            t_imgs, t_txts, t_labels = [], [], []
            with torch.no_grad():
                for imgs, txts, labels in data_loader['test']:
                    if torch.cuda.is_available():
                            imgs = imgs.to(device)
                            txts = txts.to(device)
                            labels = labels.to(device)
                    t_view1_feature, t_view2_feature, _, _ = net(imgs, txts)
                    t_imgs.append(t_view1_feature.cpu().numpy())
                    t_txts.append(t_view2_feature.cpu().numpy())
                    t_labels.append(labels.cpu().numpy())
            t_imgs = np.concatenate(t_imgs)
            t_txts = np.concatenate(t_txts)
            t_labels = np.concatenate(t_labels).argmax(1)
            img2text = fx_calc_map_label(t_imgs, t_txts, t_labels)
            txt2img = fx_calc_map_label(t_txts, t_imgs, t_labels)

            print('{} Loss: {:.4f} Img2Txt: {:.4f}  Txt2Img: {:.4f}'.format(phase, epoch_loss, img2text, txt2img))

            # deep copy the model
            if phase == 'test' and (img2text + txt2img) / 2. > best_acc:
                best_acc = (img2text + txt2img) / 2.
                best_model_wts = copy.deepcopy(net.state_dict())
                # if best_acc >0.69:
                #     torch.save(net.state_dict(), 'save_models/text_image_{:.4f}_best.pth'.format(best_acc))
            if phase == 'test':
                test_img_acc_history.append(img2text)
                test_txt_acc_history.append(txt2img)
                epoch_loss_history.append(epoch_loss)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best average ACC: {:4f}'.format(best_acc))

    # load best model weights
    net.load_state_dict(best_model_wts)
    return net,test_img_acc_history, test_txt_acc_history, epoch_loss_history

def test_model():
    print('...Evaluation on testing data...')
    view1_feature, view2_feature, view1_predict, view2_predict = net(torch.tensor(input_data_par['img_test']).to(device), torch.tensor(input_data_par['text_test']).to(device))
    label = torch.argmax(torch.tensor(input_data_par['label_test']), dim=1)
    view1_feature = view1_feature.detach().cpu().numpy()
    view2_feature = view2_feature.detach().cpu().numpy()
    view1_predict = view1_predict.detach().cpu().numpy()
    view2_predict = view2_predict.detach().cpu().numpy()
    img_to_txt = fx_calc_map_label(view1_feature, view2_feature, label)
    print('...Image to Text MAP = {}'.format(img_to_txt))

    txt_to_img = fx_calc_map_label(view2_feature, view1_feature, label)
    print('...Text to Image MAP = {}'.format(txt_to_img))
    
    MAP = (img_to_txt + txt_to_img) / 2.
    print('...Average MAP = {}'.format(MAP))

    return img_to_txt,txt_to_img,MAP