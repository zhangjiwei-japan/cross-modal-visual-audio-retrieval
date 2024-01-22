import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScale_Modal_Net(nn.Module):
    def __init__(self,num_features=1):
        super(MultiScale_Modal_Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 1,out_channels= 64,kernel_size = 32, stride = 8, padding = 12)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.BN = nn.BatchNorm1d(num_features) # 
        self.conv3_1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3_1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3_2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3_3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3_3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv5_1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool5_1 = nn.MaxPool1d(kernel_size=2 , stride=2)
        self.conv5_2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.pool5_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv5_3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.pool5_3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv7_1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.pool7_1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv7_2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.pool7_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv7_3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=7, stride=1, padding=3)
        self.pool7_3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.pool2 = nn.MaxPool1d(kernel_size=8, stride=1) # 
        # # self.pool2 = nn.AdaptiveAvgPool2d((1,1)) #
        # self.fc1 = nn.Linear(in_features, 256, bias=False)  
        # self.fc2 = nn.Linear(256, out_features, bias=False)  
        # self.fc3.apply(weights_init_classifier)
        
    def forward(self, x):
        batch_size = x.size(0)
        dimanal = x.size(1)
        x = x.view(batch_size,1,dimanal)
        x = self.BN(x)
        x = self.conv1(x)  ## x:Batch, 1, 1024
        x = self.pool1(x)
        # kernel_size 3
        x1 = self.conv3_1(x)
        x1 = self.pool3_1(x1)
        x1 = self.conv3_2(x1)
        x1 = self.pool3_2(x1)
        x1 = self.conv3_3(x1)
        x1 = self.pool3_3(x1)
        
        # kernel_size 5
        x2 = self.conv5_1(x)
        x2 = self.pool5_1(x2)
        x2 = self.conv5_2(x2)
        x2 = self.pool5_2(x2)
        x2 = self.conv5_3(x2)
        x2 = self.pool5_3(x2)
        
        # kernel_size 7
        x3 = self.conv7_1(x)
        x3 = self.pool7_1(x3)
        x3 = self.conv7_2(x3)
        x3 = self.pool7_2(x3)
        x3 = self.conv7_3(x3)
        x3 = self.pool7_3(x3)
        
        x1 = self.pool2(x1).permute(0,2,1)
        x2 = self.pool2(x2).permute(0,2,1)
        x3 = self.pool2(x3).permute(0,2,1)
        # print(x1.shape,x2.shape,x3.shape) 
        # # flatten
        # Batch, Channel, Length = x1.size()
        # x1 = x1.view(Batch, -1)
        # Batch, Channel, Length = x2.size()
        # x2 = x2.view(Batch, -1)
        # Batch, Channel, Length = x3.size()
        # x3 = x3.view(Batch, -1)
        # torch.Size([64, 256]) torch.Size([64, 256]) torch.Size([64, 256])
        # print(x1.shape,x2.shape,x3.shape) 
        feature = torch.cat((x1, x2, x3), dim=1)   
  
        # feature_1 = self.fc1(feature)  # 256
        # feature_2 = self.fc2(feature_1)  # 2
        return feature
    
