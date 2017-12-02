import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms
import torchvision.utils

import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable

from PIL import Image
from PIL import ImageOps
import os
import random

########UTILITY FUNCTION####
def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.title('Loss History')
    plt.save('LossHistory.png')

########LOAD DATA#########
class SiameseNetworkDataset(Dataset):
    
    def __init__(self, data_folder_path, img_lst_path, transform=None):
        self.folder = data_folder_path
        self.transform = transform
        self.img_lst = self.read_split_file(img_lst_path)
        
    def __getitem__(self,index):
        img0_path = os.path.join(self.folder, self.img_lst[index][0])
        img1_path = os.path.join(self.folder, self.img_lst[index][1])
        
        label = map(float, self.img_lst[index][2])
        label = torch.from_numpy(np.array(label)).float()
        
        img0 = Image.open(img0_path)
        img1 = Image.open(img1_path)
        img0 = img0.convert('RGB')
        img1 = img1.convert('RGB')
        
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1, label
    
    def __len__(self):
        return len(self.img_lst)
    
    def read_split_file(self, dir):
        res = []
        with open(dir) as f:
            for line in f:
                content = line.split()
                res.append(content)
        return res

trans = transforms.Compose([
    transforms.Scale(128),
    transforms.ToTensor(),
])

############################
train_batch_size = 16
#######PREPARE TRAIN AND TEST SETS##########
siamese_dataset_train = SiameseNetworkDataset('lfw','train.txt', trans)
siamese_dataset_test = SiameseNetworkDataset('lfw','test.txt',transform=transforms.Compose([transforms.ToTensor()]))
train_dataloader = DataLoader(siamese_dataset_train, shuffle=True, num_workers=8, batch_size=train_batch_size)
test_dataloader = DataLoader(siamese_dataset_test, shuffle=True, num_workers=8, batch_size=train_batch_size)
############################################

#####NETWORK STRUCTURE######
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
             
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(131072, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x1, x2):
        output1 = self.cnn(x1)
        output1 = output1.view(output1.size()[0], -1)
        output1 = self.fc1(output1)
        
        output2 = self.cnn(x2)
        output2 = output2.view(output2.size()[0], -1)
        output2 = self.fc1(output2)
        
        output = torch.cat((output1, output2), 1)
        output = self.fc2(output)
        
        return output
##########################################

##########Training Settings###############
#data feedaing settings
train_number_epochs = 30
##net work settings
net = SiameseNetwork()
loss_function = nn.BCELoss()
optimizer = optim.Adam(net.parameters())
##########################################

#########Training History Tracker#########
counter = []
loss_history = []
iteration_number = []
##########################################

############Training Process##############
for j in range(0,train_number_epochs):
    for i, data in enumerate(train_dataloader):
        img1, img2, label = data
        img1, img2, label = Variable(img1, requires_grad=True), Variable(img2, requires_grad=True), Variable(label)
        label = label.type(torch.FloatTensor)
        label_pred = net.forward(img1, img2)
        optimizer.zero_grad()
        loss = loss_function(label_pred, label)
        loss.backward()
        optimizer.step()
        if i % 500 == 0:
            print "Epoch number: %d, Current loss: %f" % (j, loss.data[0])
            iteration_number.append(i)
            counter.append(iteration_number)
            loss_history.append(loss.data[0])
show_plot(counter, loss_history)

#save the trained model
torch.save(net.state_dict(), "./tangyichuan18/HW3/1a_BCE.pth")
###########################################