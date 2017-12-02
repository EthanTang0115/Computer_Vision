import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms
import torchvision.utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable

from PIL import Image
from PIL import ImageOps
import os
import random

from skimage import transform as tf
import math

########UTILITY FUNCTION####
def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.title('Loss History')
    plt.savefig('1a_BCE_LossHistory.png')

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
test_batch_size = 1
#######PREPARE TRAIN AND TEST SETS##########
siamese_dataset_train = SiameseNetworkDataset('lfw','train.txt', trans)
siamese_dataset_test = SiameseNetworkDataset('lfw','test.txt', trans)
train_dataloader = DataLoader(siamese_dataset_train, shuffle=True, num_workers=8, batch_size=train_batch_size)
test_dataloader = DataLoader(siamese_dataset_test, shuffle=True, num_workers=8, batch_size=test_batch_size)
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
        
        return output1, output2
##########################################

##########Contrastive Loss################
class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((label)*torch.pow(euclidean_distance, 2)+
                                      (1-label)*torch.pow(torch.clamp(slef.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive 
##########################################

##########Training Settings###############
#data feedaing settings
train_number_epochs = 30 # use small size for CPU debug
##net work settings
net = SiameseNetwork().cuda()
loss_function = ContrastiveLoss()
optimizer = optim.Adam(net.parameters())
##########################################

#########Training History Tracker#########
counter = []
loss_history = []
iteration_number = 0
##########################################

############Training Process##############
for epoch in range(train_number_epochs):
    for i, data in enumerate(train_dataloader):
        img1, img2, label = data
        img1, img2, label = Variable(img1, requires_grad=True).cuda(), Variable(img2, requires_grad=True).cuda(), Variable(label).cuda()
        label = label.type(torch.FloatTensor)
        output_1, output_2 = net.forward(img1, img2)
        output_1 = label_pred_1.type(torch.FloatTensor)
        output_2 = label_pred_2.type(torch.FloatTensor)
        optimizer.zero_grad()
        loss = loss_function(label_pred_1, label_pred_2, label)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print "Epoch number: %d, Current loss: %f" % (epoch, loss.data[0])
            iteration_number += 100
            counter.append(iteration_number)
            loss_history.append(loss.data[0])
            
show_plot(counter, loss_history)
torch.save(net.state_dict(), "./1a_BCE.pth")
#############################################

############Testing Phase####################
#total = 0
#correct = 0
dissimilarity = []
for data in train_dataloader:
    img1, img2, label = data
    img1, img2 = Variable(img1).cuda(), Variable(img2).cuda()
    output_1, output_2 = net.forward(img1, img2)
    euclidean_distance = torch.mean(F.pairwise_distance(output_1, output_2))
    dissimilarity.append(euclidean_distance)
    #correct += np.sum((np.array(label.numpy().astype(float))).ravel() == np.array(label_pred))
    #total += test_batch_size

plt.hist(dissimilarity, 10)
plt.savefig("dissimilarity_hist.png")
#print "Dissimilarity on training data is",(dissimilarity/ float(total))

#total = 0
#correct = 0
#for data in test_dataloader:
    #img1, img2, label = data
    #img1, img2 = Variable(img1).cuda(), Variable(img2).cuda()
    #output_1, output_2 = net.forward(img1, img2)
    #euclidean_distance = F.pairwise_distance(output_1, output_2)
    #label_pred = label_pred.data.numpy()
    #label_pred[label_pred > 0.5] = 1
    #label_pred[label_pred <= 0.5] = 0
    #label_pred = label_pred.flatten()
    #correct += np.sum((np.array(label.numpy().astype(float))).ravel() == np.array(label_pred))
    #total += train_batch_size

#print "Accuracy on testing data is", (correct/ float(total))
#############################################################

