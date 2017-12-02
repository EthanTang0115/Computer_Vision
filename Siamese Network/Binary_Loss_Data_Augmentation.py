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
        ########DATA AUGMENTATION#############
        aug_prob_0 = random.uniform(0, 1)
        if aug_prob_0 > 0.3:
            mirror_filp_prob_0 = random.uniform(0, 1)
            rand_rot_0 = round(random.uniform(-1, 1)*30)
            rand_trans_0_x = round(random.uniform(-1, 1)*10)
            rand_trans_0_y = round(random.uniform(-1, 1)*10)
            rand_scale_0 = random.uniform(0.7, 1.3)

            if mirror_filp_prob_0 > 0.5:
                img0 = ImageOps.mirror(img0)

            tform = tf.SimilarityTransform(scale=rand_scale_0, rotation=rand_rot_0*math.pi/180, 
                                   translation=(rand_trans_0_x, rand_trans_0_y))
            img0 = tf.warp(img0, tform)*255
            img0 = np.array(img0.astype(np.uint8))
            img0 = Image.fromarray(img0)

        aug_prob_1 = random.uniform(0, 1)
        if aug_prob_1 > 0.3:
            mirror_filp_prob_1 = random.uniform(0, 1)
            rand_rot_1 = round(random.uniform(-1, 1)*30)
            rand_trans_1_x = round(random.uniform(-1, 1)*10)
            rand_trans_1_y = round(random.uniform(-1, 1)*10)
            rand_scale_1 = random.uniform(0.7, 1.3)

            if mirror_filp_prob_1 > 0.5:
                img1 = ImageOps.mirror(img1)

            tform = tf.SimilarityTransform(scale=rand_scale_1, rotation=rand_rot_1*math.pi/180, 
                                   translation=(rand_trans_1_x, rand_trans_1_y))
            img1 = tf.warp(img1, tform)*255
            img1 = np.array(img1.astype(np.uint8))
            img1 = Image.fromarray(img1)
        #########################################
        
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
test_batch_size = 16
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
        
        output = torch.cat((output1, output2), 1)
        output = self.fc2(output)
        
        return output
##########################################

##########Training Settings###############
#data feedaing settings
train_number_epochs = 30 # use small size for CPU debug
##net work settings
net = SiameseNetwork().cuda()
loss_function = nn.BCELoss()
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
        label_pred = net.forward(img1, img2)
        label_pred = label_pred.type(torch.FloatTensor)
        optimizer.zero_grad()
        loss = loss_function(label_pred, label)
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
total = 0
correct = 0
for data in train_dataloader:
    img1, img2, label = data
    img1, img2 = Variable(img1).cuda(), Variable(img2).cuda()
    label_pred = net.forward(img1, img2)
    label_pred = label_pred.cpu()
    label_pred = label_pred.data.numpy()
    label_pred[label_pred > 0.5]  = 1
    label_pred[label_pred <= 0.5] = 0
    label_pred = label_pred.flatten()
    correct += np.sum((np.array(label.numpy().astype(float))).ravel() == np.array(label_pred))
    total += test_batch_size

print "Accuracy on training data is",(correct/ float(total))

total = 0
correct = 0
for data in test_dataloader:
    img1, img2, label = data
    img1, img2 = Variable(img1).cuda(), Variable(img2).cuda()
    label_pred = net.forward(img1, img2)
    label_pred = label_pred.cpu()
    label_pred = label_pred.data.numpy()
    label_pred[label_pred > 0.5] = 1
    label_pred[label_pred <= 0.5] = 0
    label_pred = label_pred.flatten()
    correct += np.sum((np.array(label.numpy().astype(float))).ravel() == np.array(label_pred))
    total += train_batch_size

print "Accuracy on testing data is", (correct/ float(total))
#############################################################

