# %%
import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torch.autograd import Variable
from resnet import resnet32
from collections import OrderedDict
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
# %%
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

train_set = datasets.CIFAR10(root='./data',download=True, train=True)


adv_imgs = np.load('/root/Adversarial-attacks-DNN-18786/codebase/adv_imgs.npz', allow_pickle = True)['arr_0']
adv_labels = np.load('/root/Adversarial-attacks-DNN-18786/codebase/adv_labels.npz', allow_pickle = True)['arr_0']

total_data = np.concatenate((train_set.data, adv_imgs), axis = 0)
total_labels = np.concatenate((train_set.targets, adv_labels), axis = 0)
# original_imgs = np.load('/root/Adversarial-attacks-DNN-18786/codebase/data/cifar-10-batches-py/data_batch_1')
# %%

class custom_loader():
    def __init__(self, dataset, label, cutout = False, i = 1, j = 18):
        self.dataset = []
        self.label = []
        self.cutout = cutout
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        if self.cutout:
            self.transform = transforms.Compose([
                                                transforms.ToTensor(),
                                                normalize, Cutout(i, j)
                                                ])
        else:
            self.transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            normalize,
                                            ])           
        for data in dataset:
            self.dataset.append(data)
        
        for lab in label:
            self.label.append(lab)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image = self.dataset[index]
        label = float(self.label[index])
        image = self.transform(image)
        return image, torch.as_tensor(label).long()
