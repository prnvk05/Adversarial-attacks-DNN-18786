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
# from inpt_loader import val_set
import numpy as np
import matplotlib.pyplot as plt
from cutout import Cutout
# %% LOAD SAVED MODEL

model = resnet32().cuda()
model.eval()
checkpoint = torch.load('/root/Adversarial-attacks-DNN-18786/saved_model/resnet32-d509ac18.th')

state_dict = checkpoint['state_dict']

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
# %% DATA LOADER
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

def validation(model, valid_dataloader):
#   import pdb; pdb.set_trace()
  model.eval()
  top1_accuracy = 0
  total = 0
  for batch_num,(feats,label) in enumerate(valid_dataloader):
    # samples.append(feats)
    feats = feats.cuda().float()
    label = label.cuda()
    valid_output=model(feats)
    predictions = F.softmax(valid_output, dim=1)
    _, top1_pred_labels = torch.max(predictions,1)
    top1_accuracy += torch.sum(torch.eq(top1_pred_labels, label)).item()
    total += len(label)
  model.train()
  return top1_accuracy/total



data_loader_params = {
            'cutout': False,
            'i': 1,
            'j': 18, 
            'batch_size': 128,
            'shuffle': False,
            'num_workers':4
            }


def evaluate(model,  label_path, image_path, loader_params):
    # import pdb; pdb.set_trace()
    label = np.load(label_path, allow_pickle = True)['arr_0']
    image = np.load(image_path, allow_pickle = True)['arr_0']
    custom_dataset = custom_loader(image, label, loader_params['cutout'], loader_params['i'], loader_params['j'])
    custom_dataset_loader = torch.utils.data.DataLoader(custom_dataset,
                                                            batch_size = loader_params['batch_size'], shuffle = loader_params['shuffle'],
                                                            num_workers = loader_params['num_workers'], pin_memory=True)
    result = validation(model, custom_dataset_loader)
    return result

# %%
label_path = '/root/label_newadv_spots_inpainted10k (1).npz'
image_path =  '/root/18spots_test_denoisedadv_spots_inpainted (1).npz'
result = evaluate(model,label_path,image_path,data_loader_params)
