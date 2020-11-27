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
from inpt_test_resnet import resnet32
from collections import OrderedDict
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from inpt_loader import val_set
import numpy as np
import matplotlib.pyplot as plt
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
    def __init__(self, dataset, label):
        self.dataset = []
        self.label = []
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

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

inpaint_examples = np.load('/root/Adversarial-attacks-DNN-18786/Inpainting_experiment/test_data/test_inpainted.npy', allow_pickle = True)
inpaint_labels = np.load('/root/Adversarial-attacks-DNN-18786/Inpainting_experiment/test_data/label_inpainted.npy', allow_pickle = True)

inpaint_examples_format = inpaint_examples.squeeze(1)
inpaint_labels_format = inpaint_labels.squeeze(1)


custom_dataset = custom_loader(inpaint_examples_format, inpaint_labels_format)


test_loader = torch.utils.data.DataLoader(custom_dataset,
    batch_size=1, shuffle=False,
    num_workers=4, pin_memory=True)
# %%

def validation(model,valid_dataloader):
  model.eval()
  top1_accuracy = 0
  total = 0
  for batch_num,(feats,label) in enumerate(valid_dataloader):
    feats = feats.cuda()
    label = label.cuda()
    valid_output=model(feats)
    predictions = F.softmax(valid_output, dim=1)
    _, top1_pred_labels = torch.max(predictions,1)
    if top1_pred_labels != label:
        print(batch_num, top1_pred_labels, label)
    top1_accuracy += torch.sum(torch.eq(top1_pred_labels, label)).item()
    total += len(label)
#   model.train()
  return top1_accuracy/total


print(validation(model, test_loader))