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

# og_examples_format = np.load('/root/og_images10k.npz', allow_pickle = True)['arr_0']
# inpaint_examples_format = np.load('/root/test_inpainted10k.npz', allow_pickle = True)['arr_0']
# inpaint_labels = np.load('/root/label_inpainted10k.npz', allow_pickle = True)['arr_0']
adv_examples_cutout_format = np.load('/root/Adversarial-attacks-DNN-18786/codebase/adv_images_fgsm.npz', allow_pickle  = True)['arr_0']
adv_labels_cutout_format = np.load('/root/Adversarial-attacks-DNN-18786/codebase/adv_labels_fgsm.npz', allow_pickle = True)['arr_0']


# og_examples_format = og_
# inpaint_examples_format = inpaint_examples
# inpaint_labels_format = inpaint_labels

# %%
# custom_dataset_inpt = custom_loader(inpaint_examples_format, inpaint_labels_format)
# custom_dataset_og = custom_loader(og_examples_format, inpaint_labels_format)
# custom_dataset_cutout = custom_loader(og_examples_format, inpaint_labels_format, cutout=True)
# test_loader_inpt = torch.utils.data.DataLoader(custom_dataset_inpt,
#     batch_size=128, shuffle=False,
#     num_workers=4, pin_memory=True)

# test_loader_og = torch.utils.data.DataLoader(custom_dataset_og,
#     batch_size=128, shuffle=False,
#     num_workers=4, pin_memory=True)

# test_loader_cutout = torch.utils.data.DataLoader(custom_dataset_cutout, batch_size = 128, 
#                                                 shuffle = False, num_workers = 4, pin_memory = True)

# %%

samples = []
def validation(model,valid_dataloader):
  model.eval()
  top1_accuracy = 0
  total = 0
  for batch_num,(feats,label) in enumerate(valid_dataloader):
    samples.append(feats)
    feats = feats.cuda().float()
    label = label.cuda()
    valid_output=model(feats)
    predictions = F.softmax(valid_output, dim=1)
    _, top1_pred_labels = torch.max(predictions,1)
    # if top1_pred_labels != label:
    #     print(batch_num, top1_pred_labels, label)
    top1_accuracy += torch.sum(torch.eq(top1_pred_labels, label)).item()
    total += len(label)
#   model.train()
  return top1_accuracy/total

# accs = []
# for i in range(1, 20):
#     for j in range(1, 20):
#         custom_dataset_adv_cutout = custom_loader(adv_examples_cutout_format, adv_labels_cutout_format, cutout = True, i = i, j = j)

#         test_loader_adv_cutouts = torch.utils.data.DataLoader(custom_dataset_adv_cutout, batch_size = 128, 
#                                                 shuffle = False, num_workers = 4, pin_memory = True)

custom_dataset_adv_cutout = custom_loader(adv_examples_cutout_format, adv_labels_cutout_format, cutout = True, i = 18, j = 2)
test_loader_adv_cutouts = torch.utils.data.DataLoader(custom_dataset_adv_cutout, batch_size = 128, shuffle = False, num_workers = 4, pin_memory = True)


# print("inpainted:", validation(model, test_loader_inpt))
# print("original:", validation(model, test_loader_og))
# print("cutout:", validation(model, test_loader_cutout))
result = validation(model, test_loader_adv_cutouts)
# print("cutout adv:", i,j,result)
# accs.append((i,j,result))
print(result)

# %% plot acc

y1 = np.array([0.3095, 0.3933, 0.3933, 0.3258, 0.3258, 0.1768, 0.1768, 0.1301, 0.1301, 0.1247,
            0.1247, 0.1192, 0.1192, 0.1118, 0.1118, 0.1112, 0.1112, 0.1081, 0.1081])
y2 = np.array([0.3095, 0.3552, 0.3552, 0.359, 0.359, 0.2837, 0.2837, 0.2254, 0.2254, 0.1857, 0.1857,
               0.1663, 0.1663, 0.1537, 0.1537, 0.1402, 0.1402, 0.1268, 0.1268])
y3 = [
 0.3095,
 0.3429,
 0.3429,
 0.354,
 0.354,
 0.3157,
 0.3157,
 0.273,
 0.273,
 0.2387,
 0.2387,
 0.2177,
 0.2177,
 0.1965,
 0.1965,
 0.1758,
 0.1758,
 0.1541,
 0.1541
]
x = np.arange(1, 20, 1)

plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)
plt.show()