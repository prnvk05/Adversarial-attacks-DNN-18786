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

### ORIGINAL & INPAINTED WITHOUT NOISE
og_examples_format = np.load('/root/og_new_spots_images10k.npz', allow_pickle = True)['arr_0']
og_labels_format = np.load('/root/label_new_spots_inpainted10k.npz', allow_pickle = True)['arr_0']

inpaint_examples_format = np.load('/root/test_new_spots_inpainted10k.npz', allow_pickle = True)['arr_0']
inpaint_labels = np.load('/root/label_new_spots_inpainted10k.npz', allow_pickle = True)['arr_0']

### ORIGINAL & INPAINTED WITH ADVERSARIAL NOISE
og_adv_examples_format = np.load('/root/og_newadv_spots_images10k.npz', allow_pickle=True)['arr_0']
og_adv_labels_format = np.load('/root/label_newadv_spots_inpainted10k.npz', allow_pickle=True)['arr_0']

adv_inpainted_examples_format = np.load('/root/test_newadv_spots_inpainted10k.npz', allow_pickle = True)['arr_0']
adv_inpainted_labels_format = np.load('/root/ly10k.npz', allow_pickle = True)['arr_0']



# %%
### WITHOUT ADV NOISE
# custom_dataset_inpt = custom_loader(inpaint_examples_format, inpaint_labels)
# custom_dataset_og_cutout = custom_loader(og_examples_format, og_labels_format, cutout = True, i = 18, j = 2)
# custom_dataset_og = custom_loader(og_examples_format, og_labels_format)

# test_loader_inpt_nonoise = torch.utils.data.DataLoader(custom_dataset_inpt,
#     batch_size=128, shuffle=False,
#     num_workers=4, pin_memory=True)
# test_loader_og_nonoise = torch.utils.data.DataLoader(custom_dataset_og,
#     batch_size=128, shuffle=False,
#     num_workers=4, pin_memory=True)
# test_loader_cutout_nonoise = torch.utils.data.DataLoader(custom_dataset_og_cutout,
#     batch_size=128, shuffle=False,
#     num_workers=4, pin_memory=True)


###  WITH ADV NOISE
# custom_dataset_adv_inpt = custom_loader(adv_inpainted_examples_format, adv_inpainted_labels_format)
# custom_dataset_adv_og = custom_loader(og_adv_examples_format, og_adv_labels_format)
# custom_dataset_adv_cutout = custom_loader(og_adv_examples_format, adv_inpainted_labels_format, cutout = True, i = 18, j = 2)

# test_loader_adv_cutout_noise = torch.utils.data.DataLoader(custom_dataset_adv_cutout,
#     batch_size=128, shuffle=False,
#     num_workers=4, pin_memory=True)
# test_loader_adv_inpt_noise = torch.utils.data.DataLoader(custom_dataset_adv_inpt,
#     batch_size=128, shuffle=False,
#     num_workers=4, pin_memory=True)
# test_loader_adv_og_noise = torch.utils.data.DataLoader(custom_dataset_adv_og,
#     batch_size=128, shuffle=False,
#     num_workers=4, pin_memory=True)




# custom_dataset_adv_cutout = custom_loader(adv_examples_og, adv_examples_labels, cutout = True, i = 18, j = 2)
# custom_dataset_adv_inpainted = custom_loader(adv_examples_og, adv_examples_labels)

### LOADER FOR ORIGINAL INPAINTED W


# test_loader_cutout = torch.utils.data.DataLoader(custom_dataset_cutout, batch_size = 128, 
#                                                 shuffle = False, num_workers = 4, pin_memory = True)


# test_loader_adv_cutout = torch.utils.data.DataLoader(custom_dataset_adv_cutout, batch_size = 128, shuffle = False, num_workers = 4, pin_memory = True)

# test_loader_adv_inpainted = torch.utils.data.DataLoader(custom_dataset_adv_inpainted, batch_size = 128, shuffle = False, 
#                                                         num_workers = 4, pin_memory = True)
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

### GRID SEARCH
accs = []
for i in range(30, 50):
    # for j in range(1,):
    custom_dataset_adv_cutout = custom_loader(og_adv_examples_format, og_adv_labels_format, cutout = True, i = i, j = 2)
    test_loader_adv_cutout_noise = torch.utils.data.DataLoader(custom_dataset_adv_cutout,
        batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True)
    
    result_cutout_noise = validation(model, test_loader_adv_cutout_noise)
    print(i, 2, result_cutout_noise)
        


#### RESULT WITHOUT ADV NOISE
# result_inpainted_nonoise = validation(model, test_loader_inpt_nonoise)
# result_cutout_nonoise = validation(model, test_loader_cutout_nonoise)
# result_og_nonoise = validation(model, test_loader_og_nonoise)

#### RESULT WITH ADV NOISE 
# result_inpainted_noise = validation(model, test_loader_adv_inpt_noise)
# result_cutout_noise = validation(model, test_loader_adv_cutout_noise)
# result_og_noise =  validation(model, test_loader_adv_og_noise)


# print("#### NO NOISE ####")
# print("Inpainted:", result_inpainted_nonoise)
# print("Cutout:",result_cutout_nonoise)
# print("Original:",result_og_nonoise)
# print()
# print("#### ADV NOISE ####")
# print("Inpainted:", result_inpainted_noise)
# print("Cutout:", result_cutout_noise)
# print("Original:", result_og_noise)

# %% plot acc
from scipy.interpolate import make_interp_spline, BSpline
import numpy as np
import matplotlib.pyplot as plt

def smooth(x, y):
    xnew = np.linspace(x.min(), x.max(), 300) 
    spl = make_interp_spline(x, y, k=3)  # type: BSpline
    power_smooth = spl(xnew)
    return power_smooth

y_2_cut = np.array([
 0.3095,
 0.3239,
 0.3239,
 0.3351,
 0.3351,
 0.3381,
 0.3381,
 0.3236,
 0.3236,
 0.3122,
 0.3122,
 0.2898,
 0.2898,
 0.2698,
 0.2698,
 0.2505,
 0.2505,
 0.2316,
 0.2316])


y_8_cut = np.array([
 0.3095,
 0.3552,
 0.3552,
 0.359,
 0.359,
 0.2837,
 0.2837,
 0.2254,
 0.2254,
 0.1857,
 0.1857,
 0.1663,
 0.1663,
 0.1537,
 0.1537,
 0.1402,
 0.1402,
 0.1268,
 0.1268])

y_18_cut = np.array([
 0.3095,
 0.3933,
 0.3933,
 0.3258,
 0.3258,
 0.1768,
 0.1768,
 0.1301,
 0.1301,
 0.1247,
 0.1247,
 0.1192,
 0.1192,
 0.1118,
 0.1118,
 0.1112,
 0.1112,
 0.1081,
 0.1081
])
x = np.arange(1, 20, 1)
plt.plot(x, y_18_cut, label = '18 cut-outs')
plt.plot(x, y_8_cut, label = '8 cut-outs')
plt.plot(x, y_2_cut, label = '2 cut-outs')
# xnew = np.linspace(x.min(), x.max(), 300)
# plt.plot(xnew, smooth(x, y1), label = '18')
# plt.plot(xnew, smooth(x, y2), label = '7')
# plt.plot(xnew, smooth(x, y3), label = '8')
plt.legend()
plt.grid()
plt.xlabel('px width of cut-out')
plt.ylabel('acc %')
# plt.plot(xnew, power_smooth)
# plt.plot(x, y2)
# plt.plot(x, y3)
plt.show()
