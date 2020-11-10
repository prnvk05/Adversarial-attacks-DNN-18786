#%%
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
import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('/root/Adversarial-attacks-DNN-18786/saved_model/adv_examples.pkl', 'rb') as f:
    mynewlist = pickle.load(f)
with open('/root/Adversarial-attacks-DNN-18786/saved_model/adv_examples1.pkl', 'rb') as f:
    mynewlist1 = pickle.load(f)

with open('/root/Adversarial-attacks-DNN-18786/saved_model/adv_examples15.pkl', 'rb') as f:
    mynewlist15 = pickle.load(f)

with open('/root/Adversarial-attacks-DNN-18786/saved_model/adv_examples2.pkl', 'rb') as f:
    mynewlist2 = pickle.load(f)

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)
def imshow(img):
    img_t = torch.tensor(img)
    un_img = inv_normalize(img_t)
    npimg = un_img.numpy()


    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    # plt.show()
    plt.savefig('/root/Adversarial-attacks-DNN-18786/pics/car_0.2.png')



# imshow(torchvision.utils.make_grid(torch.from_numpy(mynewlist[0][5][4])))
# imshow(torchvision.utils.make_grid(torch.from_numpy(mynewlist[0][5][3])))
# imshow(torchvision.utils.make_grid(torch.from_numpy(mynewlist1[0][5][3])))


# imshow(torchvision.utils.make_grid(torch.from_numpy(mynewlist15[0][4][3])))

imshow(torchvision.utils.make_grid(torch.from_numpy(mynewlist2[0][4][3])))








# %%
import numpy as np
load_accuracies = np.load('/root/Adversarial-attacks-DNN-18786/PGD_DATA/alpha_accuracies_pgd.npy')
adv_examples = np.load('/root/Adversarial-attacks-DNN-18786/PGD_DATA/alpha_adv_examples_pgd.npy', allow_pickle = True)
alpha_epsilon_pgd = np.load('/root/Adversarial-attacks-DNN-18786/PGD_DATA/alpha_epsilons_pgd.npy')

# %%
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)
def imshow(img, acc):
    img_t = torch.tensor(img)
    un_img = inv_normalize(img_t)
    npimg = un_img.numpy()


    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    # plt.show()
    plt.savefig(f'/root/Adversarial-attacks-DNN-18786/pics/ship_og_{acc}.png')


# %%
# 0,3,4 - car
# 0,0,4 - sip



# im_og = imshow(torchvision.utils.make_grid(torch.from_numpy(adv_examples[0][0][4])))
im1 = adv_examples[0][0][3]
im2= adv_examples[1][2][3]
im3= adv_examples[2][2][3]
im4= adv_examples[3][2][3]
im5= adv_examples[4][2][3]

ims = [im1, im2, im3, im4, im5]

for idx, im in enumerate(ims):
    imshow(torchvision.utils.make_grid(torch.from_numpy(im)), str(load_accuracies[idx]))


