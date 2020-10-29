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


with open('/root/Adversarial-attacks-DNN-18786/saved_model/adv_examples_pgd.pkl', 'rb') as f:
    mynewlist = pickle.load(f)


def imshow(img):

    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

imshow(torchvision.utils.make_grid(torch.from_numpy(mynewlist[0][2][3])))









# %%
