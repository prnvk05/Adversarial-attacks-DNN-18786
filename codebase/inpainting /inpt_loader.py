# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from tqdm import tqdm
import torch.optim as optim
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
import pickle
import matplotlib.pyplot as plt
from inpt_test_resnet import resnet32
# %%
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

val_set = datasets.CIFAR10(root='./data',download=True, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))
print(val_set.data.shape)
 # %%
inpaint_examples = np.load('/root/Adversarial-attacks-DNN-18786/Inpainting_experiment/test_data/test_inpainted.npy', allow_pickle = True)
inpaint_labels = np.load('/root/Adversarial-attacks-DNN-18786/Inpainting_experiment/test_data/label_inpainted.npy', allow_pickle = True)

inpaint_examples_format = inpaint_examples.squeeze(1)
inpaint_labels_format = list(inpaint_labels.squeeze(1))

val_set.data = inpaint_examples_format
val_set.targets = inpaint_labels_format
#%%

