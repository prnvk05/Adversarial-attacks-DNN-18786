# %%
from attack import Attack, FGSM, PGD
import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from collections import OrderedDict
import torchvision.datasets as datasets
import torchvision.transforms as transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

                        
val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data',download=True, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=1, shuffle=False,
    num_workers=4, pin_memory=True)

fgsmatk = FGSM(val_loader, 0.05)
pgdatk = PGD(val_loader, 0.05, 4/255, steps = 5)
# final_acc, _ = fgsmatk.test()
# final_acc,_ = pgdatk.test()



# %%

# %%
