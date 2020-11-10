
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
#%%
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

val_set = datasets.CIFAR10(root='./data',download=True, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))

adv_examples = np.load('/root/Adversarial-attacks-DNN-18786/PGD_DATA/alpha_adv_examples_pgd.npy', allow_pickle = True)

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)

def pre_process(im):
    im = inv_normalize(torch.Tensor(im)).numpy()
    im = np.clip(im, 0, 1)
    im = im*255
    im = im.astype(np.uint8)
    return im

normal_ims = [pre_process(exp[4]) for exp in adv_examples[4]]
examples_4 = [pre_process(exp[3]) for exp in adv_examples[4]]
targets_4 = [tgt[0] for tgt in adv_examples[4]]
examples_4_arr = np.asarray(examples_4)
targets_4_arr = np.asarray(targets_4)
examples_4_arr_T = np.transpose(examples_4_arr, (0,2,3,1))

val_set.data = np.vstack((val_set.data, examples_4_arr_T))
val_set.targets = np.hstack((val_set.targets, targets_4_arr))






# combined_pgd_normal = (combined_data, combined_labels)
# with open('combined_pgd_normal.pickle', 'wb') as f:
#     pickle.dump(combined_pgd_normal, f)

# np.save('pgd+normal.npy', combined_set)
# %%
# with open('/root/Adversarial-attacks-DNN-18786/codebase/combined_pgd_normal.pickle', 'rb') as f:
#     mynewlist2 = pickle.load(f)