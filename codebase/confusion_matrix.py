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
import seaborn as sns; sns.set()
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from inception_train import inception_v3

#%%


device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
model = resnet32().cuda()
model.eval()
checkpoint = torch.load('/root/Adversarial-attacks-DNN-18786/saved_model/resnet32-d509ac18.th')
state_dict = checkpoint['state_dict']

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)

val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data',download=True, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=1, shuffle=False,
    num_workers=4, pin_memory=True)

target_labels = []
predicted_labels = []


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
    top1_accuracy += torch.sum(torch.eq(top1_pred_labels, label)).item()
    target_labels.append(label.item())
    predicted_labels.append(top1_pred_labels.item())
    total += len(label)
  model.train()
  return top1_accuracy/total


print("validation Acc is :",validation(model,val_loader))

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
cm = confusion_matrix(target_labels, predicted_labels)
plt.figure(figsize=(9,9))
sns.heatmap(cm,cmap=plt.cm.Blues,annot = True,fmt="d",xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

df = pd.DataFrame(cm)
df.to_csv('/root/Adversarial-attacks-DNN-18786/saved_model/confusion_matrix.csv', index = False)
# %%

## Generating CF for Inception Model
model_inception = inception_v3(device = 'cuda')
chkp = torch.load('/root/Adversarial-attacks-DNN-18786/saved_model/incepv3.pkl')
model_inception.load_state_dict(chkp)

# %%
