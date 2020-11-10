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

from custom_loader import val_set
# %%
device = "cuda"
model = resnet32().to(device)
# print(model)


train_loader = torch.utils.data.DataLoader(val_set,
    batch_size=128, shuffle=True,
    num_workers=4, pin_memory=True)

# %%
saved_model_path = '/root/Adversarial-attacks-DNN-18786/saved_model/resnet32-adv'
criterion = nn.CrossEntropyLoss().cuda()

optimizer = torch.optim.SGD(model.parameters(), 0.1,
                            momentum=0.9,
                            weight_decay=1e-4)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=[100, 150], last_epoch= - 1)
def train(epochs,model,train_dataloader):
    for epoch in range(epochs):
        avg_loss = 0.0
        model.train()
        print(epoch)
        for batch_num,(feats,targets) in enumerate(train_dataloader):
            feats, targets = feats.to(device), targets.to(device)
            if batch_num % 100 == 0:
                print("Batch No :",batch_num,"/",len(train_dataloader))

            optimizer.zero_grad()
            
            output=model(feats)
            loss = criterion(output,targets)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
          
            
        print('training loss = ', avg_loss/len(train_dataloader))
        
        lr_scheduler.step()

        torch.save({
            'model_state_dict': model.state_dict()
            },saved_model_path)

train(200, model, train_loader)

# %%
def test(model,test_dataloader):

    model.eval()
    total_acc = []
    for batch_num,(feats,targets) in enumerate(test_dataloader):
        correct_predictions = 0.0
        feats, targets = feats.to(device), targets.to(device)
        # if batch_num %:
        print("Batch No :",batch_num,"/",len(test_dataloader))
        with torch.no_grad():
            output=model(feats)
            _, predicted = torch.max(output.data, 1)
            total_predictions += target.size(0)
            correct_predictions = (predicted.cpu().numpy() == targets.cpu().numpy()).mean().item()
            total_acc.append(correct_predictions)
        # print("Accuracy", correct_predictions / len(test_dataloader))

    print("Final", sum(total_acc)/len(total_acc))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data',download=True, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=256, shuffle=False,
    num_workers=4, pin_memory=True)


chkpt = torch.load('/root/Adversarial-attacks-DNN-18786/saved_model/resnet32-adv')
model_test = resnet32()
device = "cuda"
model_test.to(device)
model_test.load_state_dict(chkpt['model_state_dict'])
test(model_test, val_loader)