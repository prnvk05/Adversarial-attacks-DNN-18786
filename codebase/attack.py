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
# %%

class Attack:

class PGD:
    def __init__(self, dataloader, epsilon, alpha, steps=5):
        self.epsilon = epsilon
        self.alpha = alpha
        self.dataloader = dataloader
        self.steps = steps
        self.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
        self.model = resnet32().to(self.device)
        self.model.eval()
        checkpoint = torch.load('/root/Adversarial-attacks-DNN-18786/saved_model/resnet32-d509ac18.th')

        state_dict = checkpoint['state_dict']

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict) 
    
    def attack(self, image, label):
        adv_img = image.clone()
        # adv_img.requires_grad = True
        for i in range(self.steps):
            # adv_img.is_leaf = True
            # import pdb; pdb.set_trace()
            if adv_img.requires_grad != True:
                adv_img.requires_grad = True
            out = self.model(adv_img)
            loss = F.nll_loss(out, label)
            # model.zero_grad()
            # loss.backward()
            grad = torch.autograd.grad(loss, adv_img,
                                        retain_graph=False, create_graph=False)[0]
            # grad = adv_img.grad
            sign_grad = grad.sign()
            adv_img = adv_img + self.alpha*sign_grad
            delta = torch.clamp(adv_img - image,-self.epsilon,self.epsilon)
            adv_img = torch.clamp(image + delta,torch.min(image).item(),torch.max(image).item())
        
        return adv_img    

    def test(self):
        correct = 0
        adv_examples = []

        # Loop over all examples in test set
        for data, target in self.dataloader:

            # Send the data and label to the device
            data, target = data.to(self.device), target.to(self.device)

            # Set requires_grad attribute of tensor. Important for Attack
            data.requires_grad = True

            # Forward pass the data through the model
            output = self.model(data)
            init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

            # If the initial prediction is wrong, dont bother attacking, just move on
            if init_pred.item() != target.item():
                continue


            # Call PGD Attack
            perturbed_data = self.attack(data, target)

            output = self.model(perturbed_data)

            # Check for success
            final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            if final_pred.item() == target.item():
                correct += 1
            else:

                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                data_ex = data.squeeze().detach().cpu().numpy()
                adv_examples.append((target.item(),init_pred.item(), final_pred.item(), adv_ex,data_ex))

        # Calculate final accuracy for this epsilon
        final_acc = correct/float(len(self.dataloader))
        print("Epsilon: {}, Alpha: {}\tTest Accuracy = {}".format(self.epsilon, self.alpha,  final_acc))

        # Return the accuracy and an adversarial example
        return final_acc, adv_examples   

class FGSM:
    def __init__(self, dataloader, epsilon):
        self.epsilon = epsilon
        self.dataloader = dataloader
        self.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

        self.model = resnet32().to(self.device)
        self.model.eval()
        checkpoint = torch.load('/root/Adversarial-attacks-DNN-18786/saved_model/resnet32-d509ac18.th')

        state_dict = checkpoint['state_dict']

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)       

    def attack(self, image, epsilon, data_grad):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + epsilon*sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, torch.min(image).item(), torch.max(image).item())
        # Return the perturbed image
        return perturbed_image 


    def test(self):

        # Accuracy counter
        correct = 0
        adv_examples = []

        # Loop over all examples in test set
        for data, target in self.dataloader:
            # Send the data and label to the device
            data, target = data.to(self.device), target.to(self.device)
            # Set requires_grad attribute of tensor. Important for Attack
            data.requires_grad = True
            # Forward pass the data through the model
            output = self.model(data)
            init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            # if init_pred.item() != target.item():
            #     continue
            # Calculate the loss
            loss = F.nll_loss(output, target)
            # Zero all existing gradients
            self.model.zero_grad()
            # Calculate gradients of model in backward pass
            loss.backward()
            # Collect datagrada
            data_grad = data.grad.data
            # Call FGSM Attack
            perturbed_data = self.attack(data, self.epsilon, data_grad)
            # Re-classify the perturbed image
            output = self.model(perturbed_data)
            # Check for success
            final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            if final_pred.item() == target.item():
                correct += 1
            
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            data_ex = data.squeeze().detach().cpu().numpy()
            adv_examples.append((target.item(),init_pred.item(), final_pred.item(), adv_ex,data_ex))
        # Calculate final accuracy for this epsilon
        final_acc = correct/float(len(self.dataloader))
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(self.epsilon, correct, self.dataloader, final_acc))
        # Return the accuracy and an adversarial example
        return final_acc, adv_examples

# %%
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

                        
val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data',download=True, train=False, transform=transforms.Compose([

        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=1, shuffle=False,
    num_workers=4, pin_memory=True)

# fgsmatk = FGSM(val_loader, 0.05)
pgdatk = PGD(val_loader, 0.05, 4/255, 5)