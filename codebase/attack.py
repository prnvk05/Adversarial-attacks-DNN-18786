
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
from inception_train import inception_v3

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
#%%
# model_inception = inception_v3(device = 'cuda')
# chkp = torch.load('/root/Adversarial-attacks-DNN-18786/saved_model/incepv3.pkl')
# model_inception.load_state_dict(chkp)
#%%
model = resnet32().cuda()
model.eval()
checkpoint = torch.load('/root/Adversarial-attacks-DNN-18786/saved_model/resnet32-d509ac18.th')

state_dict = checkpoint['state_dict']

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data',download=True, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=1, shuffle=False,
    num_workers=4, pin_memory=True)

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
    total += len(label)
  model.train()
  return top1_accuracy/total


#print(validation(model,val_loader))
# %%

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    #import pdb; pdb.set_trace()
    perturbed_image = torch.clamp(perturbed_image, torch.min(image).item(), torch.max(image).item())
    # Return the perturbed image
    return perturbed_image


        
def test( model,model_inception, device, test_loader, epsilon ):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        output_inception = model_inception(data)
        
        # import pd
        # b; pdb.set_trace()
        # If the initial prediction is wrong, dont bother attacking, just move on
        # print(init_pred,target)
        # init_pred = init_pred
        # print(init_pred,target)
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        # loss = F.nll_loss(output_inception, target)
        loss = F.nll_loss(output,target)

        # Zero all existing gradients
        # model_inception.zero_grad()
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call PGD Attack
        #perturbed_data = PGD_attack(data, epsilon, data_grad, 5)

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            #if (epsilon == 0) and (len(adv_examples) < 5):
            #    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            #    adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            data_ex = data.squeeze().detach().cpu().numpy()
            adv_examples.append((target.item(),init_pred.item(), final_pred.item(), adv_ex,data_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


accuracies = []
examples = []
# epsilons = [0.10,0.15,0.20,0.25,0.3]
epsilons = [0.05]

# Run test for each epsilon
for eps in epsilons:
    print("running:",eps)
    acc, ex = test(model,model_inception,device, val_loader, eps)
    accuracies.append(acc)
    examples.append(ex)
# %%


# # #%%
# import pickle

# with open('/root/Adversarial-attacks-DNN-18786/saved_model/adv_examples2.pkl', 'wb') as f:
#     pickle.dump(examples, f)
# # %%
# with open('/root/Adversarial-attacks-DNN-18786/saved_model/adv_examples2.pkl', 'rb') as f:
#     mynewlist = pickle.load(f)
# # %%
# #%%
# import matplotlib.pyplot as plt
# epsilons = [0,0.05,0.1,0.15,0.2,0.25,0.3]
# acc_f = [0.9263,0.3011,0.197,0.1503,0.1237,0.1092,0.1002]
# plt.grid()
# plt.axis([0,0.3,0,1])
# plt.xlabel('epsilon')
# plt.ylabel('accuracy')
# plt.plot(epsilons,acc_f)
# plt.title('FGSM Attack')
# plt.savefig('/root/Adversarial-attacks-DNN-18786/pics/plot.png')
# # %%
