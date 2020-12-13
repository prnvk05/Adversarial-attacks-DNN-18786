#%%
import torch
import torchvision.transforms as transforms
import numpy as np
# %%
# Use the helper functions below to visualize saved images. 
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)
def imshow(img):
    img_t = torch.tensor(img)
    un_img = inv_normalize(img_t)
    npimg = un_img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()