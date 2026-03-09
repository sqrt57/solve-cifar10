# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import sys
sys.path.append("..")

# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

# %%
data_path = "../data/external"
training_ds = torchvision.datasets.CIFAR10(data_path, download=True, train=True, transform=torchvision.transforms.ToTensor())
test_ds = torchvision.datasets.CIFAR10(data_path, download=True, train=False, transform=torchvision.transforms.ToTensor())

# %%
training_ds.classes

# %%
training_dl = DataLoader(

# %%
training_ds[0][1]

# %%
plt.figure(figsize=(.5, .5))
plt.imshow(training_ds[33][0].movedim(0, 2))

# %%
torch.corrcoef(training_ds[33][0].reshape(3,1024))

# %%
x = torch.tensor([[0, 1, 2], [2, 1, 0]])
torch.corrcoef(x)

# %%
x = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], dtype=torch.float)-5
y = (x/torch.sqrt((x**2).sum((1,2), keepdim=True))).reshape(3, 4)
print(x.shape)
print(x)
print(y.shape)
print(y)
torch.corrcoef(y)

# %%
ds = training_ds
fig, axs = plt.subplots(10,10)
fig.set_size_inches(12,12)
for i in range(100):
    img = ds[i][0].movedim(0, 2)
    label = ds.classes[ds[i][1]]
    ax = axs.flat[i]
    ax.imshow(img, interpolation='none')
    ax.set_yticks([], [])
    ax.set_xticks([], [])
    ax.set_xlabel(str(label))
fig.tight_layout()
plt.show()

# %%
