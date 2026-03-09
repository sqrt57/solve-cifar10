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
import math
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
# from torch.utils.data import DataLoader, Subset
import torchvision
from torch.profiler import profile, ProfilerActivity, record_function

from solve.modeling.train import Hyper, Result, Trainer
import solve.modeling.model as models
from solve.dataset import cifar10_whole_dataset, DataSet, DataLoader

# %%
device = torch.accelerator.current_accelerator().type
# device = "cpu"
print(f"Using {device} device")

# %%
data_path = "../data/external"
ds = cifar10_whole_dataset(data_path, 0.1).to(device)
training_loader = DataLoader(ds.training, batch_size=200, shuffle=True)

# %%
trainer = Trainer(training_loader, ds.validation, device=device)

# %%
# seed = 682200895
# optimizer = torch.optim.Adam
# optimizer_kwargs = {}
# # lr_schedule = [(20,1e-3), (50,1e-4), (50, 1e-5)]
# lr_schedule = [(5,3e-4)]
# hypers = [
#     Hyper(name=f'simple', seed=seed, model=models.SimpleModelPrelu, optimizer=optimizer, optimizer_kwargs=optimizer_kwargs, lr_schedule=lr_schedule, preprocess=False),
#     Hyper(name=f'Resnet8', seed=seed, model=models.Resnet8, optimizer=optimizer, optimizer_kwargs=optimizer_kwargs, lr_schedule=lr_schedule, preprocess=False),
#     Hyper(name=f'Resnet20', seed=seed, model=models.Resnet20, optimizer=optimizer, optimizer_kwargs=optimizer_kwargs, lr_schedule=lr_schedule, preprocess=False),
# ]

# %%
seed = 682200895
optimizer = torch.optim.AdamW
optimizer_kwargs = {}
# lr_schedule = [(20,1e-3), (50,1e-4), (50, 1e-5)]
lr_schedule = [(50,1e-3),(50,1e-4),(50,1e-5)]
hypers = [
    Hyper(name=f'base', seed=seed, model=models.SimpleModel, optimizer=optimizer, optimizer_kwargs=optimizer_kwargs, lr_schedule=lr_schedule, preprocess=False),
    Hyper(name=f'BN', seed=seed, model=models.SimpleModelBatchNorm, optimizer=optimizer, optimizer_kwargs=optimizer_kwargs, lr_schedule=lr_schedule, preprocess=False),
    Hyper(name=f'PReLU', seed=seed, model=models.SimpleModelPrelu, optimizer=optimizer, optimizer_kwargs=optimizer_kwargs, lr_schedule=lr_schedule, preprocess=False),
    Hyper(name=f'BN PReLU', seed=seed, model=models.SimpleModelBatchNormPrelu, optimizer=optimizer, optimizer_kwargs=optimizer_kwargs, lr_schedule=lr_schedule, preprocess=False),
    Hyper(name=f'Net', seed=seed, model=models.Net, optimizer=optimizer, optimizer_kwargs=optimizer_kwargs, lr_schedule=lr_schedule, preprocess=False),
]

# %%

# %%
results = []
for hyper in hypers:
    results.append(trainer.run_scenario(hyper))

# %%
for r in results:
    npars = sum(p.numel() for p in r.model.parameters())
    print(f"model={r.hyper.model.__name__} {npars=}")

# %%
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,  8))

for result in results:
    ax1.plot(result.epochs, result.train_metrics['Accuracy'], label=result.hyper.name)
# ax1.set_ylim(50., 100.)
ax1.set_ylim(0., 100.)
ax1.set_xlabel("Epoch")
ax1.set_title("Train accuracy")
ax1.grid(True)
ax1.legend()

for result in results:
    ax2.plot(result.epochs, result.validation_metrics['Accuracy'], label=result.hyper.name)
# ax2.set_ylim(50., 100.)
ax2.set_ylim(0., 100.)
ax2.set_xlabel("Epoch")
ax2.set_title("Test accuracy")
ax2.grid(True)
ax2.legend()

for result in results:
    ax3.plot(result.epochs, result.train_metrics['Loss'], label=result.hyper.name)
ax3.set_ylim(0., 5.)
ax3.set_xlabel("Epoch")
ax3.set_title("Train loss")
ax3.grid(True)
ax3.legend()

for result in results:
    ax4.plot(result.epochs, result.validation_metrics['Loss'], label=result.hyper.name)
ax4.set_ylim(0., 5.)
ax4.set_xlabel("Epoch")
ax4.set_title("Test loss")
ax4.grid(True)
ax4.legend()

fig.tight_layout()
plt.plot()

# %%
