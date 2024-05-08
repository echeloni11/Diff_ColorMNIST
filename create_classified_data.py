import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torchvision import datasets, transforms
import torch.nn.functional as F
import timeit
import unittest

from network import Classifier
from dataset import add_hue_confounded, classifiedMNIST

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

# Load the MNIST training, test datasets using `torchvision.datasets.MNIST` 
# using the transform defined above

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST('./data',train=True, transform=transform, download=True)
test_dataset =  datasets.MNIST('./data',train=False, transform=transform, download=True)

train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=256)
test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classifier().to(device)

punif_list = [0,0.01,0.05,0.1,1]
classifiers = {}
for p in punif_list:
    name = f'model{p}_10'
    model = Classifier().to(device)
    state_dict = torch.load(f'./trained_classifiers/{name}.pt', map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    classifiers[str(p)] = model

# Construct a classified dataset (x, y, y_hat, logits, hue)
classified_dataset = []
import os
import tqdm
os.makedirs('./classifiedMNIST/real_data', exist_ok=True)
os.makedirs('./classifiedMNIST/predicted_label', exist_ok=True)
for i, (x, y) in enumerate(train_dataloader):
    x, y = x.to(device), y.to(device)
    x, hues = add_hue_confounded(x, y, p_unif=1)
    logits = {}
    y_hats = {}
    for p, model in classifiers.items():
        logit = model(x)
        logits[p] = logit
        y_hats[p] = torch.argmax(logit, dim=1)
    torch.save((x, y, hues), f'./classifiedMNIST/real_data/batch_{i}.pth')
    torch.save((y_hats, logits), f'./classifiedMNIST/predicted_label/batch_{i}.pth')


