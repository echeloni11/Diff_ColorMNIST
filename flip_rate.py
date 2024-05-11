import os
import argparse
import random
from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from matplotlib.colors import hsv_to_rgb

from network import DDPM, ContextUnet, ContextUnetColored, Classifier
from dataset import add_hue_confounded, classifiedMNIST

dates = ["240508_32", "240508_34", "240508_36", "240508_38", "240508_40"]
p_unifs = [0,0.01,0.05,0.1,1]

rank = 0
i = 0
target_classes = [0,5]
# record the number of samples for each class in test set
# total_samples[i]: total sample predicted as i
total_samples = [0,0,0,0,0,0,0,0,0,0]   
correct_samples = {}
for t in target_classes:
    # correct_samples[t][i]: predicted as i flipped to t success
    correct_samples[t] = [0,0,0,0,0,0,0,0,0,0]

# also save detail information for each (x, y_true)
# record (batch_num, y_true, y_pred, target, logit_change) # 就不保存生成的图片了，如果要看图的话拿单个sample自己实验看看？



seed_value=1234
random.seed(a=seed_value)
np.random.seed(seed=seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)  # for all GPUs

n_T = 400 # 500
device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
n_classes = 10
n_feat = 256 # 128 ok, 256 better (but slower)
batch_size = 16
total_batch = 50
drop_prob = 0.5
p_unif = 1
independent_mask = True             # sample hue and context mask independently or same
cond_mode = "AdaGN"            # "AdaGN"
hue_uncond = True

ddpm = DDPM(nn_model=ContextUnetColored(in_channels=3, \
        n_feat=n_feat, n_classes=n_classes, cond_mode=cond_mode), betas=(1e-4, 0.02), \
        n_T=n_T, device=device, drop_prob=drop_prob, color=True, independent_mask=independent_mask)
ddpm.to(device)
diff_name = f"./experiments/{dates[i]}/model/model_29.pth"
ddpm.load_state_dict(torch.load(diff_name, map_location=device))
ddpm.eval()

classifier = Classifier().to(device)
classifier_name = f"./trained_classifiers/model{p_unifs[i]}_10.pt"
classifier.load_state_dict(torch.load(classifier_name, map_location=device))
classifier.eval()

tf = transforms.Compose([transforms.ToTensor()])
dataset = MNIST("./data", train=False, download=True, transform=tf)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)

# use test set
# use abduct and reconstruct to flip y_pred to target_class
# record total sample and correctly flipped for each starting class and target
# also record those y_pred == target and if they are flipped

with torch.no_grad():
    for i, (x, y_true) in enumerate(tqdm(dataloader)):
        if i == total_batch:
            break
        x, y_true = x.to(device), y_true.to(device)
        x, hues = add_hue_confounded(x, y_true, p_unif=p_unif)
        y_pred_logit = classifier(x)
        y_pred = torch.argmax(y_pred_logit, dim=1)
        for l in range(10):
            total_samples[l] += (y_pred == l).sum().item()
        u = ddpm.abduct(x, y_pred, size=(3,28,28), device=device, guide_w=1, hues=hues, hue_uncond=hue_uncond)
        for target in target_classes:
            x_cf = ddpm.reconstruct(u, torch.zeros_like(y_pred)+target, size=(3,28,28), device=device, guide_w=1, hues=hues, hue_uncond=hue_uncond)
            y_pred_cf_logit = classifier(x_cf)
            y_pred_cf = torch.argmax(y_pred_cf_logit, dim=1)
            for l in range(10):
                correct_samples[target][l] += ((y_pred_cf == target) & (y_pred == l)).sum().item()
            # record (batch_num, y_true, y_pred, target, logit_change)
            record = [i, y_true, y_pred, target, y_pred_cf, y_pred_cf_logit - y_pred_logit]
            with open(f"./flip_results/detail_240509_2.log", "a") as f:
                f.write(str(record)+'\n')

# print the flip rate for each (starting class, target)
for t in target_classes:
    print(f"target class: {t}")
    for l in range(10):
        print(f"from class {l}: {correct_samples[t][l]/total_samples[l]}")
    print("\n")

print(total_samples)

        




