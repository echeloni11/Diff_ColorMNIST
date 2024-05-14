import os
import argparse
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
from sklearn.decomposition import PCA

from network import DDPM, ContextUnet, ContextUnetColored, HueRegressor, DigitRegressor
from dataset import add_hue_confounded, classifiedMNIST

# load a DDPM model
# use MNIST test loader to sample one image
# add ten different hues to the image (0.05, 0.15, ..., 0.95)
# get the features for each of the ten images

# Metric 1: Cosine similarity between features of hue 0.05 and features of hue 0.55
# calculate the cosine similarity between features of hue 0.05 and features of hue 0.55

# Metric 2: Append all features labeled as (digit, hue) to a list and use PCA to reduce the dimensionality to 2 and plot
# When plotting, color the points according to the hue, use different shape for different 
# q: Are there this much shapes (for ten digits) when plotting this kind of PCA plot?
# a: Yes, there are ten different shapes for ten different digits


def main():
    parser = argparse.ArgumentParser(description='Train a model on MNIST with configurable parameters.')
    parser.add_argument('--date', type=str, help='Date for saving experiments')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--model_name', type=str, help='Path to the model')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--batch_num', type=int, default=100)
    # parser.add_argument('--plot_digit', type=int, default=None, help='if set, only plot this digit in PCA')
    args = parser.parse_args()

    calculate_feature_distance(args)

def calculate_feature_distance(args):
    device = f"cuda:{args.rank}" if torch.cuda.is_available() else "cpu"
    n_classes = 10
    n_feat = 256 # 128 ok, 256 better (but slower)
    drop_prob = 0.5
    save_freq = args.batch_num // 20
    # p_unif = args.p_unif
    total_batch = args.batch_num
    # plot_digit = args.plot_digit

    save_dir = f'./test_experiments/{args.date}/'

    independent_mask = True             # sample hue and context mask independently or same
    cond_mode = "AdaGN"            # "AdaCat" or "AdaGN"
    class_type = "label"           # "label" or "logits" (logits only used for classified data)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir + "image", exist_ok=True)
    os.makedirs(save_dir + "log", exist_ok=True)

    ddpm = DDPM(nn_model=ContextUnetColored(in_channels=3, \
            n_feat=n_feat, n_classes=n_classes, cond_mode=cond_mode), betas=(1e-4, 0.02), \
            n_T=400, device=device, drop_prob=drop_prob, color=True, independent_mask=independent_mask)

    ddpm.load_state_dict(torch.load(args.model_name, map_location=device))
    ddpm.to(device)
    ddpm.eval()

    tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1
    dataset = MNIST("./data", train=False, download=True, transform=tf)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=5)

    # loop over batch and add hue to it
    cosine_similarities = []
    extracted_features = [] # list of (digit, hue, feature)
    for i, (x, y) in enumerate(tqdm(dataloader)): 
        if i >= total_batch:
            break
        x = x.to(device)
        y = y.to(device)
        ten_features = []
        for target in range(10):
            x_hue, hue = add_hue_confounded(x, torch.tensor([target]).to(device), sigma=0)

            with torch.no_grad():
                feature = ddpm.nn_model.encode(x_hue)
            ten_features.append(feature)
        
        # calculate cosine similarity between features of hue 0.05 and hue 0.55
        cosine_similarity = F.cosine_similarity(ten_features[1], ten_features[6], dim=1)
        cosine_similarities += [val.item() for val in cosine_similarity]

        # append all features labeled as (digit, hue) to a list
        # assume batch_size = 1 first
        for i, feature in enumerate(ten_features):
            extracted_features.append((y[0].item(), i/10+0.05, feature))
    
    # print mean and std of cosine similarities
    print(f"Mean cosine similarity: {np.mean(cosine_similarities)}")
    print(f"Std cosine similarity: {np.std(cosine_similarities)}")

    # plot PCA
    pca = PCA(n_components=2)
    features = [feature.view(-1).detach().cpu().numpy() for _, _, feature in extracted_features]
    pca_features = pca.fit_transform(features)
    for plot_digit in range(10):
        fig, ax = plt.subplots()
        for i, (digit, hue, _) in enumerate(extracted_features):
            # use scatter, set each point color according to hue and shape according to digit
            # if plot_digit is not None:
            if digit != plot_digit:
                continue
            color = plt.cm.viridis(hue)
            ax.scatter(pca_features[i, 0], pca_features[i, 1], c=color, marker=digit)

        # save the plot
        plt.savefig(os.path.join(save_dir, f"image/pca_plot_{plot_digit}.png"))
    
    fig, ax = plt.subplots()
    for i, (digit, hue, _) in enumerate(extracted_features):
        # use scatter, set each point color according to hue and shape according to digit
        # if plot_digit is not None:
        color = plt.cm.viridis(hue)
        ax.scatter(pca_features[i, 0], pca_features[i, 1], c=color, marker=digit)

        # save the plot
    plt.savefig(os.path.join(save_dir, "image/pca_plot.png"))

if __name__ == "__main__":
    main()




