''' 
This script does conditional image generation on MNIST, using a diffusion model

This code is modified from,
https://github.com/cloneofsimo/minDiffusion

Diffusion model is based on DDPM,
https://arxiv.org/abs/2006.11239

The conditioning idea is taken from 'Classifier-Free Diffusion Guidance',
https://arxiv.org/abs/2207.12598

This technique also features in ImageGen 'Photorealistic Text-to-Image Diffusion Modelswith Deep Language Understanding',
https://arxiv.org/abs/2205.11487

'''
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
import cv2
import numpy as np

from matplotlib.colors import hsv_to_rgb

from network import DDPM, ContextUnet, ContextUnetColored, HueRegressor, DigitRegressor, Classifier
from dataset import add_hue_confounded, classifiedMNIST

def main():
    parser = argparse.ArgumentParser(description='Run test acc experiment for generating counterfactual.')
    parser.add_argument('--date', type=str, help='Date for saving experiments')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--model_name', type=str, help="directory of generative model")
    parser.add_argument('--clean_classifier_name', type=str, help="directory of clean classifier")
    parser.add_argument('--noisy_classifier_name', type=str, help="directory of noisy classifier")
    parser.add_argument('--dataset_type', type=str, choices=["ID", "OOD"], help="type of dataset")
    parser.add_argument('--p_unif', type=float, default=0.01, help='Used in generating test data')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for test data')
    parser.add_argument('--batch_num', type=int, default=100, help='number of batches to test')
    args = parser.parse_args()

    test(args)

def test(args):
    device = f"cuda:{args.rank}" if torch.cuda.is_available() else "cpu"
    n_classes = 10
    n_feat = 256 # 128 ok, 256 better (but slower)
    drop_prob = 0.5
    p_unif = args.p_unif

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

    classifier_clean = Classifier(input_channels=1)
    classifier_clean.load_state_dict(torch.load(args.clean_classifier_name, map_location=device))
    classifier_clean.to(device)
    classifier_clean.eval()

    classifier_noisy = Classifier(input_channels=1)
    classifier_noisy.load_state_dict(torch.load(args.noisy_classifier_name, map_location=device))
    classifier_noisy.to(device)
    classifier_noisy.eval()

    tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1
    dataset = MNIST("./data", train=False, download=True, transform=tf)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=5)

    total_num_per_class = torch.zeros(10, device=device)
    total_num_per_class_clean = torch.zeros(10, device=device)   # total_num_per_class[i] is the number of class i
    total_num_per_class_noisy = torch.zeros(10, device=device)
    # Metric 1: Accuracy
    correct_num_per_class_clean = []  # correct_num_per_class[i][j] is the number of class j do(hue=i) and remain class j
    for i in range(10):
        correct_num_per_class_clean.append(torch.zeros(10, device=device))
    correct_num_per_class_noisy = []  # correct_num_per_class[i][j] is the number of class j do(hue=i) and remain class j
    for i in range(10):
        correct_num_per_class_noisy.append(torch.zeros(10, device=device))
    
    # Metric 2: Decrease of original class logit
    # logit_decrease[i][j] is the decrease of logit of every sample of class j do(hue=i)
    # append the decrease of logit of every sample of class j do(hue=i) and calculate mean std at the end
    logit_decrease_clean = []
    for i in range(10):
        logit_decrease_clean.append([[] for _ in range(10)])
    logit_decrease_noisy = []
    for i in range(10):
        logit_decrease_noisy.append([[] for _ in range(10)])

    # Metric 3: Increase of second largest logit
    # logit_increase[i][j] is the increase of second largest logit of every sample of class j do(hue=i)
    # append the increase of second largest logit of every sample of class j do(hue=i) and calculate mean std at the end
    logit_increase_clean = []
    for i in range(10):
        logit_increase_clean.append([[] for _ in range(10)])
    logit_increase_noisy = []
    for i in range(10):
        logit_increase_noisy.append([[] for _ in range(10)])

    with torch.no_grad():
        for k, (x, c) in enumerate(tqdm(dataloader)):
            if k >= args.batch_num:
                break
            x = x.to(device)    # batch, 1, 28, 28
            c = c.to(device)    # batch

            logit_clean = classifier_clean(x)
            logit_noisy = classifier_noisy(x)
            c_pred_clean = torch.argmax(logit_clean, dim=1)
            c_pred_noisy = torch.argmax(logit_noisy, dim=1)
            error_clean = c_pred_clean != c
            error_noisy = c_pred_noisy != c

            if args.dataset_type == "ID":
                x, hues = add_hue_confounded(x, c, p_unif)
            elif args.dataset_type == "OOD":
                x, hues = add_hue_confounded(x, (c+5)%10, p_unif)

            # total_num_per_class += torch.bincount(c, minlength=10)
            for l in range(len(c)):
                total_num_per_class[c[l]] += 1
                if not error_clean[l]:
                    total_num_per_class_clean[c[l]] += 1
                if not error_noisy[l]:
                    total_num_per_class_noisy[c[l]] += 1

            u = ddpm.abduct(x, c, size=(3,28,28), device=device, guide_w=2.0, hues=hues)
            for i in range(10):
                target_hues = torch.ones_like(hues, device=device) * i / 10 + 0.05 
                x_cf_i = ddpm.reconstruct(u, c, size=(3,28,28), device=device, guide_w=2.0, hues=target_hues)
                # turn x_cf_i into grayscale

                x_cf_i_gray = 0.299 * x_cf_i[:,0,:,:] + 0.587 * x_cf_i[:,1,:,:] + 0.114 * x_cf_i[:,2,:,:]
                x_cf_i_gray = (x_cf_i_gray / x_cf_i_gray.max(dim=0)[0]).view(x_cf_i_gray.shape[0],1,x_cf_i_gray.shape[1],-1)
                
                # debug
                # print(f"x_cf_i shape: {x_cf_i.shape}")
                # print(f"x_cf_i_gray shape: {x_cf_i_gray.shape}")

                logit_cf_i_clean = classifier_clean(x_cf_i_gray)
                logit_cf_i_noisy = classifier_noisy(x_cf_i_gray)
                c_pred_cf_i_clean = torch.argmax(logit_cf_i_clean, dim=1)
                c_pred_cf_i_noisy = torch.argmax(logit_cf_i_noisy, dim=1)

                # Metric 1: Accuracy
                correct_cf_i_clean = c_pred_cf_i_clean == c
                correct_cf_i_noisy = c_pred_cf_i_noisy == c
                for l in range(len(c)):
                    if correct_cf_i_clean[l] and not error_clean[l]:
                        correct_num_per_class_clean[i][c[l]] += 1
                    if correct_cf_i_noisy[l] and not error_noisy[l]:
                        correct_num_per_class_noisy[i][c[l]] += 1
                
                # Metric 2: Decrease of original class logit
                for l in range(len(c)):
                    if not error_clean[l]:
                        logit_decrease_clean[i][c[l]].append((torch.exp(logit_clean[l][c[l]]) - torch.exp(logit_cf_i_clean[l][c[l]])).item())
                    if not error_noisy[l]:
                        logit_decrease_noisy[i][c[l]].append((torch.exp(logit_noisy[l][c[l]]) - torch.exp(logit_cf_i_noisy[l][c[l]])).item())

                # Metric 3: Increase of second largest logit
                for l in range(len(c)):
                    if not error_clean[l]:
                        second_largest_index_clean = torch.argsort(logit_cf_i_clean[l], descending=True)[1] \
                            if c[l] == torch.argmax(logit_cf_i_clean[l]) else torch.argmax(logit_cf_i_clean[l])
                        logit_increase_clean[i][c[l]].append((torch.exp(logit_cf_i_clean[l][second_largest_index_clean]) - torch.exp(logit_clean[l][second_largest_index_clean])).item())
                    if not error_noisy[l]:
                        second_largest_index_noisy = torch.argsort(logit_cf_i_noisy[l], descending=True)[1] \
                            if c[l] == torch.argmax(logit_cf_i_noisy[l]) else torch.argmax(logit_cf_i_noisy[l])
                        logit_increase_noisy[i][c[l]].append((torch.exp(logit_cf_i_noisy[l][second_largest_index_noisy]) - torch.exp(logit_noisy[l][second_largest_index_noisy])).item())
            
            # log current batch information

            current_correct_clean = [torch.sum(correct_num_per_class_clean[i])/torch.sum(total_num_per_class_clean[i]) for i in range(10)]
            current_correct_noisy = [torch.sum(correct_num_per_class_noisy[i])/torch.sum(total_num_per_class_noisy[i]) for i in range(10)]

            with open(f"{save_dir}log/log_result.txt", "a") as f:
                f.write(f"total_num_per_class: {total_num_per_class}\n")
                f.write(f"correct_num_per_class: {correct_num_per_class}\n")
                f.write(f"logit_decrease_clean: {logit_decrease_clean}\n")
                f.write(f"logit_decrease_noisy: {logit_decrease_noisy}\n")
                f.write(f"logit_increase_clean: {logit_increase_clean}\n")
                f.write(f"logit_increase_noisy: {logit_increase_noisy}\n")
                f.write(f"\n")

            with open(f"{save_dir}log/log_detail.txt", "a") as f:
                f.write(f"total_num_per_class: {total_num_per_class}\n")
                f.write(f"correct_num_per_class: {correct_num_per_class}\n")
                f.write(f"logit_decrease_clean: {logit_decrease_clean}\n")
                f.write(f"logit_decrease_noisy: {logit_decrease_noisy}\n")
                f.write(f"logit_increase_clean: {logit_increase_clean}\n")
                f.write(f"logit_increase_noisy: {logit_increase_noisy}\n")
                f.write(f"\n")
    
if __name__ == "__main__":
    main()

