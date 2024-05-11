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
import numpy as np
from matplotlib.colors import hsv_to_rgb

from network import DDPM, ContextUnet, ContextUnetColored, HueRegressor
from dataset import add_hue_confounded, classifiedMNIST

def main():
    parser = argparse.ArgumentParser(description='Train a model on MNIST with configurable parameters.')
    parser.add_argument('--date', type=str, help='Date for saving experiments')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--p_unif', type=float, default=None, help='Used in generating training data, NOT in classified data')
    parser.add_argument('--cond_mode', type=str, choices=['Attention', 'AdaCat', 'AdaGN'], default='Attention', help='Conditioning mode')
    parser.add_argument('--classifier_name', type=str, choices=[None,'0','0.01','0.05','0.1','1'], default=None, help='Classifier name for classified dataset')
    parser.add_argument('--class_type', type=str, choices=['label', 'logit'], default='label', help='Type of class encoding')
    parser.add_argument('--lmda', type=float, help='strength of CIL')
    args = parser.parse_args()

    train_mnist(args)

def train_mnist(args):

    # hardcoding these here
    n_epoch = 30
    batch_size = 256
    n_T = 400 # 500
    device = f"cuda:{args.rank}" if torch.cuda.is_available() else "cpu"
    n_classes = 10
    n_feat = 256 # 128 ok, 256 better (but slower)
    drop_prob = 0.5
    p_unif = args.p_unif
    lrate = 1e-4
    save_model = True
    save_gif = False
    save_dir = f'./experiments/{args.date}/'
    mnist_color = True                  # use color mnist
    model_color = True                  # use color model
    independent_mask = True             # sample hue and context mask independently or same
    cond_mode = args.cond_mode            # "AdaCat" or "AdaGN"
    classifier_name = args.classifier_name    # if not None, use classified dataset
    class_type = args.class_type           # "label" or "logits" (logits only used for classified data)
    ws_test = [0.0, 1.0]                # strength of generative guidance
    lmda = args.lmda                    # strength of CIL

    assert p_unif is not None or classifier_name is not None, \
        "at least one of p_unif and classifier_name should be set to indicate"

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir + "image", exist_ok=True)
    os.makedirs(save_dir + "gif", exist_ok=True)
    os.makedirs(save_dir + "model", exist_ok=True)
    if model_color:
        ddpm = DDPM(nn_model=ContextUnetColored(in_channels=3, \
            n_feat=n_feat, n_classes=n_classes, cond_mode=cond_mode), betas=(1e-4, 0.02), \
            n_T=n_T, device=device, drop_prob=drop_prob, color=True, independent_mask=independent_mask)
    else:
        ddpm = DDPM(nn_model=ContextUnet(
            in_channels=3 if mnist_color else 1, \
            n_feat=n_feat, n_classes=n_classes), \
            betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=drop_prob)
    ddpm.to(device)

    hue_reg_g = HueRegressor(n_classes, n_feat)
    hue_reg_g.to(device)
    hue_reg_h = HueRegressor(n_classes, n_feat)
    hue_reg_h.to(device)

    # optionally load a model
    # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))

    tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1
    if classifier_name is None:
        dataset = MNIST("./data", train=True, download=True, transform=tf)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    # optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
    optim_g = torch.optim.AdamW(hue_reg_g.parameters(), lr=lrate)
    optim_other = torch.optim.AdamW(list(ddpm.parameters())+list(hue_reg_h.parameters()), lr=lrate)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim_g.param_groups[0]['lr'] = lrate*(3-ep/n_epoch)/3
        optim_other.param_groups[0]['lr'] = lrate*(3-ep/n_epoch)/3
        if classifier_name is not None:
            pbar = tqdm(range(60000//batch_size))
        else:
            pbar = tqdm(dataloader)
        loss_ema = None
        for val in pbar:
            if classifier_name is not None:
                x, c_true, hues = torch.load(f'./classifiedMNIST/real_data/batch_{val}.pth', map_location=device)
                c_preds, logits = torch.load(f'./classifiedMNIST/predicted_label/batch_{val}.pth', map_location=device)
                c = c_preds[classifier_name]
                logit = logits[classifier_name]
            else:
                x, c = val
                hues = None
                if mnist_color:
                    x, hues = add_hue_confounded(x, c, p_unif)

            optim_g.zero_grad()
            x = x.to(device)    # batch, 1, 28, 28
            c = c.to(device)    # batch
            hues = hues.to(device) if hues is not None else None

            hidden_vec = ddpm.nn_model.encode(x)
            hue_pred_g = hue_reg_g(hidden_vec, c)
            loss_g = F.mse_loss(hue_pred_g.view(-1), hues.view(-1))
            loss_g.backward()
            optim_g.step()
            
            optim_other.zero_grad()
            if class_type == "label":
                loss_ddpm = ddpm(x, c, hues)
            elif class_type == "logit":
                loss_ddpm = ddpm(x, logit, hues)
            hidden_vec = ddpm.nn_model.encode(x)
            hue_pred_h = hue_reg_h(hidden_vec, torch.zeros_like(c, device=device))
            hue_pred_g = hue_reg_g(hidden_vec, c)
            loss_h = F.mse_loss(hue_pred_h.view(-1), hues.view(-1))
            loss_g = F.mse_loss(hue_pred_g.view(-1), hues.view(-1))
            loss = loss_ddpm + lmda * (loss_h - loss_g)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f} g: {loss_g.item():.4f} h: {loss_h.item():.4f} ddpm: {loss_ddpm.item():.4f}")
            optim_other.step()
        
        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        ddpm.eval()
        with torch.no_grad():
            n_sample = 4*n_classes
            for w_i, w in enumerate(ws_test):
                if ep%5==0 or ep == int(n_epoch-1):
                    x_gen, x_gen_store = ddpm.sample(n_sample, (3, 28, 28), device, guide_w=w)

                    # append some real images at bottom, order by class also
                    x_real = torch.Tensor(x_gen.shape).to(device)
                    for k in range(n_classes):
                        for j in range(int(n_sample/n_classes)):
                            try: 
                                idx = torch.squeeze((c == k).nonzero())[j]
                            except:
                                idx = 0
                            x_real[k+(j*n_classes)] = x[idx]

                    x_all = torch.cat([x_gen, x_real])
                    grid = make_grid(x_all*-1 + 1, nrow=10)
                    save_image(grid, save_dir + f"image/image_ep{ep}_w{w}.png")
                    print('saved image at ' + save_dir + f"image/image_ep{ep}_w{w}.png")

                if save_gif and (ep%5==0 or ep == int(n_epoch-1)):
                    # create gif of images evolving over time, based on x_gen_store
                    fig, axs = plt.subplots(nrows=int(n_sample/n_classes), ncols=n_classes,sharex=True,sharey=True,figsize=(8,3))
                    def animate_diff(i, x_gen_store):
                        print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                        plots = []
                        for row in range(int(n_sample/n_classes)):
                            for col in range(n_classes):
                                axs[row, col].clear()
                                axs[row, col].set_xticks([])
                                axs[row, col].set_yticks([])
                                # plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
                                plots.append(axs[row, col].imshow(-x_gen_store[i,(row*n_classes)+col,0],cmap='gray',vmin=(-x_gen_store[i]).min(), vmax=(-x_gen_store[i]).max()))
                        return plots
                    ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store],  interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])    
                    ani.save(save_dir + f"gif/gif_ep{ep}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
                    print('saved image at ' + save_dir + f"gif/gif_ep{ep}_w{w}.gif")
        # optionally save model
        if save_model and (ep % 10 == 0 or ep == int(n_epoch-1)):
            torch.save(ddpm.state_dict(), save_dir + f"model/model_{ep}.pth")
            print('saved model at ' + save_dir + f"model/model_{ep}.pth")

if __name__ == "__main__":
    main()

