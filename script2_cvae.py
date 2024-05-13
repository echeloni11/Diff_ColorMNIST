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

from network import DigitRegressor, HueRegressor
from cvae import CVAE, plot, loss_function, generate_image
from dataset import add_hue_confounded


def main():
    parser = argparse.ArgumentParser(description='Train a model on MNIST with configurable parameters.')
    parser.add_argument('--date', type=str, help='Date for saving experiments')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--p_unif', type=float, default=None, help='Used in generating training data, NOT in classified data')
    parser.add_argument('--beta', type=float, default=1, help='Beta value for KL divergence')
    parser.add_argument('--lmda', type=float, help='strength of CIL')
    parser.add_argument('--regress_type', type=str, choices=['digit','hue','both'], default='digit',help='use what regressor for CIL')
    args = parser.parse_args()

    train_mnist(args)

def train_mnist(args):

    # hardcoding these here
    n_epoch = 100
    batch_size = 256
    n_T = 400 # 500
    device = f"cuda:{args.rank}" if torch.cuda.is_available() else "cpu"
    latent_size = 64
    n_classes = 10
    n_feat = 256 # 128 ok, 256 better (but slower)
    p_unif = args.p_unif
    beta = args.beta
    lrate = 1e-4
    lmda = args.lmda
    reg_type = args.regress_type
    save_model = True
    save_dir = f'./experiments/{args.date}/'

    assert p_unif is not None or classifier_name is not None, \
        "at least one of p_unif and classifier_name should be set to indicate"

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir + "image", exist_ok=True)
    os.makedirs(save_dir + "gif", exist_ok=True)
    os.makedirs(save_dir + "model", exist_ok=True)

    cvae = CVAE(3, n_feat, latent_size, n_classes, device=device).to(device)

    if reg_type == 'hue' or reg_type == 'both':
        hue_reg_g = HueRegressor(n_classes, n_feat)
        hue_reg_g.to(device)
        hue_reg_h = HueRegressor(n_classes, n_feat)
        hue_reg_h.to(device)
    
    if reg_type == 'digit' or reg_type == 'both':
        digit_reg_g = DigitRegressor(n_feat=n_feat)
        digit_reg_g.to(device)
        digit_reg_h = DigitRegressor(n_feat=n_feat)
        digit_reg_h.to(device)

    # optim = torch.optim.Adam(cvae.parameters(), lr=lrate)
    if reg_type == 'digit':
        optim_g = torch.optim.AdamW(digit_reg_g.parameters(), lr=lrate)
        optim_other = torch.optim.AdamW(list(cvae.parameters())+list(digit_reg_h.parameters()), lr=lrate)
    elif reg_type == 'hue':
        optim_g = torch.optim.AdamW(hue_reg_g.parameters(), lr=lrate)
        optim_other = torch.optim.AdamW(list(cvae.parameters())+list(hue_reg_h.parameters()), lr=lrate)
    elif reg_type == 'both':
        raise NotImplementedError

    # optionally load a model
    # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))

    tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1
    dataset = MNIST("./data", train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    testset = MNIST("./data", train=False, download=True, transform=tf)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=5)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        cvae.train()

        # linear lrate decay
        optim_g.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)
        optim_other.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(dataloader)
        total_loss_ema = None
        reconstruction_loss_ema = None
        kld_loss_ema = None
        for i, val in enumerate(pbar):
            x, c = val
            hues = None

            x, hues = add_hue_confounded(x, c, p_unif)


            x = x.to(device)    # batch, 3, 28, 28
            c = c.to(device)    # batch
            hues = hues.to(device) if hues is not None else None
            
            hidden_vec = cvae.get_hiddenvec(x)
            hidden_vec = hidden_vec.view(hidden_vec.shape[0], hidden_vec.shape[1], 1, 1)
            if reg_type == 'hue':
                optim_g.zero_grad()
                hue_pred_g = hue_reg_g(hidden_vec, c)
                loss_g = F.mse_loss(hue_pred_g.view(-1), hues.view(-1))
                loss_g.backward()
                optim_g.step()
            elif reg_type == 'digit':
                optim_g.zero_grad()
                digit_pred_g = digit_reg_g(hidden_vec, hues)
                loss_g = F.cross_entropy(digit_pred_g, c)

                # loss_g = F.mse_loss(digit_pred_g, nn.functional.one_hot(c, num_classes=10))
                loss_g.backward(retain_graph=True)
                optim_g.step()
            
            loss_g_first = loss_g.detach().clone()

            optim_other.zero_grad()
            pred, mu, logvar = cvae(x, c, hues)
            recon_loss, kld = loss_function(x, pred, mu, logvar)
            loss_cvae = recon_loss + beta * kld
            if reg_type == 'hue':
                hue_pred_h = hue_reg_h(hidden_vec, torch.zeros_like(c, device=device))
                hue_pred_g = hue_reg_g(hidden_vec, c)
                loss_h = F.mse_loss(hue_pred_h.view(-1), hues.view(-1))
                loss_g = F.mse_loss(hue_pred_g.view(-1), hues.view(-1))
            elif reg_type == 'digit':
                digit_pred_h = digit_reg_h(hidden_vec)
                digit_pred_g = digit_reg_g(hidden_vec, hues)
                loss_h = F.cross_entropy(digit_pred_h, c)
                loss_g = F.cross_entropy(digit_pred_g, c)
                # loss_g = F.mse_loss(digit_pred_g, nn.functional.one_hot(c, num_classes=10))
            loss = loss_cvae + lmda * (loss_h - loss_g)

            loss.backward()
            optim_other.step()


            # if i == 0:
            #     print("Gradients")
            #     for name,param in cvae.named_parameters():
            #         if "bias" in name:
            #             print(name,param.grad[0],end=" ")
            #         else:
            #             print(name,param.grad[0,0],end=" ")
            #         print()
            if total_loss_ema is None:
                total_loss_ema = loss.item()
                # reconstruction_loss_ema = recon_loss.item()
                # kld_loss_ema = kld.item()
            else:
                total_loss_ema = 0.95 * total_loss_ema + 0.05 * loss.item()
                # reconstruction_loss_ema = 0.95 * reconstruction_loss_ema + 0.05 * recon_loss.item()
                # kld_loss_ema = 0.95 * kld_loss_ema + 0.05 * kld.item()

            # Note That here the recon loss and kld loss shown are not ema (unlike in script_cvae.py)!!!
            pbar.set_description(f"loss: {total_loss_ema:.4f} g: {loss_g.item():.4f} h: {loss_h.item():.4f} cvae: {loss_cvae.item()} recon loss: {recon_loss.item():.4f} kld loss: {kld.item():.4f}")
        
        # test and generate sample image
        cvae.eval()
        with torch.no_grad():
            total_loss = 0
            reconstruction_loss = 0
            kld_loss = 0
            for i,(x,c) in enumerate(testloader):
                x, hues = add_hue_confounded(x, c, p_unif)
                x = x.to(device)
                c = c.to(device)
                hues = hues.to(device)
                pred, mu, logvar = cvae(x, c, hues)
                recon_loss, kld = loss_function(x,pred, mu, logvar)
                loss = recon_loss + kld

                total_loss += loss.cpu().data.numpy()*x.shape[0]
                reconstruction_loss += recon_loss.cpu().data.numpy()*x.shape[0]
                kld_loss += kld.cpu().data.numpy()*x.shape[0]
                # if i == 0:
                #     # print("gr:", x[0,0,:5,:5])
                #     # print("pred:", pred[0,0,:5,:5])
                #     plot(epoch, pred.cpu().data.numpy(), y.cpu().data.numpy())
            reconstruction_loss /= len(testloader.dataset)
            kld_loss /= len(testloader.dataset)
            total_loss /= len(testloader.dataset)
            print(f"test: total loss: {total_loss:.4f} recon loss: {reconstruction_loss:.4f} kld loss: {kld_loss:.4f}")

        
        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
            n_sample = 4*n_classes
            if ep%5==0 or ep == int(n_epoch-1):
                y = torch.arange(n_classes).repeat(n_sample//n_classes).to(device)
                z = torch.randn(n_sample, latent_size).to(device)
                x_gen = generate_image(ep, z, y, cvae, device=device)

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
                save_image(grid, save_dir + f"image/image_ep{ep}.png")
                print('saved image at ' + save_dir + f"image/image_ep{ep}.png")

        # optionally save model
        if save_model and (ep % 10 == 0 or ep == int(n_epoch-1)):
            torch.save(cvae.state_dict(), save_dir + f"model/model_{ep}.pth")
            print('saved model at ' + save_dir + f"model/model_{ep}.pth")

if __name__ == "__main__":
    main()

