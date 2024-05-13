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

from cvae import CVAE, plot, loss_function, generate_image


def main():
    parser = argparse.ArgumentParser(description='Train a model on MNIST with configurable parameters.')
    parser.add_argument('--date', type=str, help='Date for saving experiments')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--p_unif', type=float, default=None, help='Used in generating training data, NOT in classified data')
    parser.add_argument('--beta', type=float, default=1, help='Beta value for KL divergence')
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
    save_model = True
    save_dir = f'./experiments/{args.date}/'

    assert p_unif is not None or classifier_name is not None, \
        "at least one of p_unif and classifier_name should be set to indicate"

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir + "image", exist_ok=True)
    os.makedirs(save_dir + "gif", exist_ok=True)
    os.makedirs(save_dir + "model", exist_ok=True)

    cvae = CVAE(n_feat, latent_size, n_classes).to(device)
    optim = torch.optim.AdamW(cvae.parameters(), lr=lrate)


    # optionally load a model
    # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))

    tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1
    if classifier_name is None:
        dataset = MNIST("./data", train=True, download=True, transform=tf)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)


    for ep in range(n_epoch):
        print(f'epoch {ep}')
        cvae.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)
        if classifier_name is not None:
            pbar = tqdm(range(60000//batch_size))
        else:
            pbar = tqdm(dataloader)
        total_loss_ema = None
        reconstruction_loss_ema = None
        kld_loss_ema = None
        for val in pbar:
            x, c = val
            hues = None
            if mnist_color:
                x, hues = add_hue_confounded(x, c, p_unif)

            optim.zero_grad()
            x = x.to(device)    # batch, 3, 28, 28
            c = c.to(device)    # batch
            hues = hues.to(device) if hues is not None else None
            pred, mu, logvar = cvae(x, c, hues)
            recon_loss, kld = loss_function(x, pred, mu, logvar)
            loss = recon_loss + beta * kld
            loss.backward()
            optim.step()

            total_loss += loss.cpu().data.numpy()*x.shape[0]
            reconstruction_loss += recon_loss.cpu().data.numpy()*x.shape[0]
            kld_loss += kld.cpu().data.numpy()*x.shape[0]
            if i == 0:
                print("Gradients")
                for name,param in model.named_parameters():
                    if "bias" in name:
                        print(name,param.grad[0],end=" ")
                    else:
                        print(name,param.grad[0,0],end=" ")
                    print()
            if total_loss_ema is None:
                total_loss_ema = loss.item()
                reconstruction_loss_ema = recon_loss.item()
                kld_loss_ema = kld.item()
            else:
                total_loss_ema = 0.95 * total_loss_ema + 0.05 * loss.item()
                reconstruction_loss_ema = 0.95 * reconstruction_loss_ema + 0.05 * recon_loss.item()
                kld_loss_ema = 0.95 * kld_loss_ema + 0.05 * kld.item()

            pbar.set_description(f"total loss: {total_loss_ema:.4f} recon loss: {reconstruction_loss_ema:.4f} kld loss: {kld_loss_ema:.4f}")
        
        # test and generate sample image
        cvae.eval()
        with torch.no_grad():
            for i,(x,y) in enumerate(test_loader):
                x, hues = add_hue_confounded(x, c, p_unif)
                x = x.to(device)
                c = c.to(device)
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
            reconstruction_loss /= len(test_loader.dataset)
            kld_loss /= len(test_loader.dataset)
            total_loss /= len(test_loader.dataset)
            print(f"test: total loss: {total_loss:.4f} recon loss: {reconstruction_loss:.4f} kld loss: {kld_loss:.4f}")

        
        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
            n_sample = 4*n_classes
            if ep%5==0 or ep == int(n_epoch-1):
                y = torch.arange(n_classes).repeat(n_sample//n_classes)
                z = torch.randn(n_sample, latent_size).to(device)
                x_gen = generate_image(ep, z, y, cvae)

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

