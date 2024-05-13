import torch 
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn. functional as F
import torch.optim as optim
import os
import sys

class ResidualConvBlock(nn.Module):
    '''
    standard ResNet style convolutional block
    in: [batch, in_channels, H, W]
    out: [batch, out_channels, H, W]
    '''
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    '''
    process and downscale the image feature maps
    in: [batch, in_channels, H, W]
    out: [batch, out_channels, H/2, W/2]
    '''
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UnetUp(nn.Module):
    '''
    process and upscale the image feature maps
    in: [batch, in_channels, H, W]
    out: [batch, out_channels, H*2, W*2]
    '''
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x

class EmbedFC(nn.Module):
    '''
    generic one layer FC NN for embedding things
    in: [batch, input_dim]
    out: [batch, emb_dim]  
    '''
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class CVAE(nn.Module):
    def __init__(self, in_channels=3, n_feat=256, latent_size=32, num_classes=10, device="cuda"):
        super(CVAE,self).__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.device = device

        # For encode
        # self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=2)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        # self.linear1 = nn.Linear(4*4*32,300)
        self.mu = nn.Linear(2 * n_feat, self.latent_size)
        self.logvar = nn.Linear(2 * n_feat, self.latent_size)

        # For decoder
        # self.linear2 = nn.Linear(self.latent_size + self.num_classes, 300)
        # self.linear3 = nn.Linear(300,4*4*32)
        # self.conv3 = nn.ConvTranspose2d(32, 16, kernel_size=5,stride=2)
        # self.conv4 = nn.ConvTranspose2d(16, 1, kernel_size=5, stride=2)
        # self.conv5 = nn.ConvTranspose2d(1, 1, kernel_size=4)

        self.linear = nn.Linear(self.latent_size, 2 * n_feat)
        self.contextembed1 = EmbedFC(num_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(num_classes, 1*n_feat)
        self.hueembed1 = EmbedFC(1, 2*n_feat)
        self.hueembed2 = EmbedFC(1, 1*n_feat)

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7), # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(2 * n_feat, n_feat)
        self.up2 = UnetUp(n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def encoder(self,x):

        # y = torch.ones(x.shape).to(device)*y
        # t = torch.cat((x,y),dim=1)

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2).view(x.shape[0], -1)
        mu = self.mu(hiddenvec)
        logvar = self.logvar(hiddenvec)
        return mu, logvar, down1, down2
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std).to(self.device)
        return eps*std + mu
    
    # def unFlatten(self, x):
    #     return x.reshape((x.shape[0], 32, 4, 4))

    def decoder(self, latent, c, hue):
        # t = F.relu(self.linear2(z))
        # t = F.relu(self.linear3(t))
        # t = self.unFlatten(t)
        # t = F.relu(self.conv3(t))
        # t = F.relu(self.conv4(t))
        # t = F.relu(self.conv5(t))
        # return t
                      
        # convert context to one hot embedding
        # [256] -> [256,10]
        if c.dim() == 1:
            c = nn.functional.one_hot(c, num_classes=self.num_classes).type(torch.float)
        
        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        hemb1 = self.hueembed1(hue).view(-1, self.n_feat * 2, 1, 1)
        hemb2 = self.hueembed2(hue).view(-1, self.n_feat, 1, 1)

        latent = self.linear(latent)  # [batch, latent_size] -> [batch, 2*n_feat]
        latent = latent.view(latent.shape[0], latent.shape[1], 1, 1)
        up1 = self.up0(latent)           # [batch, 2*n_feat] -> [batch, 2*n_feat, 7, 7]
        # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings

        up2 = self.up1(cemb1*up1+ hemb1)  # add and multiply embeddings
        up3 = self.up2(cemb2*up2+ hemb2)
                
        out = self.out(up3)
        return out

    def forward(self, x, c, hue):
        mu, logvar, down1, down2 = self.encoder(x)
        z = self.reparameterize(mu,logvar)

        # Class and hue conditioning
        # z = torch.cat((z, y.float()), dim=1)
        pred = self.decoder(z, c, hue)
        return pred, mu, logvar


def plot(epoch, pred, y,name='test_'):
    if not os.path.isdir('./images'):
        os.mkdir('./images')
    fig = plt.figure(figsize=(16,16))
    for i in range(6):
        ax = fig.add_subplot(3,2,i+1)
        ax.imshow(pred[i,0])
        ax.axis('off')
        ax.title.set_text(str(y[i]))
    plt.savefig("./images/{}epoch_{}.jpg".format(name, epoch))
    # plt.figure(figsize=(10,10))
    # plt.imsave("./images/pred_{}.jpg".format(epoch), pred[0,0], cmap='gray')
    plt.close()


def loss_function(x, pred, mu, logvar):
    recon_loss = F.mse_loss(pred, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss, kld

def generate_image(epoch, z, y, model, device="cuda"):
    with torch.no_grad():
        hue = (y+0.5) * 0.1 % 1.0
        pred = model.decoder(z, y, hue)
    return pred

    