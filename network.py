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
from typing import Dict, Tuple
import math

import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F


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

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x
        
class UnetDownAttention(nn.Module):
    '''
    process and downscale the image feature maps
    in: [batch, in_channels, H, W]
    temb: [batch, in_channels, 1, 1]
    mixemb: [batch, in_channels]
    out: [batch, out_channels, H/2, W/2]
    '''
    def __init__(self, in_channels, out_channels):
        super(UnetDownAttention, self).__init__()
        self.res = ResidualConvBlock(in_channels, out_channels)
        # 什么形状？？
        self.embfc = EmbedFC(in_channels, in_channels)
        self.attention = AttentionBlock(out_channels, encoder_channels=in_channels)
        self.downsample = nn.MaxPool2d(2)
    def forward(self, x, temb, mixemb):
        emb = temb.view(-1,temb.shape[1]) + mixemb
        emb = self.embfc(emb).view(-1, temb.shape[1], 1, 1)
        mixemb_mm = torch.einsum("ab,ac->abc", mixemb, mixemb)
        # diffscm在resblock里会再embed一次，这里仅简单相加
        h = self.res(x + emb)
        h = self.attention(h, mixemb_mm)
        h = self.downsample(h)
        return h

class UnetUpAttention(nn.Module):
    '''
    process and upscale the image feature maps
    in: [batch, in_channels/2, H, W]
    skip: [batch, in_channels/2, H, W]
    temb: [batch, in_channels/2, 1, 1]
    mixemb: [batch, in_channels/2]
    out: [batch, out_channels, H*2, W*2]
    '''
    def __init__(self, in_channels, out_channels):
        super(UnetUpAttention, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.res1 = ResidualConvBlock(out_channels, out_channels)
        self.res2 = ResidualConvBlock(out_channels, out_channels)
        self.embfc = EmbedFC(in_channels, out_channels)
        self.attention = AttentionBlock(out_channels, encoder_channels=in_channels//2)
    def forward(self, x, skip, temb, mixemb):
        emb = torch.cat((temb.view(-1, temb.shape[1]), mixemb),1)
        emb = self.embfc(emb)
        emb = emb.view(-1, emb.shape[1], 1, 1)
        mixemb_mm = torch.einsum("ab,ac->abc", mixemb, mixemb)
        h = torch.cat((x, skip), 1)
        h = self.upsample(h)
        h = self.attention(h, mixemb_mm)
        h = self.res1(h + emb)
        h = self.res2(h)
        return h

    
class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        encoder_channels=None,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)

        if encoder_channels is not None:
            self.encoder_kv = nn.Conv1d(encoder_channels, channels * 2, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)
        for p in self.proj_out.parameters():
            p.detach().zero_()

    def forward(self, x, encoder_out=None):
        """
        in: [batch, channels, H, W]
        encoder_out: [batch, channels, channels]
        out: [batch, channels, H, W]
        """
        b, c, *spatial = x.shape
        qkv = self.qkv(self.norm(x).view(b, c, -1))
        if encoder_out is not None:
            encoder_out_expand = self.encoder_kv(encoder_out)
            h = self.attention(qkv, encoder_out_expand)
        else:
            h = self.attention(qkv)
        h = self.proj_out(h)
        return x + h.reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, encoder_kv=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        if encoder_kv is not None:
            assert encoder_kv.shape[1] == self.n_heads * ch * 2
            ek, ev = encoder_kv.reshape(bs * self.n_heads, ch * 2, -1).split(ch, dim=1)
            k = torch.cat([ek, k], dim=-1)
            v = torch.cat([ev, v], dim=-1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

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


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_classes=10):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7), # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep, 
        # context_mask says which samples to block the context on

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # convert context to one hot embedding
        # [256] -> [256,10]
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
        
        # mask out context if context_mask == 1
        context_mask = context_mask[:, None]
        # q: what is context_mask[:, None]? 
        # a: it is a broadcasting operation, it adds a new dimension to the tensor
        # q: so the new shape is [256,1]?
        # a: yes, and it is broadcasted to [256,10] when multiplied with c
        context_mask = context_mask.repeat(1,self.n_classes)
        # debug
        # context_mask = (-1*(1-context_mask)) # need to flip 0 <-> 1
        context_mask = 1-context_mask
        c = c * context_mask
        # now c is [256,10] and some batches are masked to 0
        # a whole batch is masked, so it is also correctly masked after embed
        
        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        up1 = self.up0(hiddenvec)
        # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings

        up2 = self.up1(cemb1*up1+ temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2*up2+ temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out

class ContextUnetColored(ContextUnet):
    def __init__(self, in_channels, n_feat = 256, n_classes=10, cond_mode='AdaGN'):
        """
        cond_mode = "AdaGN", "Concat", "AdaCat", "Attention"
        AdaGN: mix context and hue embedding then use as gain in GroupNorm
        Concat: concat the context and hue embedding to the hidden vector
        AdaCat: concat the context embedding, use hue embedding as gain in GroupNorm
        Attention: use context embedding as attention to hidden vector
        """
        super(ContextUnetColored, self).__init__(in_channels, n_feat, n_classes)
        self.hueembed1 = EmbedFC(1, 2*n_feat)
        self.hueembed2 = EmbedFC(1, 1*n_feat)

        self.mixembed1 = EmbedFC(4*n_feat, 2*n_feat)
        self.mixembed2 = EmbedFC(2*n_feat, 1*n_feat)

        self.huenullembed1 = nn.Parameter(torch.zeros(2*n_feat)).view(-1,1,1)
        self.huenullembed2 = nn.Parameter(torch.zeros(1*n_feat)).view(-1,1,1)

        self.cond_mode = cond_mode

        if cond_mode == "Concat":
            raise NotImplementedError("Concat mode not implemented yet")

        if cond_mode == "Attention":
            self.down1 = UnetDownAttention(n_feat, n_feat)
            self.down2 = UnetDownAttention(n_feat, 2 * n_feat)
            self.up1 = UnetUpAttention(4 * n_feat, n_feat)
            self.up2 = UnetUpAttention(2 * n_feat, n_feat)

        
        if cond_mode == "AdaCat":
            self.up0 = nn.Sequential(
                nn.ConvTranspose2d(4 * n_feat, 2 * n_feat, 7, 7), # when concat cemb end up w 4*n_feat
                nn.GroupNorm(8, 2 * n_feat),
                nn.ReLU(),
            )


    def forward(self, x, c, hue, t, context_mask, hue_mask):
        # x is (noisy) image, c is context label, t is timestep, 
        # context_mask says which samples to block the context on
        # hue is the color to add to the image [0,1]
                # convert context to one hot embedding
        # [256] -> [256,10]
        if c.dim() == 1:
            c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)

        
        # mask out context if context_mask == 1
        context_mask = context_mask[:, None]
        # q: what is context_mask[:, None]? 
        # a: it is a broadcasting operation, it adds a new dimension to the tensor
        # q: so the new shape is [256,1]?
        # a: yes, and it is broadcasted to [256,10] when multiplied with c
        context_mask = context_mask.repeat(1,self.n_classes)
        # debug
        # context_mask = (-1*(1-context_mask)) # need to flip 0 <-> 1
        context_mask = 1-context_mask
        c = c * context_mask
        # now c is [256,10] and some batches are masked to 0
        # a whole batch is masked, so it is also correctly masked after embed

        # # mask out hue if hue_mask == 1 and fill the hue with 2
        # hue = hue * (1-hue_mask) + 2*hue_mask
        
        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
        hemb1 = self.hueembed1(hue).view(-1, self.n_feat * 2, 1, 1)
        hemb2 = self.hueembed2(hue).view(-1, self.n_feat, 1, 1)

        # mask out hemb if hue_mask == 1 and fill with huenullembed
        for i in range(hue_mask.shape[0]):
            if hue_mask[i]:
                hemb1[i] = self.huenullembed1
                hemb2[i] = self.huenullembed2

        if self.cond_mode == "Attention":
            mixemb1 = self.mixembed1(torch.cat((cemb1, hemb1), 1))
            mixemb2 = self.mixembed2(torch.cat((cemb2, hemb2), 1))
            x = self.init_conv(x)
            down1 = self.down1(x, temb2, mixemb2)
            down2 = self.down2(down1, temb2, mixemb2)
            hiddenvec = self.to_vec(down2)
            up1 = self.up0(hiddenvec)
            up2 = self.up1(up1, down2, temb1, mixemb1)
            up3 = self.up2(up2, down1, temb2, mixemb2)
        else:
            x = self.init_conv(x)
            down1 = self.down1(x)
            down2 = self.down2(down1)
            hiddenvec = self.to_vec(down2)

            # debug
            # print(f'c {c.shape} hue_mask {hue_mask.shape}, hue {hue.shape}')
            # print(f'cemb1 {cemb1.shape} temb1 {temb1.shape} hemb1 {hemb1.shape}')
            if self.cond_mode == "AdaGN":
                mixemb1 = self.mixembed1(torch.cat((cemb1, hemb1), 1)).view(-1, self.n_feat * 2, 1, 1)
                mixemb2 = self.mixembed2(torch.cat((cemb2, hemb2), 1)).view(-1, self.n_feat * 1, 1, 1)

                # could concatenate the context embedding here instead of adaGN
                # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

                up1 = self.up0(hiddenvec)
                # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings

                up2 = self.up1(mixemb1*up1+ temb1, down2)  # add and multiply embeddings
                up3 = self.up2(mixemb2*up2+ temb2, down1)
                
            elif self.cond_mode == "AdaCat":
                hiddenvec = torch.cat((hiddenvec, cemb1), 1)
                up1 = self.up0(hiddenvec)
                up2 = self.up1(hemb1*up1+ temb1, down2)  # add and multiply embeddings
                up3 = self.up2(hemb2*up2+ temb2, down1)

        out = self.out(torch.cat((up3, x), 1))
        return out

    def encode(self, x, t=None):
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)
        return hiddenvec

    def decode(self, x, hiddenvec, c, hue, t, context_mask, hue_mask):
                # x is (noisy) image, c is context label, t is timestep, 
        # context_mask says which samples to block the context on
        # hue is the color to add to the image [0,1]
                # convert context to one hot embedding
        # [256] -> [256,10]
        if c.dim() == 1:
            c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)

        # mask out context if context_mask == 1
        context_mask = context_mask[:, None]
        # q: what is context_mask[:, None]? 
        # a: it is a broadcasting operation, it adds a new dimension to the tensor
        # q: so the new shape is [256,1]?
        # a: yes, and it is broadcasted to [256,10] when multiplied with c
        context_mask = context_mask.repeat(1,self.n_classes)
        # debug
        # context_mask = (-1*(1-context_mask)) # need to flip 0 <-> 1
        context_mask = 1-context_mask
        c = c * context_mask
        # now c is [256,10] and some batches are masked to 0
        # a whole batch is masked, so it is also correctly masked after embed

        # # mask out hue if hue_mask == 1 and fill the hue with 2
        # hue = hue * (1-hue_mask) + 2*hue_mask
        
        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
        hemb1 = self.hueembed1(hue).view(-1, self.n_feat * 2, 1, 1)
        hemb2 = self.hueembed2(hue).view(-1, self.n_feat, 1, 1)

        # mask out hemb if hue_mask == 1 and fill with huenullembed
        for i in range(hue_mask.shape[0]):
            if hue_mask[i]:
                hemb1[i] = self.huenullembed1
                hemb2[i] = self.huenullembed2

        if self.cond_mode == "Attention":
            mixemb1 = self.mixembed1(torch.cat((cemb1, hemb1), 1))
            mixemb2 = self.mixembed2(torch.cat((cemb2, hemb2), 1))
            up1 = self.up0(hiddenvec)
            up2 = self.up1(up1, down2, temb1, mixemb1)
            up3 = self.up2(up2, down1, temb2, mixemb2)
        else:
            # debug
            # print(f'c {c.shape} hue_mask {hue_mask.shape}, hue {hue.shape}')
            # print(f'cemb1 {cemb1.shape} temb1 {temb1.shape} hemb1 {hemb1.shape}')
            if self.cond_mode == "AdaGN":
                mixemb1 = self.mixembed1(torch.cat((cemb1, hemb1), 1)).view(-1, self.n_feat * 2, 1, 1)
                mixemb2 = self.mixembed2(torch.cat((cemb2, hemb2), 1)).view(-1, self.n_feat * 1, 1, 1)

                # could concatenate the context embedding here instead of adaGN
                # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

                up1 = self.up0(hiddenvec)
                # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings

                up2 = self.up1(mixemb1*up1+ temb1, down2)  # add and multiply embeddings
                up3 = self.up2(mixemb2*up2+ temb2, down1)
                
        out = self.out(torch.cat((up3, x), 1))
        return out



def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    ma_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": ma_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1, color=False, independent_mask=False):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()
        self.color = color
        self.independent_mask = independent_mask

    def forward(self, x, c, hue=None):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros(c.shape[0])+self.drop_prob).to(self.device)
        if self.independent_mask:
            hue_mask = torch.bernoulli(torch.zeros_like(hue)+self.drop_prob).to(self.device) if self.color else None
        else:
            hue_mask = context_mask if self.color else None
        # return MSE between added noise, and our predicted noise
        if self.color:
            return self.loss_mse(noise, self.nn_model(x_t, c, hue, _ts / self.n_T, context_mask, hue_mask))
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device, guide_w=0.0, ddim=False):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

       
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        c_i = torch.arange(0,10).to(device) # context for us just cycles throught the mnist labels
        c_i = c_i.repeat(int(n_sample/c_i.shape[0]))
        
        hues_i = None
        if self.color:
            hues_i = (torch.arange(0,10)+0.5).to(device) * 0.1 % 1.0
            hues_i = hues_i.repeat(int(n_sample/hues_i.shape[0]))
        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(device)

        # double the batch
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1. # makes second half of batch context free
        hues_i = hues_i.repeat(2) if self.color else None

        x_i_store = [] # keep track of generated steps in case want to plot something 
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            # double batch
            x_i = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            if self.color:
                eps = self.nn_model(x_i, c_i, hues_i, t_is, context_mask, hue_mask=context_mask)
            else:
                eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2

            x_i = x_i[:n_sample]

            if ddim:
                x_i = (
                    self.oneover_sqrta[i] * x_i  + (
                    self.sqrtmab[i-1] - self.oneover_sqrta[i] * self.sqrtmab[i]) * eps
                )
            else:
                x_i = (
                    self.oneover_sqrta[i] * x_i  - (
                    self.oneover_sqrta[i] * self.mab_over_sqrtmab[i]) * eps
                    + self.sqrt_beta_t[i] * self.sqrtmab[i - 1]/self.sqrtmab[i] * z
                )

            # if ddim:
            #     c1 = 0
            # else:
            #     c1 = self.sqrt_beta_t[i] * self.sqrtmab[i - 1]/self.sqrtmab[i]
            # c2 = torch.sqrt((1 - self.alphabar_t[i - 1]) - c1 ** 2)

            # # x0_i = (x_i - eps * self.sqrtmab[i]) / self.sqrtab[i]
            # # x_i = self.sqrtab[i-1] * x0_i + c1 * z + c2 * eps
            # coef_eps_neg_ddim = c2 - self.oneover_sqrta[i] * self.sqrtmab[i]
            # # coef_eps_neg_ddpm = self.oneover_sqrta[i] * self.mab_over_sqrtmab[i]

            # # debug
            # # print(f"coef_eps_neg_ddim {coef_eps_neg_ddim} coef_eps_neg_ddpm {coef_eps_neg_ddpm}")
            # x_i_next = (
            #     self.oneover_sqrta[i] * x_i  + (
            #     coef_eps_neg_ddim) * eps
            #     + c1 * z
            # )

            # # debug
            # # x_i_ddpm = (
            # #     self.oneover_sqrta[i] * x_i - (
            # #     coef_eps_neg_ddpm) * eps 
            # #     + c1 * z
            # # )
            # # print(f"x_i_ddim {x_i_next[0][0][0][:10]} \nx_i_ddpm {x_i_ddpm[0][0][0][:10]} \ndiff {(x_i_next-x_i_ddpm)[0][0][0][:10]}")

            # x_i = x_i_next.detach()
            # # else:
            # #     x_i = (
            # #         self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
            # #         + self.sqrt_beta_t[i] * self.sqrtmab[i - 1]/self.sqrtmab[i] * z
            # #     )

            x_i = x_i.detach()
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store
    
    def abduct(self, x, c, size=(1, 28, 28), device='cuda', guide_w=0.0, hues=None, hue_uncond=False, c_uncond=False):
        """
        x in shape [batch, 1, 28, 28], c in shape [batch]
        """
        x_i = x.view(-1, *size).to(device)
        n_sample = x_i.shape[0]
        # don't drop context at test time
        context_mask = torch.zeros_like(c).to(device)

        # double the batch
        c_i = c.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1. # makes second half of batch context free
        hue_mask = context_mask.clone()
        if c_uncond:
            context_mask[0:n_sample] = 1.
        hues_i = hues.repeat(2) if self.color else None

        if hue_uncond:
            hue_mask[0:n_sample] = 1.

        x_i_store = [] # keep track of generated steps in case want to plot something 
        print()
        # for i in range(self.n_T, 0, -1):
        for i in range(0, self.n_T):
            # print(f'sampling timestep {i}',end='\r')
            # print(f'abducting timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            # double batch
            x_i = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)

            # split predictions and compute weighting
            # eps = self.nn_model(x_i, c_i, t_is, context_mask)
            if self.color:
                eps = self.nn_model(x_i, c_i, hues_i, t_is, context_mask, hue_mask)
            else:
                eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2

            x_i = x_i[:n_sample]

            # x_i = (
            #     self.oneover_sqrta[i] * x_i  + (
            #     self.sqrtmab[i+1] - self.oneover_sqrta[i] * self.sqrtmab[i]) * eps
            # )

            x_i = (
                self.alpha_t[i+1].sqrt() * x_i  + (
                self.sqrtmab[i+1] - self.alpha_t[i+1].sqrt() * self.sqrtmab[i]) * eps
            )

            x_i = x_i.detach()
        return x_i
    
    def reconstruct(self, u, c, size=(1, 28, 28), device='cuda', guide_w=0.0, hues=None, hue_uncond=False, c_uncond=False):
        # reconstruct a sample from noise u
        # similar to self.sample but given input noise

        x_i = u.view(-1, *size).to(device)
        n_sample = x_i.shape[0]

        context_mask = torch.zeros(c.shape[0]).to(device)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1.
        hue_mask = context_mask.clone()
        if c_uncond:
            context_mask[0:n_sample] = 1.
        if hue_uncond:
            hue_mask[0:n_sample] = 1.
        c_i = c.repeat(2) if c.dim() == 1 else c.repeat(2,1)
        hues_i = hues.repeat(2) if self.color else None

        for i in range(self.n_T, 0, -1):
            # print(f'reconstructing timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            # double batch
            x_i = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)

            # split predictions and compute weighting
            # eps = self.nn_model(x_i, c_i, t_is, context_mask)
            if self.color:
                eps = self.nn_model(x_i, c_i, hues_i, t_is, context_mask, hue_mask)
            else:
                eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2

            x_i = x_i[:n_sample]

            x_i = (
                self.oneover_sqrta[i] * x_i  + (
                self.sqrtmab[i-1] - self.oneover_sqrta[i] * self.sqrtmab[i]) * eps
            )

            x_i = x_i.detach()

        return x_i

class HueRegressor(nn.Module):
    def __init__(self, n_classes=10, n_feat=256):
        super(HueRegressor, self).__init__()
        self.yemb = EmbedFC(n_classes, n_feat)
        self.fc1 = nn.Linear(2*n_feat, 2*n_feat)
        self.fc2 = nn.Linear(2*n_feat, n_feat)
        self.fc3 = nn.Linear(n_feat, 1)

    def forward(self, x, y):
        yemb = self.yemb(y).view(-1, 2*n_feat, 1, 1)
        x = torch.cat((x, yemb), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Classifier(nn.Module):
    # simple CNN for classificaiton
    def __init__(self):
        super(Classifier, self).__init__()
        
        # define a conv layer with output channels as 16, kernel size of 3 and stride of 1
        self.conv11 = nn.Conv2d(3, 16, 3, 1) # Input = 1x28x28  Output = 16x26x26
        self.conv12 = nn.Conv2d(3, 16, 5, 1) # Input = 1x28x28  Output = 16x24x24
        self.conv13 = nn.Conv2d(3, 16, 7, 1) # Input = 1x28x28  Output = 16x22x22
        self.conv14 = nn.Conv2d(3, 16, 9, 1) # Input = 1x28x28  Output = 16x20x20

        # define a conv layer with output channels as 32, kernel size of 3 and stride of 1
        self.conv21 = nn.Conv2d(16, 32, 3, 1) # Input = 16x26x26 Output = 32x24x24
        self.conv22 = nn.Conv2d(16, 32, 5, 1) # Input = 16x24x24 Output = 32x20x20
        self.conv23 = nn.Conv2d(16, 32, 7, 1) # Input = 16x22x22 Output = 32x16x16
        self.conv24 = nn.Conv2d(16, 32, 9, 1) # Input = 16x20x20  Output = 32x12x12

        # define a conv layer with output channels as 64, kernel size of 3 and stride of 1
        self.conv31 = nn.Conv2d(32, 64, 3, 1) # Input = 32x24x24 Output = 64x22x22
        self.conv32 = nn.Conv2d(32, 64, 5, 1) # Input = 32x20x20 Output = 64x16x16
        self.conv33 = nn.Conv2d(32, 64, 7, 1) # Input = 32x16x16 Output = 64x10x10
        self.conv34 = nn.Conv2d(32, 64, 9, 1) # Input = 32x12x12 Output = 64x4x4
        

        # define a max pooling layer with kernel size 2
        self.maxpool = nn.MaxPool2d(2) # Output = 64x11x11
        #self.maxpool1 = nn.MaxPool2d(1)
        # define dropout layer with a probability of 0.25
        self.dropout1 = nn.Dropout(0.25)
        # define dropout layer with a probability of 0.5
        self.dropout2 = nn.Dropout(0.5)

        # define a linear(dense) layer with 128 output features
        self.fc11 = nn.Linear(64*11*11, 256)
        self.fc12 = nn.Linear(64*8*8, 256)      # after maxpooling 2x2
        self.fc13 = nn.Linear(64*5*5, 256)
        self.fc14 = nn.Linear(64*2*2, 256)

        # define a linear(dense) layer with output features corresponding to the number of classes in the dataset
        self.fc21 = nn.Linear(256, 128)
        self.fc22 = nn.Linear(256, 128)
        self.fc23 = nn.Linear(256, 128)
        self.fc24 = nn.Linear(256, 128)

        self.fc33 = nn.Linear(128*4,10)
        #self.fc33 = nn.Linear(64*3,10)
        

    def forward(self, inp):
        # Use the layers defined above in a sequential way (folow the same as the layer definitions above) and 
        # write the forward pass, after each of conv1, conv2, conv3 and fc1 use a relu activation. 
        

        x = F.relu(self.conv11(inp))
        x = F.relu(self.conv21(x))
        x = F.relu(self.maxpool(self.conv31(x)))
        #print(x.shape)
        #x = torch.flatten(x, 1)
        x = x.view(-1,64*11*11)
        x = self.dropout1(x)
        x = F.relu(self.fc11(x))
        x = self.dropout2(x)
        x = self.fc21(x)

        y = F.relu(self.conv12(inp))
        y = F.relu(self.conv22(y))
        y = F.relu(self.maxpool(self.conv32(y)))
        #x = torch.flatten(x, 1)
        y = y.view(-1,64*8*8)
        y = self.dropout1(y)
        y = F.relu(self.fc12(y))
        y = self.dropout2(y)
        y = self.fc22(y)

        z = F.relu(self.conv13(inp))
        z = F.relu(self.conv23(z))
        z = F.relu(self.maxpool(self.conv33(z)))
        #x = torch.flatten(x, 1)
        z = z.view(-1,64*5*5)
        z = self.dropout1(z)
        z = F.relu(self.fc13(z))
        z = self.dropout2(z)
        z = self.fc23(z)

        ze = F.relu(self.conv14(inp))
        ze = F.relu(self.conv24(ze))
        ze = F.relu(self.maxpool(self.conv34(ze)))
        #x = torch.flatten(x, 1)
        ze = ze.view(-1,64*2*2)
        ze = self.dropout1(ze)
        ze = F.relu(self.fc14(ze))
        ze = self.dropout2(ze)
        ze = self.fc24(ze)

        out_f = torch.cat((x, y, z, ze), dim=1)
        #out_f1 = torch.cat((out_f, ze), dim=1)
        out = self.fc33(out_f)
        
        output = F.log_softmax(out, dim=1)
        return output