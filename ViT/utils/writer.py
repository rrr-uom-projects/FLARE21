"""
Custom Writer to monitor training
"""
from os import stat
import matplotlib
matplotlib.use('Agg')
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from scipy.ndimage import center_of_mass


class customWriter(SummaryWriter):
    def __init__(self, batch_size):
        super().__init__()
        self.epoch = 0
        self.batch_size = batch_size

    def reset_losses(self):
        self.metrics = {key: [] for key in self.metrics.keys()}

    @staticmethod
    def sigmoid(x):
        return 1/(1+torch.exp(-x))

    @staticmethod
    def norm_img(img):
        return (img-img.min())/(img.max()-img.min())

    @staticmethod
    def find_coords(mask, sdx, sdx2=None):
            #@ Get COM for plotting 3D vol. as 2D image
            if sdx2:
                coords = center_of_mass(np.logical_or(mask == sdx, mask == sdx2))
            else:
                coords = center_of_mass(mask == sdx)
            return np.round(coords).astype(int)

    def plot_inputs(self, title, img, targets=None):
        fig = plt.figure(figsize=(10, 10))
        plt.tight_layout()
        for idx in np.arange(self.batch_size):
            ax = fig.add_subplot(self.batch_size // 2, self.batch_size // 2,
                                 idx+1, label='Inputs')
            plt_img = rearrange(img[idx].cpu().numpy(), 'c h w -> h w c')
            plt_img = self.norm_img(plt_img)
            ax.imshow(plt_img, cmap='gray')
            if targets is not None:
                pass
                


    
