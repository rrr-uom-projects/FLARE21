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


class customWriter(SummaryWriter):
    def __init__(self, batch_size):
        super().__init__()
        self.epoch = 0
        self.batch_size = batch_size
        self.metrics = {'train_loss': [], 'val_loss':[]}

    def reset_losses(self):
        self.metrics = {key: [] for key in self.metrics.keys()}

    @staticmethod
    def sigmoid(x):
        return 1/(1+torch.exp(-x))

    def plot_inputs(self, title, img):
        #TODO
        pass