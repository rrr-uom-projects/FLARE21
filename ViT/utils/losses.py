"""
Custom losses - copied from Ed's folder
Easier w.o dealing with relative path import 
"""
import torch
import torch.nn.functional as F
import numpy as np

def exp_log_loss(prediction, mask, ignore_index, device='cuda'):
    """
    paper: 3D Segmentation with Exponential Logarithmic Loss for Highly Unbalanced Object Sizes
    https://arxiv.org/pdf/1809.00076.pdf
    
    ### ! needs raw scores from the network ! ###
    """
    gamma = 0.3
    # Dice loss
    num_classes = prediction.size()[1]
    smooth = 1.
    dice_pred = F.softmax(prediction, dim=1)
    if (ignore_index == False).any():
        # we are missing gold standard masks for some structures
        # change all predicted pixels of the missing structure to background -> 0
        # that way the loss will be 0 in regions of missing gold standard labels
        ablation_mask = torch.zeros_like(dice_pred, dtype=bool)
        missing_inds = torch.where(~ignore_index)
        for imdx, sdx in zip(missing_inds[0], missing_inds[1]):
            ablation_mask[imdx][dice_pred.clone().detach()[imdx] == sdx] = True
        dice_pred = dice_pred.masked_fill(ablation_mask, 0)

    pred_flat = dice_pred.view(-1, num_classes)
    mask_flat = mask.view(-1, num_classes)
    intersection = (pred_flat*mask_flat).sum(dim=0)
    # numerator
    num = 2. * intersection + smooth
    # denominator
    denom = pred_flat.sum(dim=0) + mask_flat.sum(dim=0) + smooth
    # calculate dice
    dice = num / denom
    dice_loss = torch.mean(
        torch.pow(torch.clamp(-torch.log(dice), min=1e-6), gamma))

    # XE loss
    # background, liver, kidney L, kidney R, spleen, pancreas
    label_freq = np.array(
        [358074923, 14152955, 1698684, 1643118,  2153875, 812381])

    class_weights = np.power(
        np.full((num_classes), label_freq.sum()) / label_freq, 0.5)
    xe_pred = F.log_softmax(prediction, dim=1)
    if (ignore_index == False).any():
        # same again - missing inds retained from above
        ablation_mask = torch.zeros_like(xe_pred, dtype=bool)
        for imdx, sdx in zip(missing_inds[0], missing_inds[1]):
            ablation_mask[imdx][xe_pred.clone().detach()[imdx] == sdx] = True
        xe_pred = xe_pred.masked_fill(ablation_mask, 0)
    mask = torch.argmax(mask, dim=4)
    xe_loss = torch.mean(torch.pow(torch.clamp(torch.nn.NLLLoss(weight=torch.FloatTensor(
        class_weights).to(device), reduction='none')(xe_pred, mask), min=1e-6), gamma))

    w_dice = 0.5
    w_xe = 0.5
    return (w_dice*dice_loss) + (w_xe*xe_loss)


class RunningAverage:
    # Computes and stores the average
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count
