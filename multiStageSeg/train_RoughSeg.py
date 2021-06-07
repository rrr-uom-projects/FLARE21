# train_headHunter_bits.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import argparse as ap

from models import roughSegmenter
from Trainers import roughSegmenter_trainer
from utils import k_fold_split_train_val_test, get_logger, get_number_of_learnable_parameters, getFiles, windowLevelNormalize

imagedir = "/data/FLARE21/training_data/scaled_ims/"
maskdir = "/data/FLARE21/training_data/scaled_masks/"

def setup_argparse():
    parser = ap.ArgumentParser(prog="Main training program for 3D location-finding network \"headhunter\"")
    parser.add_argument("--fold_num", choices=[1,2,3,4,5], type=int, help="The fold number for the kfold cross validation")
    parser.add_argument("--GPUs", choices=[1,2], type=int, default=1, help="Number of GPUs to use")
    global args
    args = parser.parse_args()

def main():
    # get args
    setup_argparse()
    global args

    # decide checkpoint directory
    checkpoint_dir = "/data/FLARE21/models/roughSegmenter/fold"+str(args.fold_num)
    # Create main logger
    logger = get_logger('organHunter_Training')

    # Create the model
    model = roughSegmenter(filter_factor=2, n_classes=6, in_channels=1, p_drop=0)

    for param in model.parameters():
        param.requires_grad = True

    # put the model on GPU(s)
    device='cuda'
    model.to(device)

    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')
    
    train_BS = int(4 * args.GPUs)
    val_BS = int(3 * args.GPUs)
    train_workers = int(4)
    val_workers = int(3)

    # allocate ims to train, val and test
    dataset_size = 92 #len(sorted(getFiles(imagedir)))
    train_inds, val_inds, test_inds = k_fold_split_train_val_test(dataset_size, fold_num=args.fold_num, seed=230597)

    # Create them dataloaders
    train_data = roughSegmenter_Dataset(imagedir=imagedir, maskdir=maskdir, image_inds=train_inds, shift_augment=True, flip_augment=False)
    train_loader = DataLoader(dataset=train_data, batch_size=train_BS, shuffle=True, pin_memory=True, num_workers=train_workers, worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))
    val_data = roughSegmenter_Dataset(imagedir=imagedir, maskdir=maskdir, image_inds=val_inds, shift_augment=False, flip_augment=False)
    val_loader = DataLoader(dataset=val_data, batch_size=val_BS, shuffle=True, pin_memory=True, num_workers=val_workers, worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))

    # Create the optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.005)

    # Create learning rate adjustment strategy
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=175, verbose=True)
    
    # Create model trainer
    trainer = roughSegmenter_trainer(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, device=device, train_loader=train_loader, 
                                    val_loader=val_loader, logger=logger, checkpoint_dir=checkpoint_dir, max_num_epochs=1000, patience=500, iters_to_accumulate=1)
    
    # Start training
    trainer.fit()

    # Romeo Dunn
    return
    
class roughSegmenter_Dataset(data.Dataset):
    def __init__(self, imagedir, maskdir, image_inds, shift_augment=True, flip_augment=False):
        self.imagedir = imagedir
        self.maskdir = maskdir
        self.availableImages = [sorted(getFiles(imagedir))[ind] for ind in image_inds]
        self.shifts = shift_augment
        self.flips = flip_augment
        self.rotations = False
        self.scaling = False
        self.ignore_oars = np.load("/data/FLARE21/training_data/labels_present.npy")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
           idx = idx.tolist()
        imageToUse = self.availableImages[idx]
        #spacing = np.load("/data/FLARE21/training_data/spacings_scaled.npy")[idx][[2,0,1]]
        ct_im = np.load(os.path.join(self.imagedir, imageToUse))
        mask = np.load(os.path.join(self.maskdir, imageToUse))
        ignore_index = self.ignore_oars[idx]
        
        # Augmentations
        if self.shifts:
            # find shift values
            cc_shift, ap_shift, lr_shift = random.randint(-2,2), random.randint(-4,4), random.randint(-4,4)
            # pad for shifting into
            ct_im = np.pad(ct_im, pad_width=((2,2),(4,4),(4,4)), mode='edge')
            mask = np.pad(mask, pad_width=((2,2),(4,4),(4,4)), mode='edge')
            # crop to complete shift
            ct_im = ct_im[2+cc_shift:66+cc_shift, 4+ap_shift:132+ap_shift, 4+lr_shift:132+lr_shift]
            mask = mask[2+cc_shift:66+cc_shift, 4+ap_shift:132+ap_shift, 4+lr_shift:132+lr_shift]

        if self.rotations:
            raise NotImplementedError # TODO !! - I have an implementation in 3DSegmentationNetwork which can be applied -> rotations in the axial plane only I should think? -10->10 degrees

        if self.scaling:
            raise NotImplementedError # TODO -> same story here    
        
        if self.flips:
            raise NotImplementedError # LR flips shouldn't be applied I don't think
    
        # perform window-levelling here, create 3 channels
        #ct_im3 = np.zeros(shape=(3,) + ct_im.shape)
        #ct_im3[0] = windowLevelNormalize(ct_im, level=50, window=400)   # abdomen "soft tissues"
        #ct_im3[1] = windowLevelNormalize(ct_im, level=30, window=150)   # liver
        #ct_im3[2] = windowLevelNormalize(ct_im, level=400, window=1800) # spine bone level

        # start with a single soft tissue channel
        ct_im = windowLevelNormalize(ct_im, level=50, window=400)[np.newaxis]   # abdomen "soft tissues"
        
        # use one-hot masks
        mask = (np.arange(6) == mask[...,None]).astype(int)

        # send it
        return {'ct_im': ct_im, 'mask': mask, 'ignore_index': ignore_index}
        
    def __len__(self):
        return len(self.availableImages)
    
if __name__ == '__main__':
    main()