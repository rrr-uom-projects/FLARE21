"""
Main script for training fine-scale segmentation
"""
import os
import torch
import numpy as np
import cv2
from argparse import ArgumentParser, ArgumentTypeError
import sys
sys.path.append('..')

from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import torch.optim as optim

from model.model import SegmentationModel
from utils.transforms import Resize, NormalizeImage, PrepareForNet, RotateImage, ShiftImage, ScaleImage
from utils.dataset import customDataset
from utils.writer import customWriter
from utils.loops import FineSegmentation
from multiStageSeg.utils import (get_logger, get_number_of_learnable_parameters, 
                                k_fold_split_train_val_test, getFiles)


img_dir = '/data/FLARE21/training_data/scaled_ims/'
mask_dir = '/data/FLARE21/training_data/scaled_masks/'

batch_size = 16
num_epochs=1
lr = 3e-4

train_workers = 4
test_workers = 2


#* Arg Parser.
parser = ArgumentParser(prog="Main training script for fine-scale organ segmentation") 
parser.add_argument("--fold_num", choices=[1, 2, 3, 4, 5], type=int, help="K for K-fold cross-val")
parser.add_argument("--GPUs", choices=[1, 2], type=int, default=1, help="# GPUs to use")
args = parser.parse_args()
checkpoint_dir = f"/data/FLARE21/models/FineSeg/fold_{args.fold_num}/"
os.makedirs(checkpoint_dir, exist_ok=True)

def main():
    train_transforms = Compose([
        ShiftImage(range_=[2, 4, 4, 0], p=0.5),
        RotateImage(range_=10, p=0.5),
        ScaleImage(range_=0.2, p=0.5),

        NormalizeImage(mean=[0.5], std=[0.5]),
        PrepareForNet(), #* This just converts to channels first 
    ])
    test_transforms = Compose([
        NormalizeImage(mean=[0.5], std=[0.5]),
        PrepareForNet(),
    ]) 
    #* Prepare data
    dataset_size = len(getFiles(img_dir))
    train_idx, val_idx, test_idx = k_fold_split_train_val_test(dataset_size, fold_num=args.fold_num, seed=230597)


    train_dataset = customDataset(img_dir, mask_dir, train_transforms, indices=train_idx, apply_WL=True)
    test_dataset = customDataset(img_dir, mask_dir, test_transforms, indices=val_idx, apply_WL=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False,
                              num_workers=train_workers, worker_init_fn=lambda _: np.random.seed(
                                  int(torch.initial_seed()) % (2**32-1)))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=False,
                             num_workers=test_workers, worker_init_fn=lambda _: np.random.seed(
                                 int(torch.initial_seed()) % (2**32-1)))

    #* Prepare model 
    model = SegmentationModel(num_classes=1, path=None,
                              backbone="vitb16_384")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    writer = customWriter(batch_size)
    
    #*Create main logger
    logger = get_logger('FineScale_Seg')
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')
    
    #* Train Object
    kwargs = {'iters_to_accumulate': 4, # For simulating larger batch sizes
            'patience': 10, # For early stopping
            'num_iterations': 1
    }

    seg = FineSegmentation(model, optimizer, train_loader, 
                test_loader, writer, logger, num_epochs=num_epochs, output_path=checkpoint_dir, device="cuda:0", **kwargs)

    seg.forward()

    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
