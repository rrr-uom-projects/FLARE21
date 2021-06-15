import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
from scipy.ndimage import zoom, rotate
import random
import os
import argparse as ap

from models import light_segmenter
from trainer import segmenter_trainer
from roughSeg.utils import k_fold_split_train_val_test, get_logger, get_number_of_learnable_parameters, getFiles, windowLevelNormalize

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
    model = light_segmenter(n_classes=6, in_channels=1, p_drop=0.25)

    for param in model.parameters():
        param.requires_grad = True

    # put the model on GPU(s)
    device='cuda'
    model.to(device)

    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')
    train_BS = int(8)
    val_BS = int(6)
    train_workers = int(4)
    val_workers = int(2)

    # allocate ims to train, val and test
    dataset_size = len(sorted(getFiles(imagedir)))
    train_inds, val_inds, test_inds = k_fold_split_train_val_test(dataset_size, fold_num=args.fold_num, seed=230597)

    # Create them dataloaders
    train_data = segmenter_Dataset(imagedir=imagedir, maskdir=maskdir, image_inds=train_inds, shift_augment=True, rotate_augment=True, scale_augment=True, flip_augment=False)
    train_loader = DataLoader(dataset=train_data, batch_size=train_BS, shuffle=True, pin_memory=False, num_workers=train_workers, worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))
    val_data = segmenter_Dataset(imagedir=imagedir, maskdir=maskdir, image_inds=val_inds, shift_augment=False, rotate_augment=False, scale_augment=False, flip_augment=False)
    val_loader = DataLoader(dataset=val_data, batch_size=val_BS, shuffle=True, pin_memory=False, num_workers=val_workers, worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))

    # Create the optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.005)

    # Create learning rate adjustment strategy
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=175, verbose=True)
    
    # Create model trainer
    trainer = segmenter_trainer(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, device=device, train_loader=train_loader, 
                                    val_loader=val_loader, logger=logger, checkpoint_dir=checkpoint_dir, max_num_epochs=1000, patience=500, iters_to_accumulate=1)
    
    # Start training
    trainer.fit()

    # Romeo Dunn
    return
    
class segmenter_Dataset(data.Dataset):
    def __init__(self, imagedir, maskdir, image_inds, shift_augment=True, rotate_augment=True, scale_augment=True, flip_augment=False):
        self.imagedir = imagedir
        self.maskdir = maskdir
        self.availableImages = [sorted(getFiles(imagedir))[ind] for ind in image_inds]
        self.image_inds = image_inds
        self.shifts = shift_augment
        self.flips = flip_augment
        self.rotations = rotate_augment
        self.scaling = scale_augment
        self.ignore_oars = np.load("/data/FLARE21/training_data/labels_present360.npy")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
           idx = idx.tolist()
        imageToUse = self.availableImages[idx]
        #spacing = np.load("/data/FLARE21/training_data/spacings_scaled.npy")[idx][[2,0,1]]
        ct_im = np.load(os.path.join(self.imagedir, imageToUse))
        mask = np.load(os.path.join(self.maskdir, imageToUse))
        ignore_index = self.ignore_oars[self.image_inds[idx]]

        # Augmentations
        if self.shifts:
            # find shift values
            cc_shift, ap_shift, lr_shift = random.randint(-2,2), random.randint(-4,4), random.randint(-4,4)
            # pad for shifting into
            ct_im = np.pad(ct_im, pad_width=((2,2),(4,4),(4,4)), mode='constant', constant_values=-1024)
            mask = np.pad(mask, pad_width=((2,2),(4,4),(4,4)), mode='constant', constant_values=0)
            # crop to complete shift
            ct_im = ct_im[2+cc_shift:66+cc_shift, 4+ap_shift:132+ap_shift, 4+lr_shift:132+lr_shift]
            mask = mask[2+cc_shift:66+cc_shift, 4+ap_shift:132+ap_shift, 4+lr_shift:132+lr_shift]

        if self.rotations and random.random()<0.5:
            # taking implementation from 3DSegmentationNetwork which can be applied -> rotations in the axial plane only I should think? -10->10 degrees?
            roll_angle = np.clip(np.random.normal(loc=0,scale=3), -10, 10)
            ct_im = self.rotation(ct_im, roll_angle, rotation_plane=(1,2), is_mask=False)
            mask = self.rotation(mask, roll_angle, rotation_plane=(1,2), is_mask=True)

        if self.scaling and random.random()<0.5:
            # same here -> zoom between 80-120%
            scale_factor = np.clip(np.random.normal(loc=1.0,scale=0.05), 0.8, 1.2)
            ct_im = self.scale(ct_im, scale_factor, is_mask=False)
            mask = self.scale(mask, scale_factor, is_mask=True)
        
        if self.flips:
            raise NotImplementedError # LR flips shouldn't be applied I don't think
    
        # perform window-levelling here, create 3 channels

        ###
        # This is where to add in extra augmentations and channels
        ###

        #ct_im3 = np.zeros(shape=(3,) + ct_im.shape)
        #ct_im3[0] = windowLevelNormalize(ct_im, level=50, window=400)   # abdomen "soft tissues"
        #ct_im3[1] = windowLevelNormalize(ct_im, level=30, window=150)   # liver
        #ct_im3[2] = windowLevelNormalize(ct_im, level=400, window=1800) # spine bone level

        # start with a single soft tissue channel
        ct_im = windowLevelNormalize(ct_im, level=50, window=400)[np.newaxis]   # abdomen "soft tissues"
        
        # use one-hot masks
        mask = (np.arange(6) == mask[...,None]).astype(int)
        mask = np.transpose(mask, axes=(3,0,1,2))

        # send it
        return {'ct_im': ct_im, 'mask': mask, 'ignore_index': ignore_index}
        
    def __len__(self):
        return len(self.availableImages)

    def scale(self, image, scale_factor, is_mask):
        # scale the image or mask using scipy zoom function
        order, cval = (0, 0) if is_mask else (3, -1024)
        height, width, depth = image.shape
        zheight = int(np.round(scale_factor*height))
        zwidth = int(np.round(scale_factor*width))
        zdepth = int(np.round(scale_factor*depth))
        # zoomed out
        if scale_factor < 1.0:
            new_image = np.full_like(image, cval)
            ud_buffer = (height-zheight) // 2
            ap_buffer = (width-zwidth) // 2
            lr_buffer = (depth-zdepth) // 2
            new_image[ud_buffer:ud_buffer+zheight, ap_buffer:ap_buffer+zwidth, lr_buffer:lr_buffer+zdepth] = zoom(input=image, zoom=scale_factor, order=order, mode='constant', cval=cval)[0:zheight, 0:zwidth, 0:zdepth]
            return new_image
        elif scale_factor > 1.0:
            new_image = zoom(input=image, zoom=scale_factor, order=order, mode='constant', cval=cval)[0:zheight, 0:zwidth, 0:zdepth]
            ud_extra = (new_image.shape[0] - height) // 2
            ap_extra = (new_image.shape[1] - width) // 2
            lr_extra = (new_image.shape[2] - depth) // 2
            new_image = new_image[ud_extra:ud_extra+height, ap_extra:ap_extra+width, lr_extra:lr_extra+depth]
            return new_image
        return image
    
    def rotation(self, image, rotation_angle, rotation_plane, is_mask):
        # rotate the image or mask using scipy rotate function
        order, cval = (0, 0) if is_mask else (3, -1024)
        return rotate(input=image, angle=rotation_angle, axes=rotation_plane, reshape=False, order=order, mode='constant', cval=cval)
    
if __name__ == '__main__':
    main()