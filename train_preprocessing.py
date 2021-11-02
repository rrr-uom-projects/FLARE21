## train_preprocessing.py
# A script to prepare images and gold standard segmentations prior to training
# 1. read in the .nii.gz image and mask pair
# 2. correct orientation errors using SimpleITK
# 3. generate body mask with thresholding and binary operations
# 4. identify missing OAR labels 
# 5. resample to a common size (96 x 192 x 192 voxels)
# 6. save output to training images and masks source directory

import numpy as np
from skimage.transform import resize
from scipy.ndimage import binary_fill_holes, binary_closing
import SimpleITK as sitk
from utils import getFiles, try_mkdir
import os
from tqdm import tqdm

imdir = "/data/FLARE_datasets/AbdomenCT-1K/Image/"      # DIRECTORY PATH TO SETUP
maskdir = "/data/FLARE_datasets/AbdomenCT-1K/Mask/"     # DIRECTORY PATH TO SETUP
out_dir = "/data/FLARE21/AbdomenCT-1K_training_data/"   # DIRECTORY PATH TO SETUP
out_imdir = os.path.join(out_dir, "scaled_ims/")
out_maskdir = os.path.join(out_dir, "scaled_masks/")

out_resolution = (96,192,192)

# create directories to save preprocessed data to
try_mkdir(out_dir)
try_mkdir(out_imdir)
try_mkdir(out_maskdir)

# OARs : 1 - Liver, 2 - Kidneys, 3 - Spleen, 4 - Pancreas
# IMPORTANT! : sitk_image.GetDirection()[-1] -> (1 or -1) -> flip cranio-caudally if -1
# Output: OAR labels : 1- Body, 2 - Liver, 3 - Kidneys , 4 - Spleen, 5 - Pancreas

startn = 800
stopn = 1000

fnames = sorted(getFiles(maskdir))[startn:stopn]
n_images = len(fnames)
print(f"n_images: {n_images}")

sizes_scaled = np.zeros((n_images, 3))
spacings_scaled = np.zeros((n_images, 3))
labels_present = np.empty(shape=(n_images, 4), dtype=bool)
label_freq = np.zeros((6))
for pdx, fname in enumerate(tqdm(fnames)):
    # load files
    #print(f"Processing {fname.replace('_0000.nii.gz','')}")
    sitk_im = sitk.ReadImage(os.path.join(imdir, fname.replace('.nii.gz','_0000.nii.gz')))
    im = sitk.GetArrayFromImage(sitk_im)
    sitk_mask = sitk.ReadImage(os.path.join(maskdir, fname))
    mask = sitk.GetArrayFromImage(sitk_mask).astype(float)

    # check if flip required
    if sitk_im.GetDirection()[-1] == -1:
        print("Image upside down, CC flip required!")
        im = np.flip(im, axis=0)        # flip CC
        im = np.flip(im, axis=2)        # flip LR --> this works, should be made more robust though (with sitk cosine matrix)
        mask = np.flip(mask, axis=0)
        mask = np.flip(mask, axis=2)

    # use id body to generate body delineation too
    # threshold
    body_mask = np.zeros_like(im)
    body_inds = im > -200
    body_mask[body_inds] += 1

    # generate closing
    closing = binary_closing(body_mask, iterations=3)
    body_mask[(closing != 0)] = 1

    # binary fill in axial plane
    fill_struct = np.zeros((3,3,3))
    fill_struct[1] = np.array([[0,1,0],[1,1,1],[0,1,0]])
    filling = binary_fill_holes(body_mask, structure=fill_struct)
    body_mask[(filling != 0)] = 1

    # increment existing structures
    labelled_inds = (mask != 0)
    mask[labelled_inds] += 1
    
    # add body as background (1st eliminating overlap with existing structures)
    body_mask[labelled_inds] = 0
    mask += body_mask

    # identify missing labels
    labels_present[pdx] = np.array([(mask==oar_label).any() for oar_label in range(2,6)])
    
    # resample all images to common size
    spacing = np.array(sitk_im.GetSpacing())
    size = np.array(im.shape)
    scale_factor = np.array(out_resolution) / size
    im = resize(im, output_shape=out_resolution, order=3, anti_aliasing=True, preserve_range=True).astype(np.float16)
    mask = np.round(resize(mask, output_shape=out_resolution, order=0, anti_aliasing=False, preserve_range=True)).astype(np.uint8)
    
    # rescale spacings
    spacing /= scale_factor[[2,1,0]]
    
    # lil bit of output
    #print(f"Rescaling, factor: {scale_factor}, new spacing {spacing} ...")

    # finally clip intensity range (true HU - not Wm HU)
    im = np.clip(im, -1024, 2000)

    # output
    np.save(os.path.join(out_imdir, fname.replace('.nii.gz','.npy')), im)
    np.save(os.path.join(out_maskdir, fname.replace('.nii.gz','.npy')), mask)

    # extras
    spacings_scaled[pdx] = spacing
    for odx in range(6):
        label_freq[odx] += (mask==odx).sum()

# save newly scaled spacings and sizes
np.save(os.path.join(out_dir, f"spacings_scaled_{startn}_{stopn}.npy"), spacings_scaled)
np.save(os.path.join(out_dir, f"labels_present_{startn}_{stopn}.npy"), labels_present)
print(label_freq)
np.save(os.path.join(out_dir, f"label_freq_{startn}_{stopn}.npy"), label_freq)
