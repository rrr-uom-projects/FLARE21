# generate_targets.py
## script to generate the the CoM targets of the organs
import numpy as np
from skimage.transform import rescale, resize
import SimpleITK as sitk
from utils import getFiles
import os

imdir = "/data/FLARE21/training_data/TrainingImg/"
maskdir = "/data/FLARE21/training_data/TrainingMask/"
targetdir = "/data/FLARE21/training_data/CoM_targets/"

outputdir = "/data/FLARE21/training_data/scaled_ims/" 

# OARs : 1 - Liver, 2 - Kidneys, 3 - Spleen, 4 - Pancreas
# IMPORTANT! : sitk_image.GetDirection()[-1] -> (1 or -1) -> flip cranio-caudally if -1

sizes_scaled = np.zeros((361,3))
spacings_scaled = np.zeros((361,3))
for pdx, fname in enumerate(sorted(getFiles(imdir))):
    # load files
    print(f"Processing {fname.replace('_0000.nii.gz','')}")
    sitk_im = sitk.ReadImage(os.path.join(imdir, fname))
    im = sitk.GetArrayFromImage(sitk_im)
    #sitk_mask = sitk.ReadImage(os.path.join(maskdir, fname.replace('_0000','')))
    #mask = sitk.GetArrayFromImage(sitk_mask)

    # check if flip required
    if sitk_im.GetDirection()[-1] == -1:
        print("Image upside down, CC flip required!")
        im = np.flip(im, axis=0)    # flip CC
        im = np.flip(im, axis=2)    # flip LR --> this works, should be made more robust though (with sitk cosine matrix)

    # load targets
    CoM_targets = np.load(os.path.join(targetdir, fname.replace('_0000.nii.gz','_targets.npy')))

    # resample all images to common slice thickness (2.5mm)
    spacing = np.array(sitk_im.GetSpacing())
    size = np.array(im.shape)
    scale_factor = np.array([64,128,128]) / size
    im = resize(im, output_shape=(64,128,128), order=3, anti_aliasing=True, preserve_range=True)
    # also rescale the targets - skip scaling -235 codes
    errcode_mask = (CoM_targets != -235).astype(bool)
    CoM_targets *= ((scale_factor * errcode_mask) + ~errcode_mask)
    # rescale spacings
    spacing /= scale_factor[[2,1,0]]
    # lil bit of output
    print(f"Rescaling, factor: {scale_factor}, new spacing {spacing} ...")

    # finally clip intensity range (true HU - not Wm HU)
    im = np.clip(im, -1024, 2000)

    # output
    np.save(os.path.join(outputdir, fname.replace('_0000.nii.gz','.npy')), im)
    np.save(os.path.join(targetdir, fname.replace('_0000.nii.gz','_targets_scaled.npy')), CoM_targets)
    
    # extras
    spacings_scaled[pdx] = spacing

# save newly scaled spacings and sizes
np.save(os.path.join("/data/FLARE21/training_data/", "spacings_scaled.npy"), spacings_scaled)


'''
# maybe useful later? realised not needed now
# resample all images to common slice thickness (2.5mm)
    spacing = np.array(sitk_im.GetSpacing())
    slice_thickness = spacing[-1]
    if slice_thickness != 2.5:
        scale_factor = slice_thickness / 2.5
        im = rescale(im, scale=(scale_factor,1,1), order=3, anti_aliasing=True, preserve_range=True)
        # also rescale the targets - skip scaling -235 codes
        errcode_mask = (CoM_targets[:,:,0] != -235).astype(bool)
        CoM_targets[:,:,0] *= ((scale_factor * errcode_mask) + ~errcode_mask)
        # rescale spacings
        spacing[-1] = slice_thickness / scale_factor
        # lil bit of output
        print(f"Rescaling, factor: {scale_factor}, new spacing {spacing} ...")
'''