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

out_imdir = "/data/FLARE21/training_data/scaled_ims/"
out_maskdir = "/data/FLARE21/training_data/scaled_masks/"

# OARs : 1 - Liver, 2 - Kidneys, 3 - Spleen, 4 - Pancreas
# IMPORTANT! : sitk_image.GetDirection()[-1] -> (1 or -1) -> flip cranio-caudally if -1

def split_kidneys(mask):
    mask_new = mask.copy()
    mask_new[mask > 1] += 1             # bump up the kidneys, spleen and pancreas -> OAR labels : 1 - Liver, 3 - Kidneys, 4 - Spleen, 5 - Pancreas
    mask_new[:,:,:256][(mask[:,:,:256] == 2)] = 2 # reassign the left kidney                 -> OAR labels : 1 - Liver, 2 - Kidney L, 3 - Kidney R, 4 - Spleen, 5 - Pancreas
    return mask_new

sizes_scaled = np.zeros((361,3))
spacings_scaled = np.zeros((361,3))
labels_present = np.empty(shape=(361,5), dtype=bool)
label_freq = np.zeros((6))
for pdx, fname in enumerate(sorted(getFiles(imdir))):
    # load files
    print(f"Processing {fname.replace('_0000.nii.gz','')}")
    sitk_im = sitk.ReadImage(os.path.join(imdir, fname))
    im = sitk.GetArrayFromImage(sitk_im)
    sitk_mask = sitk.ReadImage(os.path.join(maskdir, fname.replace('_0000','')))
    mask = sitk.GetArrayFromImage(sitk_mask)

    # check if flip required
    if sitk_im.GetDirection()[-1] == -1:
        print("Image upside down, CC flip required!")
        im = np.flip(im, axis=0)        # flip CC
        im = np.flip(im, axis=2)        # flip LR --> this works, should be made more robust though (with sitk cosine matrix)
        mask = np.flip(mask, axis=0)
        mask = np.flip(mask, axis=2)
    
    # split the kidneys
    mask = split_kidneys(mask)

    # identify missing labels
    labels_present[pdx] = np.array([(mask==oar_label).any() for oar_label in range(1,6)])
    
    # resample all images to common size
    spacing = np.array(sitk_im.GetSpacing())
    size = np.array(im.shape)
    scale_factor = np.array([64,128,128]) / size
    im = resize(im, output_shape=(64,128,128), order=3, anti_aliasing=True, preserve_range=True)
    mask = np.round(resize(mask, output_shape=(64,128,128), order=0, anti_aliasing=False, preserve_range=True)).astype(np.uint8)
    
    # rescale spacings
    spacing /= scale_factor[[2,1,0]]
    
    # lil bit of output
    print(f"Rescaling, factor: {scale_factor}, new spacing {spacing} ...")

    # finally clip intensity range (true HU - not Wm HU)
    im = np.clip(im, -1024, 2000)

    # output
    #np.save(os.path.join(out_imdir, fname.replace('_0000.nii.gz','.npy')), im)
    assert((im == np.load(os.path.join(out_imdir, fname.replace('_0000.nii.gz','.npy')))).all())
    np.save(os.path.join(out_maskdir, fname.replace('_0000.nii.gz','.npy')), mask)

    # extras
    spacings_scaled[pdx] = spacing
    for odx in range(6):
        label_freq[odx] += (mask==odx).sum()

# save newly scaled spacings and sizes
#np.save(os.path.join("/data/FLARE21/training_data/", "spacings_scaled.npy"), spacings_scaled)
assert((spacings_scaled == np.load(os.path.join("/data/FLARE21/training_data/", "spacings_scaled.npy"))).all())
np.save("/data/FLARE21/training_data/labels_present.npy", labels_present)
print(label_freq)
np.save("/data/FLARE21/training_data/label_freq.npy", label_freq)