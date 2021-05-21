# generate_targets.py
## script to generate the the CoM targets of the organs
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.transform import rescale, resize
from scipy.ndimage import center_of_mass
import SimpleITK as sitk
from utils import getFiles
import os

maskdir = "/data/FLARE21/training_data/TrainingMask/"
targetdir = "/data/FLARE21/training_data/CoM_targets/"

# OARs : 1 - Liver, 2 - Kidneys, 3 - Spleen, 4 - Pancreas
# organHunter OARs : 1 - Liver, 2 - Kidney L, 3 - Kidney R, 4 - Spleen, 5 - Pancreas
# IMPORTANT! : sitk_image.GetDirection()[-1] -> (1 or -1) -> flip cranio-caudally if -1

def get_CoM(binary_volume):
    # Get centre of mass of structure in binary volume, return error code (-235's) if no structure - sparse labels
    if binary_volume.sum():
        return np.array(center_of_mass(binary_volume))
    return np.array([-235, -235, -235])

CoM_targets = np.zeros(shape=(361, 5, 3)) # (patient, oar, (cc,ap,lr))
for pdx, fname in enumerate(sorted(getFiles(maskdir))):
    # load files
    print(f"Processing {fname.replace('.nii.gz','')}")
    oars_mask_sitk = sitk.ReadImage(os.path.join(maskdir, fname))
    oars_mask = sitk.GetArrayFromImage(oars_mask_sitk)

    # check if flip required
    if oars_mask_sitk.GetDirection()[-1] == -1:
        print("Image upside down, CC flip required!")
        oars_mask = np.flip(oars_mask, axis=0)          # flip CC
        oars_mask = np.flip(oars_mask, axis=2)          # flip LR --> this works, should be made more robust though (with sitk cosine matrix)

    # calculate CoM for each oar
    for oar_idx, (oar_label, oar_name) in enumerate(zip([1,2,3,4],["Liver","Kidneys","Spleen","Pancreas"])):
        binary_oar_mask = (oars_mask==oar_label)
        if oar_name == "Kidneys":
            # split LR - reimplement this better at a later time
            CoM_L = get_CoM(binary_oar_mask[:,:,:256])  # find kidney in first half
            binary_oar_mask[:,:,:256] = 0               # blank first kidney
            CoM_R = get_CoM(binary_oar_mask)            # find second kidney
            print(f"{oar_name.replace('s',' L')} CoM: {CoM_L}, {oar_name.replace('s',' R')} CoM: {CoM_R}")
            CoM_targets[pdx, oar_idx] = CoM_L
            CoM_targets[pdx, oar_idx+1] = CoM_R
        else:
            CoM = get_CoM(binary_oar_mask)
            print(f'{oar_name}CoM: {CoM}')
            if oar_name == "Liver":                    # ugly but necessary due to splitting the kidneys
                CoM_targets[pdx, oar_idx] = CoM
            else: 
                CoM_targets[pdx, oar_idx+1] = CoM
        # rough sanity checks
        #if CoM_targets[pdx, 0, 1] < CoM_targets[pdx, 1, 1]:
            #print("Liver behind kidney!")
            #exit()
        if CoM_targets[pdx, 0, 2] < 256:
            print("Liver on the left!")
            exit()
            
    np.save(os.path.join(targetdir, f"{fname.replace('.nii.gz','')}_targets.npy"), CoM_targets[pdx])
np.save(os.path.join("/data/FLARE21/training_data/", f"all_targets.npy"), CoM_targets)