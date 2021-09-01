import os
import numpy as np
import SimpleITK as sitk

from roughSeg.utils import getFiles

im_dir = "/data/FLARE21/validation_data/ValidationImg/"
seg_dir = "/data/FLARE21/RRR_MCRVal1/"

fnames = sorted(getFiles(seg_dir))

vols = np.zeros((len(fnames), 4))
for fdx, fname in enumerate(fnames):
    segs = sitk.ReadImage(os.path.join(seg_dir, fname))
    spacing = segs.GetSpacing()
    vox_vol = spacing[0] * spacing[1] * spacing[2]
    segs = sitk.GetArrayFromImage(segs)
    for oar_idx, oar_label in enumerate([1,2,3,4]):
        vols[fdx, oar_idx] = round((segs==oar_label).sum() * vox_vol)
    print(f"{fname.replace('.nii.gz','')} vols: {vols[fdx]}")
np.save("/data/FLARE21/val1_oar_volumes.npy", vols)

