"""
Frankenscript to input & output nii.gz images
"""

import SimpleITK as sitk

from torch.utils.data import Dataset, DataLoader

import argparse as ap
import os
import numpy as np
import numba
import time

## Global configuration variables

out_resolution = (96,160,160)
level = 400
window = 50
minval = level - window//2
maxval = level + window//2


class InferenceRecord:
    def __init__(self, image, flipped, original_size, original_spacing):
        self.npy_image = image
        self.flipped = flipped
        self.original_size = original_size
        self.original_spacing = original_spacing



@numba.jit(parallel=True, cache=True)
def WL_norm(img):
    """
    Apply a window and level transformation to the image data

    Notes:
        - Uses global variables to reduce computation in the function
        - Profiling suggests this is about as fast as it can go
        - Fastest when image is kept as int16
    """
    wld = np.minimum(maxval, np.maximum(img, minval)) #np.clip(img, minval, maxval)
    wld -= minval
    wld = wld // window
    return wld

preprocessings = []

def load_nifty(path):
    read_start = time.time()
    sitk_im = sitk.ReadImage(path)
    read_end = time.time()

    ## immediately downsample to expected resolution
    resamp_start = time.time()
    reference_image = sitk.Image(out_resolution, sitk_im.GetPixelIDValue())
    reference_image.SetOrigin(sitk_im.GetOrigin())
    reference_image.SetDirection(sitk_im.GetDirection())
    reference_image.SetSpacing([sz*spc/nsz for nsz,sz,spc in zip(out_resolution, sitk_im.GetSize(), sitk_im.GetSpacing())])
    
    ## Should we do some smoothing here? 
    sitk_im_resamp = sitk.Resample(sitk_im, reference_image)
    resamp_end = time.time()

    
    im_o = sitk.GetArrayFromImage(sitk_im_resamp).astype(np.int16) ## cast back to short - some appear to be stored as float32?

    # check if flip required
    flipped = False
    if sitk_im.GetDirection()[-1] == -1:
        # print("Image upside down, CC flip required!")
        flip_start = time.time()
        im_o = np.flip(im_o, axis=0)        # flip CC
        im_o = np.flip(im_o, axis=2)        # flip LR --> this works, should be made more robust though (with sitk cosine matrix)
        flip_end = time.time()
        flipped = True
        
    ## perform normalisation
    wld_start = time.time()
    im_o_n = WL_norm(im_o) ## this actually takes the longest time!
    wld_end = time.time()

    # print(f"all_preproc:{wld_end - resamp_start}\nall_load:{wld_end - read_start}")
    preprocessings.append(wld_end - read_start)

    return InferenceRecord(im_o_n.astype(np.float32), flipped, sitk_im.GetSize(), sitk_im.GetSpacing())



def main(args):
    images_2_segment = [os.path.join(args.input_dir, im) for im in os.listdir(args.input_dir)]
    print(f"Detected {len(images_2_segment)} images to segment...")

    inference_targets = [load_nifty(impath) for impath in images_2_segment]

    print(np.mean(preprocessings))

    pass






if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("input_dir", help="Directory containing images to segment")
    parser.add_argument("output_dir", help="Directory in which to put the output")

    args = parser.parse_args()
    main(args)
