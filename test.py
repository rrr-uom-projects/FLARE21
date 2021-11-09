import torch
import numpy as np
from scipy import ndimage
import os
import time
from tqdm import tqdm
import SimpleITK as sitk
from skimage.transform import resize

from models import nano_segmenter
from utils import get_logger, getFiles, windowLevelNormalize
import archive.roughSeg.deepmind_metrics as deepmind_metrics

nii_source_dir = "/data/FLARE_datasets/AbdomenCT-1K/"
npy_source_dir = "/data/FLARE21/AbdomenCT-1K_training_data/"
image_dir = "/data/FLARE21/AbdomenCT-1K_training_data/scaled_ims/"
mask_dir = "/data/FLARE21/AbdomenCT-1K_training_data/scaled_masks/"
output_dir = "/data/FLARE21/results/AbdomenCT-1K_tumor/"
input_size = (96,192,192)

organs = ["liver", "kidneys", "spleen", "pancreas"]

def dice(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    return (2. * (a*b).sum()) / (a.sum() + b.sum())

def main():
    # Create logger
    logger = get_logger('fullRes_testing')

    # get stuff
    all_fnames = sorted(getFiles(mask_dir))
    labels_present_all = np.load(os.path.join(npy_source_dir, "labels_present.npy"))
    try:
        os.mkdir(output_dir)
    except OSError:
        pass

    # Create the model
    model = nano_segmenter(n_classes=7, in_channels=2, p_drop=0) #, initial_levels=[1,1,1], initial_windows=[1,1,1]

    # put the model on GPU
    model.to('cuda')

    # get checkpoint dir
    checkpoint_dir = f"/data/FLARE21/models/AbdomenCT-1K_tumor/"

    # load in the best model version
    model.load_best(checkpoint_dir, logger)
    for param in model.parameters():
        param.requires_grad = False

    # fix model
    model.eval()

    # get test fnames
    all_fnames = sorted(getFiles(mask_dir))
    test_fnames = list(filter(lambda fname: fname not in sorted(getFiles("/data/FLARE21/AbdomenCT-1K_training_data/scaled_masks_w_tumors/")), all_fnames))

    # setup result grids
    res = np.full(shape=(len(test_fnames), 4, 2), fill_value=np.nan)

    # iterate over each testing image
    for pat_idx, test_fname in enumerate(tqdm(test_fnames)):
        # load image and normalise
        t = time.time()
        sitk_image = sitk.ReadImage(os.path.join(nii_source_dir, "Image", test_fname.replace('.npy','_0000.nii.gz')))
        ct_im = sitk.GetArrayFromImage(sitk_image)
        # load gold standard segmentation in full resolution
        sitk_mask = sitk.ReadImage(os.path.join(nii_source_dir, "Mask", test_fname.replace('.npy','.nii.gz')))
        gold_mask = sitk.GetArrayFromImage(sitk_mask).astype(int)
        # reorient if required
        if sitk_mask.GetDirection()[-1] == -1:
            gold_mask = np.flip(gold_mask, axis=0).copy()
            ct_im = np.flip(ct_im, axis=0).copy()
            gold_mask = np.flip(gold_mask, axis=2).copy() 
            ct_im = np.flip(ct_im, axis=2).copy()
        #logger.info(f"Image loading took {time.time()-t:.4f} seconds")
        t = time.time()
        ct_im = resize(ct_im, output_shape=input_size, order=3, anti_aliasing=True, preserve_range=True)
        #logger.info(f"Image downsampling took {time.time()-t:.4f} seconds")
        # preprocessing
        ct_im = np.clip(ct_im, -1024, 2000)
        #ct_im = windowLevelNormalize(ct_im, level=50, window=400)[np.newaxis, np.newaxis] # add dummy batch and channels axes
        ct_im2 = np.zeros(shape=(2,) + ct_im.shape)
        ct_im2[0] = windowLevelNormalize(ct_im, level=50, window=400)   # abdomen "soft tissues"
        ct_im2[1] = windowLevelNormalize(ct_im, level=60, window=100)   # pancreas
        ct_im = ct_im2[np.newaxis].copy() # add dummy batch axis
        # run forward pass
        t = time.time()
        prediction = model(torch.tensor(ct_im, dtype=torch.float).to('cuda'))
        #logger.info(f"{test_fname} inference took {time.time()-t:.4f} seconds")
        # change prediction from one-hot to mask and move back to cpu for metric calculation
        prediction = torch.squeeze(prediction)
        prediction = torch.argmax(prediction, dim=0)
        prediction = prediction.cpu().numpy().astype(int)
        # drop the body and label the kidneys together          # OAR labels : 1 - Body, 2 - Liver, 3 - Kidneys, 4 - Spleen, 5 - Pancreas
        prediction -= 1                                         
        prediction = np.clip(prediction, 0, prediction.max())   # -> OAR labels : 0 - Background, 1 - Liver, 2 - Kidneys, 3 - Spleen, 4 - Pancreas
        # rescale the prediction to match the full-resolution mask
        t = time.time()
        prediction = np.round(resize(prediction, output_shape=gold_mask.shape, order=0, anti_aliasing=False, preserve_range=True)).astype(np.uint8)
        #logger.info(f"Image upsampling took {time.time()-t:.4f} seconds")
        # save output
        try:
            os.mkdir(os.path.join(output_dir, "full_res_test_segs/"))
        except OSError:
            pass
        np.save(os.path.join(output_dir, "full_res_test_segs/", 'pred_'+test_fname), prediction)
        # get spacing for this image
        spacing = np.array(sitk_mask.GetSpacing())[[2,0,1]]
        t = time.time()
        # get present labels
        test_ind = np.argwhere(np.array(test_fname)==np.array(all_fnames))[0,0]
        labels_present = labels_present_all[test_ind]
        # calculate metrics
        first_oar_idx = 1
        for organ_idx, organ_num in enumerate(range(first_oar_idx, gold_mask.max()+1)):
            # check if label present in gs, skip if not
            if not labels_present[organ_idx]:
                #logger.info(f"{test_fname} missing {organs[organ_idx]}, skipping...")
                continue
            # Need to binarise the masks for the metric computation
            gs = (gold_mask==organ_num).astype(int)
            pred = (prediction==organ_num).astype(int)
            # post-processing using scipy.ndimage.label to eliminate extraneous voxels
            labels, num_features = ndimage.label(input=pred)
            if num_features > 1:
                # disconnected bits present, iterate over them to check which to keep
                # if less than 20% the volume of the primary region, get rid
                primary_vol_threshold = (labels==1).sum() * 0.20
                for feature_label in range(2, num_features+1):
                    if primary_vol_threshold > (labels==feature_label).sum():
                        pred[labels==feature_label] = 0
            # compute the surface distances
            surface_distances = deepmind_metrics.compute_surface_distances(gs.astype(bool), pred.astype(bool), spacing)
            # compute desired metric
            surface_DSC = deepmind_metrics.compute_surface_dice_at_tolerance(surface_distances, tolerance_mm=1.)
            # store result
            res[pat_idx, organ_idx, 0] = dice(gs, pred)
            res[pat_idx, organ_idx, 1] = surface_DSC
        #logger.info(f"Seg processing took {time.time()-t:.4f} seconds")

    # save results
    np.save(os.path.join(output_dir, "full_res_results_grid.npy"), res)

    # Romeo Dunn
    return

if __name__ == '__main__':
    main()