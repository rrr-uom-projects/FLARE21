import torch
import numpy as np
from scipy import ndimage
import os
import time
import SimpleITK as sitk

from roughSeg.utils import k_fold_split_train_val_test, get_logger, getFiles, windowLevelNormalize
import roughSeg.deepmind_metrics as deepmind_metrics

source_dir = "/data/FLARE21/training_data_192_sameKidneys/"
input_dir = "/data/FLARE21/models/full_runs/tiny_segmenter_192/fold1/outputs/"
mask_dir = "/data/FLARE21/training_data/TrainingMask/"
output_dir = "/data/FLARE21/results/full_runs/tiny_segmenter_192_donalOutputs/"
input_size = (96,192,192)
folds = [1]
organs = ["liver", "kidneys", "spleen", "pancreas"]

def dice(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    return (2. * (a*b).sum()) / (a.sum() + b.sum())

def main():
    # Create logger
    logger = get_logger('fullRes_testing')

    # get stuff
    imagedir = os.path.join(source_dir, "scaled_masks/")
    dataset_size = len(sorted(getFiles(imagedir))) # 72
    all_fnames = sorted(getFiles(imagedir))
    labels_present_all = np.load(os.path.join(source_dir, "labels_present.npy"))
    try:
        os.mkdir(output_dir)
    except OSError:
        pass

    # setup result grids
    _, _, dummy_test_inds = k_fold_split_train_val_test(dataset_size, fold_num=1, seed=230597)
    res = np.full(shape=(len(folds), len(dummy_test_inds), 4, 2), fill_value=np.nan)

    # allocate ims to train, val and test
    train_inds, val_inds, test_inds = k_fold_split_train_val_test(dataset_size, fold_num=1, seed=230597)

    # get test fnames
    test_mask_fnames = [all_fnames[ind] for ind in test_inds]

    # iterate over each testing image
    for pat_idx, (test_fname, test_ind) in enumerate(zip(test_mask_fnames, test_inds)):
        # load gold standard segmentation in full resolution
        sitk_mask = sitk.ReadImage(os.path.join(mask_dir, test_fname.replace('.npy','.nii.gz')))
        gold_mask = sitk.GetArrayFromImage(sitk_mask).astype(int)
        # get spacing for this image
        spacing = np.array(sitk_mask.GetSpacing())[[2,0,1]]
        # reorient if required
        if sitk_mask.GetDirection()[-1] == -1:
            gold_mask = np.flip(gold_mask, axis=0).copy()
            gold_mask = np.flip(gold_mask, axis=2).copy()
        #load output
        #prediction = np.squeeze(np.load(os.path.join(input_dir, test_fname)))
        #prediction = np.argmax(prediction, axis=0).astype(int8)
        prediction = np.load(os.path.join(input_dir, test_fname))
        # check
        print(test_fname)
        print(prediction.shape)
        print(gold_mask.shape)
        assert(prediction.shape==gold_mask.shape)
        # get present labels
        labels_present = labels_present_all[test_ind]
        # calculate metrics
        first_oar_idx = 1
        for organ_idx, organ_num in enumerate(range(first_oar_idx, gold_mask.max()+1)):
            # check if label present in gs, skip if not
            if not labels_present[organ_idx]:
                logger.info(f"{test_fname} missing {organs[organ_idx]}, skipping...")
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
            surface_DSC = deepmind_metrics.compute_surface_dice_at_tolerance(surface_distances, tolerance_mm=5.)
            # store result
            res[0, pat_idx, organ_idx, 0] = dice(gs, pred)
            res[0, pat_idx, organ_idx, 1] = surface_DSC

    # save results
    np.save(os.path.join(output_dir, "full_res_results_grid.npy"), res)

    # Romeo Dunn
    return

if __name__ == '__main__':
    main()