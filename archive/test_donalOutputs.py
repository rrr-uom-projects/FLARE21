import torch
import numpy as np
from scipy import ndimage
import os
import time
import SimpleITK as sitk
from skimage.transform import resize

from utils import k_fold_split_train_val_test, get_logger, getFiles, windowLevelNormalize
import roughSeg.deepmind_metrics as deepmind_metrics

source_dir = "/data/FLARE21/training_data_192_sameKidneys/"
input_dir = "/data/FLARE21/models/full_runs/nano_segmenter_192/fold1/outputs/"
mask_dir = "/data/FLARE21/training_data/TrainingMask/"
output_dir = "/data/FLARE21/results/full_runs/nano_segmenter_192_donalOutputs/"
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
        print(test_fname)
        #if test_fname != 'train_136.npy': continue
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
        assert(prediction.shape==(96,192,192))
        # drop the body and label the kidneys together          # OAR labels : 1 - Body, 2 - Liver, 3 - Kidney L, 4 - Kidney R, 5 - Spleen, 6 - Pancreas
        prediction -= 1                                         # -> OAR labels : 0 - Body, 1 - Liver, 2 - Kidney L, 3 - Kidney R, 4 - Spleen, 5 - Pancreas
        print("WARNING: assuming model trained with kidneys as same label...")
        #prediction[prediction >= 3] -= 1                        # -> OAR labels : 0 - Body, 1 - Liver, 2 - Kidneys, 3 - Spleen, 4 - Pancreas
        prediction = np.clip(prediction, 0, prediction.max())   # -> OAR labels : 0 - Background, 1 - Liver, 2 - Kidneys, 3 - Spleen, 4 - Pancreas        
        # rescale the prediction to match the full-resolution mask
        t = time.time()
        prediction = np.round(resize(prediction, output_shape=gold_mask.shape, order=0, anti_aliasing=False, preserve_range=True)).astype(np.uint8)
        np.save(
            f'/home/donal/data/FLARE21/results/full_runs/tiny_segmenter_192_donalOutputs/masks/{test_fname}', prediction)
        logger.info(f"Image upsampling took {time.time()-t:.4f} seconds")
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
            print(organ_idx, dice(gs, pred))
            res[0, pat_idx, organ_idx, 0] = dice(gs, pred)
            res[0, pat_idx, organ_idx, 1] = surface_DSC
            

    # save results
    np.save(os.path.join(output_dir, "full_res_results_grid.npy"), res)

    # Romeo Dunn
    return

if __name__ == '__main__':
    main()
