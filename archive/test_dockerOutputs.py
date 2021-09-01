import torch
import numpy as np
from scipy import ndimage
import os
import time
import SimpleITK as sitk
from skimage.transform import resize

from roughSeg.utils import k_fold_split_train_val_test, get_logger, getFiles, windowLevelNormalize
import roughSeg.deepmind_metrics as deepmind_metrics

docker_dir = "/data/FLARE21/test_output_BS/"
pytorch_dir = "/data/FLARE21/results/full_runs/nano_segmenter_192_1mm/full_res_test_segs/"
mask_dir = "/data/FLARE21/training_data/TrainingMask/"
output_dir = "/data/FLARE21/results/full_runs/nano_segmenter_192_dockerTestBS/"
input_size = (96,192,192)
folds = [2]
organs = ["liver", "kidneys", "spleen", "pancreas"]

def dice(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    return (2. * (a*b).sum()) / (a.sum() + b.sum())

def main():
    # Create logger
    logger = get_logger('fullRes_testing')

    # get stuff
    dataset_size = len(sorted(getFiles(mask_dir))) # 72
    all_fnames = sorted(getFiles(mask_dir))
    labels_present_all = np.load(os.path.join("/data/FLARE21/training_data_192_sameKidneys/", "labels_present.npy"))
    try:
        os.mkdir(output_dir)
    except OSError:
        pass

    # setup result grids
    _, _, dummy_test_inds = k_fold_split_train_val_test(dataset_size, fold_num=2, seed=230597)
    res = np.full(shape=(len(folds), len(dummy_test_inds), 4, 2), fill_value=np.nan)

    # allocate ims to train, val and test
    train_inds, val_inds, test_inds = k_fold_split_train_val_test(dataset_size, fold_num=2, seed=230597)

    # get test fnames
    test_mask_fnames = [all_fnames[ind] for ind in test_inds]

    # iterate over each testing image
    for pat_idx, (test_fname, test_ind) in enumerate(zip(test_mask_fnames, test_inds)):
        print(test_fname)
        #if test_fname != 'train_136.npy': continue
        # load gold standard segmentation in full resolution
        sitk_mask = sitk.ReadImage(os.path.join(mask_dir, test_fname))
        gold_mask = sitk.GetArrayFromImage(sitk_mask).astype(int)
        # load docker output
        docker_pred = sitk.ReadImage(os.path.join(docker_dir, test_fname.replace('.nii','_0000.nii')))
        docker_spac = np.array(docker_pred.GetSpacing())[[2,0,1]]
        docker_direction = docker_pred.GetDirection()
        docker_pred = sitk.GetArrayFromImage(docker_pred)
        # get spacing for this image
        spacing = np.array(sitk_mask.GetSpacing())[[2,0,1]]
        # reorient if required
        assert(docker_direction==sitk_mask.GetDirection())
        assert(gold_mask.shape==docker_pred.shape)
        if sitk_mask.GetDirection()[-1] == -1:
            gold_mask = np.flip(gold_mask, axis=0).copy()
            gold_mask = np.flip(gold_mask, axis=2).copy()
            docker_pred = np.flip(docker_pred, axis=0).copy()
            docker_pred = np.flip(docker_pred, axis=2).copy()
        # load in pytorch version
        pytorch_pred = np.load(os.path.join(pytorch_dir, test_fname.replace('.nii.gz','.npy').replace('train', 'pred_train')))
        assert(pytorch_pred.shape==docker_pred.shape)
        # get present labels
        labels_present = labels_present_all[test_ind]
        # calculate metrics
        first_oar_idx = 1
        '''
        try:
            assert((pytorch_pred==docker_pred).all())
            print(f"Perfect match for: {test_fname} !")
        except AssertionError:
            print(f"Imperfect match for {test_fname}:")
            print(f"Overlap: {(pytorch_pred==docker_pred).sum() / pytorch_pred.size}")
            input()
        '''
        for organ_idx, organ_num in enumerate(range(first_oar_idx, gold_mask.max()+1)):
            # check if label present in gs, skip if not
            if not labels_present[organ_idx]:
                logger.info(f"{test_fname} missing {organs[organ_idx]}, skipping...")
                continue
            # Need to binarise the masks for the metric computation
            gs = (gold_mask==organ_num).astype(int)
            pred = (docker_pred==organ_num).astype(int)
            # post-processing using scipy.ndimage.label to eliminate extraneous voxels
            '''
            labels, num_features = ndimage.label(input=pred)
            if num_features > 1:
                # disconnected bits present, iterate over them to check which to keep
                # if less than 20% the volume of the primary region, get rid
                primary_vol_threshold = (labels==1).sum() * 0.2
                for feature_label in range(2, num_features+1):
                    if primary_vol_threshold > (labels==feature_label).sum():
                        pred[labels==feature_label] = 0
            '''
            # compute the surface distances
            surface_distances = deepmind_metrics.compute_surface_distances(gs.astype(bool), pred.astype(bool), spacing)
            # compute desired metric
            surface_DSC = deepmind_metrics.compute_surface_dice_at_tolerance(surface_distances, tolerance_mm=1.)
            # store result
            res[0, pat_idx, organ_idx, 0] = dice(gs, pred)
            res[0, pat_idx, organ_idx, 1] = surface_DSC

    # save results
    np.save(os.path.join(output_dir, "full_res_results_grid.npy"), res)

    # Romeo Dunn
    return
if __name__ == '__main__':
    main()
