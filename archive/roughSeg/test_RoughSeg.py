import torch
import numpy as np
from scipy import ndimage
import os
import time

from models import roughSegmenter_deeper
from utils import k_fold_split_train_val_test, get_logger, get_number_of_learnable_parameters, getFiles, windowLevelNormalize
import deepmind_metrics

imagedir = "/data/FLARE21/training_data/scaled_ims/"
maskdir = "/data/FLARE21/training_data/scaled_masks/"
folds = [2]#[1,2,3,4,5]
dataset_size = len(sorted(getFiles(imagedir)))
all_fnames = sorted(getFiles(imagedir))
spacings = np.load("/data/FLARE21/training_data/spacings_scaled.npy")[:,[2,0,1]]    # change order from (AP,LR,CC) to (CC,AP,LR)
labels_present_all = np.load("/data/FLARE21/training_data/labels_present.npy")
organs = ["liver", "kidney L", "kidney R", "spleen", "pancreas"]

def dice(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    return (2. * (a*b).sum()) / (a.sum() + b.sum())

def main():
    # Create logger
    logger = get_logger('roughSeg_testing')

    # Create the model
    model = roughSegmenter_deeper(n_classes=6, in_channels=1, p_drop=0)

    # put the model on GPU
    model.to('cuda')

    # setup result grids
    res = np.full(shape=(len(folds), dataset_size // 5, 5, 2), fill_value=np.nan)

    # iterate over folds
    for fdx, fold_num in enumerate(folds):
        # get checkpoint dir
        checkpoint_dir = f"/data/FLARE21/models/roughSegmenter/fold{fold_num}/"

        # load in the best model version
        model.load_best(checkpoint_dir, logger)
        for param in model.parameters():
            param.requires_grad = False

        # fix model
        model.eval()

        # allocate ims to train, val and test
        train_inds, val_inds, test_inds = k_fold_split_train_val_test(dataset_size, fold_num=fold_num, seed=230597)

        # get test fnames
        test_im_fnames = [all_fnames[ind] for ind in test_inds]

        # iterate over each testing image
        for pat_idx, (test_fname, test_ind) in enumerate(zip(test_im_fnames, test_inds)):
            # load image and normalise
            ct_im = np.load(os.path.join(imagedir, test_fname))
            ct_im = windowLevelNormalize(ct_im, level=50, window=400)[np.newaxis, np.newaxis] # add dummy batch and channels axes
            # load gold standard segmentation
            gold_mask = np.load(os.path.join(maskdir, test_fname))
            # run forward pass
            t = time.time()
            prediction = model(torch.tensor(ct_im, dtype=torch.float).to('cuda'))
            logger.info(f"{test_fname} seg. took {time.time()-t:.4f} seconds")
            # change prediction from one-hot to mask and move back to cpu for metric calculation
            prediction = torch.squeeze(prediction)
            prediction = torch.argmax(prediction, dim=0)
            prediction = prediction.cpu().numpy().astype(int)
            # save output
            np.save(os.path.join("/data/FLARE21/results/roughSegmenter/test_segs/", 'pred_'+test_fname), prediction)
            # get spacing for this image
            spacing = spacings[test_ind]
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
                gs = np.zeros(shape=gold_mask.shape)
                pred = np.zeros(shape=prediction.shape)
                gs[(gold_mask==organ_num)] = 1
                pred[(prediction==organ_num)] = 1
                # post-processing using scipy.ndimage.label to eliminate extraneous voxels
                labels, num_features = ndimage.label(input=pred, structure=np.ones((3,3,3)))
                sizes = ndimage.sum(pred, labels, range(num_features+1))
                pred[(labels!=np.argmax(sizes))] = 0
                # complete extra post processing for the weird right kidney behaviour -> need to figure out whats going on here?
                if organ_idx == 2:
                    pred[-3:] = 0   # Weird massive activation for the right kidney at the cottom for every model? look into this...
                # compute the surface distances
                surface_distances = deepmind_metrics.compute_surface_distances(gs.astype(bool), pred.astype(bool), spacing)
                # compute desired metric
                surface_DSC = deepmind_metrics.compute_surface_dice_at_tolerance(surface_distances, tolerance_mm=5.)
                # store result
                res[fdx, pat_idx, organ_idx, 0] = dice(gs, pred)
                res[fdx, pat_idx, organ_idx, 1] = surface_DSC

    # save results
    np.save("/data/FLARE21/results/roughSegmenter/results_grid.npy", res)

    # Romeo Dunn
    return

if __name__ == '__main__':
    main()