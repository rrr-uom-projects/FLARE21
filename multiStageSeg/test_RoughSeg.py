import torch
import numpy as np
from scipy import ndimage
import os
import time

from models import roughSegmenter
from utils import k_fold_split_train_val_test, get_logger, get_number_of_learnable_parameters, getFiles, windowLevelNormalize
import deepmind_metrics

imagedir = "/data/FLARE21/training_data/scaled_ims/"
maskdir = "/data/FLARE21/training_data/scaled_masks/"
folds = [1]#[1,2,3,4,5]
dataset_size = 92 #len(sorted(getFiles(imagedir)))
all_fnames = sorted(getFiles(imagedir))
spacings = np.load("/data/FLARE21/training_data/spacings_scaled.npy")[:,[2,0,1]]    # change order from (AP,LR,CC) to (CC,AP,LR)

def main():
    # Create logger
    logger = get_logger('roughSeg_testing')

    # Create the model
    model = roughSegmenter(n_classes=6, in_channels=1, p_drop=0)

    # put the model on GPU
    model.to('cuda')

    # setup result grids
    res = np.zeros(shape=(len(folds), dataset_size, 5, 2))

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
            logger.info(f"{test_fname} seg. took {time.time()-t:.2f} seconds")
            # change prediction from one-hot to mask and move back to cpu for metric calculation
            prediction = torch.squeeze(prediction)
            prediction = torch.argmax(prediction, dim=0)
            prediction = prediction.cpu().numpy().astype(int)
            # save output
            #np.save(os.path.join("/data/FLARE21/results/roughSegmenter/test_segs/", 'pred_'+test_fname), prediction)
            # get spacing for this image
            spacing = spacings[test_ind]
            spacing = spacing[2,0,1]
            # calculate metrics
            first_oar_idx = 1
            for organ_idx, organ_num in enumerate(range(first_oar_idx, gold_mask.max()+1)):
                # Need to binarise the masks for the metric computation
                gs = np.zeros(shape=gold_mask.shape)
                pred = np.zeros(shape=prediction.shape)
                gs[(gold_mask==organ_num)] = 1
                pred[(prediction==organ_num)] = 1
                # post-processing using scipy.ndimage.label to eliminate extraneous voxels
                labels, num_features = ndimage.label(input=pred, structure=np.ones((3,3,3)))
                sizes = ndimage.sum(pred, labels, range(num_features+1))
                pred[(labels!=np.argmax(sizes))] = 0
                # compute the surface distances
                surface_distances = deepmind_metrics.compute_surface_distances(gs.astype(bool), pred.astype(bool), spacing)
                # compute desired metric
                hausdorff95 = deepmind_metrics.compute_robust_hausdorff(surface_distances, percent=95.)
                meanDTA = deepmind_metrics.compute_average_surface_distance(surface_distances)
                print(hausdorff95, meanDTA)
                # store result
                res[fdx, pat_idx, organ_idx, 0] = hausdorff95
                res[fdx, pat_idx, organ_idx, 1] = meanDTA

    # save results
    np.save("/data/FLARE21/results/roughSegmenter/results_grid.npy", res)

    # Romeo Dunn
    return

if __name__ == '__main__':
    main()