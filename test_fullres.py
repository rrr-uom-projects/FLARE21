import torch
import numpy as np
from scipy import ndimage
import os
import time
import SimpleITK as sitk
from skimage.transform import resize
import nvgpu
from multiprocessing import Process, Value

from model_archive import yolo_segmenter
from models import superres_segmenter, fullRes_segmenter, yolo_transpose_plusplus, tiny_segmenter, tiny_inference_segmenter, tiny_attention_segmenter, nano_segmenter
# light_segmenter, bottleneck_yolo_segmenter, asymmetric_yolo_segmenter, asym_bottleneck_yolo_segmenter, 
# bridged_yolo_segmenter, yolo_transpose, yolo_transpose_plusplus, ytp_learnableWL
from roughSeg.utils import k_fold_split_train_val_test, get_logger, getFiles, windowLevelNormalize
import roughSeg.deepmind_metrics as deepmind_metrics

source_dir = "/data/FLARE21/training_data_192_sameKidneys/"
input_dir = "/data/FLARE21/training_data/TrainingImg/"
mask_dir = "/data/FLARE21/training_data/TrainingMask/"
output_dir = "/data/FLARE21/results/full_runs/nano_segmenter_192_1mm/"
input_size = (96,192,192)
folds = [1,2,3,4,5]
organs = ["liver", "kidneys", "spleen", "pancreas"]
base_vram = nvgpu.gpu_info()[0]['mem_used']

def dice(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    return (2. * (a*b).sum()) / (a.sum() + b.sum())

def gpu_mem(maxM):
    while True:
        newM = nvgpu.gpu_info()[0]['mem_used']
        if newM > maxM.value:
            maxM.value = newM

def main():
    # Create logger
    logger = get_logger('fullRes_testing')

    # track gpu memory
    maxMem = Value('i', 0)
    p = Process(target=gpu_mem, args=(maxMem,))
    p.start()

    # get stuff
    imagedir = os.path.join(source_dir, "scaled_ims/")
    dataset_size = len(sorted(getFiles(imagedir))) # 72
    all_fnames = sorted(getFiles(imagedir))
    spacings = np.load(os.path.join(source_dir, "spacings_scaled.npy"))[:,[2,0,1]]    # change order from (AP,LR,CC) to (CC,AP,LR)
    labels_present_all = np.load(os.path.join(source_dir, "labels_present.npy"))
    try:
        os.mkdir(output_dir)
    except OSError:
        pass

    # Create the model
    model = nano_segmenter(n_classes=6, in_channels=2, p_drop=0) #, initial_levels=[1,1,1], initial_windows=[1,1,1]

    # put the model on GPU
    model.to('cuda')

    # setup result grids
    _, _, dummy_test_inds = k_fold_split_train_val_test(dataset_size, fold_num=1, seed=230597)
    res = np.full(shape=(len(folds), len(dummy_test_inds), 4, 2), fill_value=np.nan)

    # iterate over folds
    for fdx, fold_num in enumerate(folds):
        # get checkpoint dir
        checkpoint_dir = f"/data/FLARE21/models/full_runs/nano_segmenter_192/fold{fold_num}/"

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
            t = time.time()
            sitk_image = sitk.ReadImage(os.path.join(input_dir, test_fname.replace('.npy','_0000.nii.gz')))
            ct_im = sitk.GetArrayFromImage(sitk_image)
            # load gold standard segmentation in full resolution
            sitk_mask = sitk.ReadImage(os.path.join(mask_dir, test_fname.replace('.npy','.nii.gz')))
            gold_mask = sitk.GetArrayFromImage(sitk_mask).astype(int)
            # reorient if required
            if sitk_mask.GetDirection()[-1] == -1:
                gold_mask = np.flip(gold_mask, axis=0).copy()
                ct_im = np.flip(ct_im, axis=0).copy()
                gold_mask = np.flip(gold_mask, axis=2).copy() 
                ct_im = np.flip(ct_im, axis=2).copy()
            logger.info(f"Image loading took {time.time()-t:.4f} seconds")
            t = time.time()
            ct_im = resize(ct_im, output_shape=input_size, order=3, anti_aliasing=True, preserve_range=True)
            logger.info(f"Image downsampling took {time.time()-t:.4f} seconds")
            # preprocessing
            ct_im = np.clip(ct_im, -1024, 2000)
            #ct_im = windowLevelNormalize(ct_im, level=50, window=400)[np.newaxis, np.newaxis] # add dummy batch and channels axes
            ct_im2 = np.zeros(shape=(2,) + ct_im.shape)
            ct_im2[0] = windowLevelNormalize(ct_im, level=50, window=400)   # abdomen "soft tissues"
            ct_im2[1] = windowLevelNormalize(ct_im, level=60, window=100)   # pancreas
            ct_im = ct_im2[np.newaxis].copy() # add dummy batch axis
            # run forward pass
            t = time.time()
            #prediction = model(torch.tensor(ct_im, dtype=torch.float).to('cuda'), gold_mask.shape)
            prediction = model(torch.tensor(ct_im, dtype=torch.float).to('cuda'))
            logger.info(f"{test_fname} inference took {time.time()-t:.4f} seconds")
            # change prediction from one-hot to mask and move back to cpu for metric calculation
            prediction = torch.squeeze(prediction)
            prediction = torch.argmax(prediction, dim=0)
            prediction = prediction.cpu().numpy().astype(int)
            # drop the body and label the kidneys together          # OAR labels : 1 - Body, 2 - Liver, 3 - Kidney L, 4 - Kidney R, 5 - Spleen, 6 - Pancreas
            prediction -= 1                                         # -> OAR labels : 0 - Body, 1 - Liver, 2 - Kidney L, 3 - Kidney R, 4 - Spleen, 5 - Pancreas
            print("WARNING: assuming model trained with kidneys as same label...")
            #prediction[prediction >= 3] -= 1                        # -> OAR labels : 0 - Body, 1 - Liver, 2 - Kidneys, 3 - Spleen, 4 - Pancreas
            prediction = np.clip(prediction, 0, prediction.max())   # -> OAR labels : 0 - Background, 1 - Liver, 2 - Kidneys, 3 - Spleen, 4 - Pancreas
            # rescale the prediction to match the full-resolution mask
            t = time.time()
            prediction = np.round(resize(prediction, output_shape=gold_mask.shape, order=0, anti_aliasing=False, preserve_range=True)).astype(np.uint8)
            logger.info(f"Image upsampling took {time.time()-t:.4f} seconds")
            # save output
            try:
                os.mkdir(os.path.join(output_dir, "full_res_test_segs/"))
            except OSError:
                pass
            np.save(os.path.join(output_dir, "full_res_test_segs/", 'pred_'+test_fname), prediction)
            # get spacing for this image
            spacing = np.array(sitk_mask.GetSpacing())[[2,0,1]]
            print(spacing)
            t = time.time()
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
                res[fdx, pat_idx, organ_idx, 0] = dice(gs, pred)
                res[fdx, pat_idx, organ_idx, 1] = surface_DSC
            logger.info(f"Seg processing took {time.time()-t:.4f} seconds")

    # save results
    np.save(os.path.join(output_dir, "full_res_results_grid.npy"), res)

    # get maximum gpu memory consumption
    print(f"Maximum VRAM consumed: {maxMem.value-base_vram}MB")
    np.save(os.path.join(output_dir, "max_vram.npy"), np.array(maxMem.value-base_vram))
    p.terminate()

    # Romeo Dunn
    return

if __name__ == '__main__':
    main()