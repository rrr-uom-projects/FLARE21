import os
import copy
import torch
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
from train import segmenter_Dataset
from models import yolo_transpose_plusplus
from trainer import segmenter_trainer
from utils import k_fold_split_train_val_test, get_logger, get_number_of_learnable_parameters, getFiles, windowLevelNormalize

source_dir = "/data/FLARE21/training_data_256/"
imagedir = os.path.join(source_dir, "scaled_ims/")
maskdir = os.path.join(source_dir, "scaled_masks/")
organs = ["liver", "kidney L", "kidney R", "spleen", "pancreas"]

def dice(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    return (2. * (a*b).sum()) / (a.sum() + b.sum())

def evaluate_model(model, test_inds):
    test_im_fnames = [sorted(getFiles(imagedir))[ind] for ind in test_inds]
    res = np.full(shape=(len(test_im_fnames), 5), fill_value=np.nan)
    # iterate over each testing image
    for pat_idx, (test_fname, test_ind) in enumerate(zip(test_im_fnames, test_inds)):
        # load image and normalise
        ct_im = np.load(os.path.join(imagedir, test_fname))
        ct_im = windowLevelNormalize(ct_im, level=50, window=400)[np.newaxis, np.newaxis] # add dummy batch and channels axes
        # load gold standard segmentation
        gold_mask = np.load(os.path.join(maskdir, test_fname))
        # run forward pass
        prediction = model(torch.tensor(ct_im, dtype=torch.float).to('cuda'))
        # change prediction from one-hot to mask and move back to cpu for metric calculation
        prediction = torch.squeeze(prediction)
        prediction = torch.argmax(prediction, dim=0)
        prediction = prediction.cpu().numpy().astype(int)
        # get present labels
        labels_present = np.load(os.path.join(source_dir, "labels_present.npy"))[test_ind]
        # calculate metrics
        first_oar_idx = 1
        for organ_idx, organ_num in enumerate(range(first_oar_idx, gold_mask.max()+1)):
            # check if label present in gs, skip if not
            if not labels_present[organ_idx]:
                print(f"{test_fname} missing {organs[organ_idx]}, skipping...")
                continue
            # Need to binarise the masks for the metric computation
            gs = (gold_mask==organ_num).astype(int)
            pred = (prediction==organ_num).astype(int)
            # store result
            res[pat_idx, organ_idx] = dice(gs, pred)
    return res

def measure_module_sparsity(module, weight=True, bias=False, use_mask=False):
    num_zeros = 0
    num_elements = 0

    if use_mask == True:
        for buffer_name, buffer in module.named_buffers():
            if "weight_mask" in buffer_name and weight == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
            if "bias_mask" in buffer_name and bias == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
    else:
        for param_name, param in module.named_parameters():
            if "weight" in param_name and weight == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()
            if "bias" in param_name and bias == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()

    sparsity = num_zeros / num_elements
    return num_zeros, num_elements, sparsity

def measure_global_sparsity(model, weight=True, bias=False, conv3d_use_mask=False):
    num_zeros = 0
    num_elements = 0

    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d) or isinstance(module, torch.nn.ConvTranspose3d):
            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=conv3d_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements

    sparsity = num_zeros / num_elements
    return num_zeros, num_elements, sparsity

def iterative_pruning_finetuning(model, train_loader, val_loader, test_inds, label_freq, logger, checkpoint_dir, conv3d_prune_amount=0.4, num_iterations=10):
    for i in range(num_iterations):
        print("Pruning and Finetuning {}/{}".format(i + 1, num_iterations))
        print("Pruning...")

        # Global pruning
        parameters_to_prune = []
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv3d):
                parameters_to_prune.append((module, "weight"))
            if isinstance(module, torch.nn.ConvTranspose3d):
                parameters_to_prune.append((module, "weight"))
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=conv3d_prune_amount)

        res = evaluate_model(model, test_inds)
        print(np.nanmean(res, axis=1))

        num_zeros, num_elements, sparsity = measure_global_sparsity(
            model,
            weight=True,
            bias=False,
            conv3d_use_mask=True)

        print("Global Sparsity:")
        print("{:.2f}".format(sparsity))

        print("Fine-tuning...")
        # Fine tune here
        # Create the optimizer
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.001)

        # Create learning rate adjustment strategy
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15, verbose=True)
        
        # Create model trainer
        trainer = segmenter_trainer(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, device='cuda', train_loader=train_loader, val_loader=val_loader,
                                    label_freq=label_freq, logger=logger, checkpoint_dir=checkpoint_dir, max_num_epochs=50, patience=25, iters_to_accumulate=2)
        trainer.fit()
        
        # re evaluate
        res = evaluate_model(model, test_inds)
        print(np.nanmean(res, axis=1))

        num_zeros, num_elements, sparsity = measure_global_sparsity(
            model,
            weight=True,
            bias=False,
            conv2d_use_mask=True,
            linear_use_mask=False)
        print("Global Sparsity:")
        print("{:.2f}".format(sparsity))

        # save model

    return model

def remove_parameters(model):
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d) or isinstance(module, torch.nn.ConvTranspose3d):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass
    return model

def main():
    fold_num=1
    checkpoint_dir = "/data/FLARE21/models/full_runs/yolo_transpose_plusplus/fold"+str(fold_num)
    # Create the model
    n_classes = 7
    model = yolo_transpose_plusplus(n_classes=n_classes, in_channels=2, p_drop=0)
    model.load_best(checkpoint_dir)

    for param in model.parameters():
        param.requires_grad = True

    logger = get_logger('Pruning')

    # put the model on GPU(s)
    device='cuda'
    model.to(device)

    # allocate ims to train, val and test
    dataset_size = len(sorted(getFiles(imagedir)))
    train_inds, val_inds, test_inds = k_fold_split_train_val_test(dataset_size, fold_num, seed=230597)

    # get label frequencies for weighted loss fns
    label_freq = np.load(os.path.join(source_dir, "label_freq.npy"))
    train_BS = int(3)
    val_BS = int(2)
    train_workers = int(4)
    val_workers = int(2)

    train_data = segmenter_Dataset(imagedir=imagedir, maskdir=maskdir, image_inds=train_inds, n_classes=n_classes, shift_augment=True, rotate_augment=True, scale_augment=True, flip_augment=False)
    train_loader = DataLoader(dataset=train_data, batch_size=train_BS, shuffle=True, pin_memory=False, num_workers=train_workers, worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))
    val_data = segmenter_Dataset(imagedir=imagedir, maskdir=maskdir, image_inds=val_inds, n_classes=n_classes, shift_augment=False, rotate_augment=False, scale_augment=False, flip_augment=False)
    val_loader = DataLoader(dataset=val_data, batch_size=val_BS, shuffle=True, pin_memory=False, num_workers=val_workers, worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))

    num_zeros, num_elements, sparsity = measure_global_sparsity(model)

    print("Iterative Pruning + Fine-Tuning...")

    pruned_model = copy.deepcopy(model)

    iterative_pruning_finetuning(model=pruned_model, train_loader=train_loader, val_loader=val_loader, test_inds=test_inds, label_freq=label_freq, logger=logger, checkpoint_dir=checkpoint_dir,
                                conv3d_prune_amount=0.98, num_iterations=5)

    # Apply mask to the parameters and remove the mask.
    remove_parameters(model=pruned_model)

    res = evaluate_model(pruned_model, test_inds)
    print(np.nanmean(res, axis=1))

    num_zeros, num_elements, sparsity = measure_global_sparsity(pruned_model)

    # save model

if __name__ == "__main__":
    main()