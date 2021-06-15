import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import roughSeg.utils as utils
import time
from scipy.ndimage import center_of_mass

#####################################################################################################
############################################ loss fns ###############################################
#####################################################################################################

def multiclass_simple_dice_loss(prediction, mask, ignore_index, useWeights=True):
    # prediction is (B,C,H,W,D)
    # target_mask NEEDS to be one-hot encoded [(B,C,H,W,D) <-- one-hot encoded]
    # Use weights for highly unbalanced classes
    num_classes = prediction.size()[1]
    if useWeights:
        label_count = np.array([358074923, 14152955, 1698684, 1643118,  2153875, 812381])
        class_weights = np.power(label_count.sum() / label_count, 1/3)
        class_weights /= np.sum(class_weights)
    else:
        class_weights = np.full((num_classes), 1/num_classes)   # flat normalised
    smooth = 1.
    loss = 0.
    dice_pred = F.softmax(prediction, dim=1)

    if (ignore_index==False).any():
        # we are missing gold standard masks for some structures
        # change all predicted pixels of the missing structure to background -> 0
        # that way the loss will be 0 in regions of missing gold standard labels
        ablation_mask = torch.zeros_like(dice_pred, dtype=bool)
        missing_inds = torch.where(~ignore_index)
        for imdx, sdx in zip(missing_inds[0], missing_inds[1]):
            ablation_mask[imdx, sdx+1] = True
        dice_pred = dice_pred.masked_fill(ablation_mask, 0)

    for c in range(num_classes):
        pred_flat = dice_pred[:,c].reshape(-1)
        mask_flat = mask[:,c].reshape(-1)
        intersection = (pred_flat*mask_flat).sum()
        w = class_weights[c]
        # numerator
        num = 2. * intersection + smooth
        # denominator
        denom = pred_flat.sum() + mask_flat.sum() + smooth
        # loss
        loss += w*(1 - (num/denom))
    return loss

def exp_log_loss(prediction, mask, ignore_index, device='cuda'):
    """
    paper: 3D Segmentation with Exponential Logarithmic Loss for Highly Unbalanced Object Sizes
    https://arxiv.org/pdf/1809.00076.pdf
    
    ### ! needs raw scores from the network ! ###
    """
    gamma = 0.3
    # Dice loss
    num_classes = prediction.size()[1]
    smooth = 1.
    dice_pred = F.softmax(prediction, dim=1)
    if (ignore_index==False).any():
        # we are missing gold standard masks for some structures
        # change all predicted pixels of the missing structure to background -> 0
        # that way the loss will be 0 in regions of missing gold standard labels
        ablation_mask = torch.zeros_like(dice_pred, dtype=bool)
        missing_inds = torch.where(~ignore_index)
        for imdx, sdx in zip(missing_inds[0], missing_inds[1]):
            np.save("/data/FLARE21/training_data/preproc.npy", np.argmax(dice_pred[imdx].clone().detach().cpu().numpy(), axis=0))
            ablation_mask[imdx, sdx+1] = True
        dice_pred = dice_pred.masked_fill(ablation_mask, 0)
        np.save("/data/FLARE21/training_data/postproc.npy", np.argmax(dice_pred[imdx].clone().detach().cpu().numpy(), axis=0))
        np.save("/data/FLARE21/training_data/maskproc.npy", np.argmax(mask[imdx].clone().detach().cpu().numpy(), axis=3))

    pred_flat = dice_pred.view(-1, num_classes)
    mask_flat = mask.view(-1, num_classes)
    intersection = (pred_flat*mask_flat).sum(dim=0)
    # numerator
    num = 2. * intersection + smooth
    # denominator
    denom = pred_flat.sum(dim=0) + mask_flat.sum(dim=0) + smooth        
    # calculate dice
    dice = num / denom
    dice_loss = torch.mean(torch.pow(torch.clamp(-torch.log(dice), min=1e-6), gamma))

    # XE loss
    label_freq = np.array([358074923, 14152955, 1698684, 1643118,  2153875, 812381])   # background, liver, kidney L, kidney R, spleen, pancreas
    
    class_weights = np.power(np.full((num_classes), label_freq.sum()) / label_freq, 0.5)
    xe_pred = F.log_softmax(prediction, dim=1)
    if (ignore_index==False).any():
        # same again - missing inds retained from above
        ablation_mask = torch.zeros_like(xe_pred, dtype=bool)
        for imdx, sdx in zip(missing_inds[0], missing_inds[1]):
            ablation_mask[imdx, sdx+1] = True
        xe_pred = xe_pred.masked_fill(ablation_mask, 0)
    mask = torch.argmax(mask, dim=1)
    xe_loss = torch.mean(torch.pow(torch.clamp(torch.nn.NLLLoss(weight=torch.FloatTensor(class_weights).to(device), reduction='none')(xe_pred, mask), min=1e-6), gamma))

    w_dice = 0.5
    w_xe = 0.5
    return (w_dice*dice_loss) + (w_xe*xe_loss)

#####################################################################################################
############################################ trainer ################################################
#####################################################################################################

class segmenter_trainer:
    def __init__(self, model, optimizer, lr_scheduler, device, train_loader, val_loader, logger, checkpoint_dir, max_num_epochs=100,
                num_iterations=1, num_epoch=0, patience=10, iters_to_accumulate=4, best_eval_score=None, eval_score_higher_is_better=False):
        self.logger = logger
        self.logger.info(model)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.eval_score_higher_is_better = eval_score_higher_is_better
        # initialize the best_eval_score
        if not best_eval_score:
            if eval_score_higher_is_better:
                self.best_eval_score = float('-inf')
            else:
                self.best_eval_score = float('+inf')
        else:
            self.best_eval_score = best_eval_score
        self.patience = patience
        self.epochs_since_improvement = 0
        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))
        self.fig_dir = os.path.join(checkpoint_dir, 'figs')
        try:
            os.mkdir(self.fig_dir)
        except OSError:
            pass
        self.num_iterations = num_iterations
        self.iters_to_accumulate = iters_to_accumulate
        self.num_epoch = num_epoch
        self.epsilon = 1e-6
        self.scaler = torch.cuda.amp.GradScaler()

    def fit(self):
        self._save_init_state()
        for _ in range(self.num_epoch, self.max_num_epochs):
            # train for one epoch
            t = time.time()
            should_terminate = self.train(self.train_loader)
            print("Epoch trained in " + str(int(time.time()-t)) + " seconds.")
            if should_terminate:
                print("Hit termination condition...")
                break
            self.num_epoch += 1
        self.writer.close()
        return self.num_iterations, self.best_eval_score

    def train(self, train_loader):
        """Trains the model for 1 epoch.
        Args:
            train_loader (torch.utils.data.DataLoader): training data loader
        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        train_losses = utils.RunningAverage()
        improved = False        # for early stopping
        self.model.train()      # set the model in training mode
        for batch_idx, sample in enumerate(train_loader):
            self.logger.info(f'Training iteration {self.num_iterations}. Batch {batch_idx + 1}. Epoch [{self.num_epoch + 1}/{self.max_num_epochs}]')
            ct_im = sample['ct_im'].type(torch.HalfTensor)
            mask = sample['mask'].type(torch.HalfTensor)
            ignore_index = sample['ignore_index']

            # send tensors to GPU
            ct_im = ct_im.to(self.device)
            mask = mask.to(self.device)
            ignore_index= ignore_index.to(self.device)
            
            # forward
            output, loss = self._forward_pass(ct_im, mask, ignore_index)
            train_losses.update(loss.item(), self._batch_size(ct_im))
            
            # compute gradients and update parameters
            # simulate larger batch sizes using gradient accumulation
            loss = loss / self.iters_to_accumulate

            # Native AMP training step
            self.scaler.scale(loss).backward()
            
            # Every iters_to_accumulate, call step() and reset gradients:
            if self.num_iterations % self.iters_to_accumulate == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                # log stats
                self.logger.info(f'Training stats. Loss: {train_losses.avg}')
                self._log_stats('train', train_losses.avg)
            
            self.num_iterations += 1

        # evaluate on validation set
        self.model.eval()
        eval_score = self.validate()

        # adjust learning rate if necessary
        self.scheduler.step(eval_score)

        # log current learning rate in tensorboard
        self._log_lr()

        # remember best validation metric
        improved = True if self._is_best_eval_score(eval_score) else False
        
        # save checkpoint
        self._save_checkpoint(improved)

        # implement early stopping here
        if not improved:
            self.epochs_since_improvement += 1
        if(self.epochs_since_improvement > self.patience):  # Model has not improved for certain number of epochs
            self.logger.info(
                    f'Model not improved for {self.patience} epochs. Finishing training...')
            return True
        return False    # Continue training...
        

    def validate(self):
        self.logger.info('Validating...')
        val_losses = utils.RunningAverage()
        with torch.no_grad():
            which_to_show = np.random.randint(0, self.val_loader.batch_size)    # show a random example from a batch
            for batch_idx, sample in enumerate(self.val_loader):
                self.logger.info(f'Validation iteration {batch_idx + 1}')
                ct_im = sample['ct_im'].type(torch.HalfTensor)
                mask = sample['mask'].type(torch.HalfTensor)
                ignore_index = sample['ignore_index']
                
                # send tensors to GPU
                ct_im = ct_im.to(self.device)
                mask = mask.to(self.device)
                ignore_index= ignore_index.to(self.device)
                
                output, loss = self._forward_pass(ct_im, mask, ignore_index)
                val_losses.update(loss.item(), self._batch_size(ct_im))

                if (batch_idx == 0) and (ignore_index==True).all() and ((self.num_epoch < 50) or (self.num_epoch < 100 and not self.num_epoch%5) or (self.num_epoch < 500 and not self.num_epoch%25) or (not self.num_epoch%100)):                   
                    # transferring between the gpu and cpu with .cpu() is really inefficient
                    # -> only transfer slices for plotting not entire volumes (& only plot every so often ... ^ what this mess up here is doing)
                    # plot ims -> - $tensorboard --logdir=MODEL_DIRECTORY --port=6006 --bind_all --samples_per_plugin="images=0"

                    # unfortunately have to transfer entire volumes here, needs a np array to use scipy's center_of_mass
                    mask = torch.argmax(mask, dim=1)[which_to_show].cpu().numpy()
                    output = torch.argmax(output, dim=1)[which_to_show].cpu().numpy()

                    # CoM of targets plots
                    # coronal view of the Liver plot
                    sdx = 1
                    coords = self.find_coords(mask, sdx)
                    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)
                    coronal_slice = ct_im[which_to_show, 0, :, coords[1]].cpu().numpy().astype(float)     # <-- batch_num, contrast_channel, ax_dim(:), coronal_slice
                    ax0.imshow(np.flip(coronal_slice, axis=0), aspect=2.5, cmap='Greys_r', vmin=0, vmax=1)
                    coronal_slice = mask[:, coords[1]].astype(float)
                    ax1.imshow(np.flip(coronal_slice, axis=0), aspect=2.5, cmap='nipy_spectral', vmin=0, vmax=5)
                    coronal_slice = output[:, coords[1]].astype(float)
                    ax2.imshow(np.flip(coronal_slice, axis=0), aspect=2.5, cmap='nipy_spectral', vmin=0, vmax=5)
                    self.writer.add_figure(tag='Liver_pred', figure=fig, global_step=self.num_epoch)
                    fig.savefig(os.path.join(self.fig_dir, 'Liver_pred_'+str(self.num_epoch)+'.png'))
                    
                    # axial view of the kidneys (centre kidney L for simplicity)
                    sdx = 2
                    coords = self.find_coords(mask, sdx, 3)
                    fig2, (ax3, ax4, ax5) = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)
                    ax_slice = ct_im[which_to_show, 0, coords[0]].cpu().numpy().astype(float)             # <-- batch_num, contrast_channel, ax_slice
                    ax3.imshow(ax_slice, aspect=1.0, cmap='Greys_r')
                    ax_slice = mask[coords[0]].astype(float)
                    ax4.imshow(ax_slice, aspect=1.0, cmap='nipy_spectral', vmin=0, vmax=5)
                    ax_slice = output[coords[0]].astype(float)
                    ax5.imshow(ax_slice, aspect=1.0, cmap='nipy_spectral', vmin=0, vmax=5)
                    self.writer.add_figure(tag='Kidneys_pred', figure=fig2, global_step=self.num_epoch)
                    fig2.savefig(os.path.join(self.fig_dir, 'Kidneys_pred_'+str(self.num_epoch)+'.png'))
                    
                    # sagittal view of the spleen
                    sdx = 4
                    coords = self.find_coords(mask, sdx)
                    fig3, (ax6, ax7, ax8) = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)
                    sag_slice = ct_im[which_to_show, 0, :, :, coords[2]].cpu().numpy().astype(float)            # <-- batch_num, contrast_channel, sag_slice
                    ax6.imshow(np.flip(sag_slice, axis=0), aspect=2.5, cmap='Greys_r')
                    sag_slice = mask[:, :, coords[2]].astype(float)
                    ax7.imshow(np.flip(sag_slice, axis=0), aspect=2.5, cmap='nipy_spectral', vmin=0, vmax=5)
                    sag_slice = output[:, :, coords[2]].astype(float)
                    ax8.imshow(np.flip(sag_slice, axis=0), aspect=2.5, cmap='nipy_spectral', vmin=0, vmax=5)
                    self.writer.add_figure(tag='Spleen_pred', figure=fig3, global_step=self.num_epoch)
                    fig3.savefig(os.path.join(self.fig_dir, 'Spleen_pred_'+str(self.num_epoch)+'.png'))
                    
                    # axial view of the pancreas
                    sdx = 5
                    coords = self.find_coords(mask, sdx)
                    fig4, (ax9, ax10, ax11) = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)
                    ax_slice = ct_im[which_to_show, 0, coords[0]].cpu().numpy().astype(float)             # <-- batch_num, contrast_channel, ax_slice
                    ax9.imshow(ax_slice, aspect=1.0, cmap='Greys_r')
                    ax_slice = mask[coords[0]].astype(float)
                    ax10.imshow(ax_slice, aspect=1.0, cmap='nipy_spectral', vmin=0, vmax=5)
                    ax_slice = output[coords[0]].astype(float)
                    ax11.imshow(ax_slice, aspect=1.0, cmap='nipy_spectral', vmin=0, vmax=5)
                    self.writer.add_figure(tag='Pancreas_pred', figure=fig4, global_step=self.num_epoch)
                    fig4.savefig(os.path.join(self.fig_dir, 'Pancreas_pred_'+str(self.num_epoch)+'.png'))
                
            self._log_stats('val', val_losses.avg)
            self.logger.info(f'Validation finished. Loss: {val_losses.avg}')
            return val_losses.avg

    def _forward_pass(self, ct_im, mask, ignore_index):
        with torch.cuda.amp.autocast():
            # forward pass
            output = self.model(ct_im)
            # use exp_log_loss
            #loss = exp_log_loss(output, mask, ignore_index)
            # or use simpler multi-class weighted dice loss
            loss = multiclass_simple_dice_loss(output, mask, ignore_index)
            return output, loss

    def find_coords(self, mask, sdx, sdx2=None):
        if sdx2:
            coords = center_of_mass(np.logical_or(mask == sdx, mask == sdx2))
        else:
            coords = center_of_mass(mask == sdx)
        return np.round(coords).astype(int)

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score
        if is_best:
            self.logger.info(f'Saving new best evaluation metric: {eval_score}')
            self._log_new_best(eval_score)
            self.best_eval_score = eval_score
            self.epochs_since_improvement = 0
        return is_best

    def _save_init_state(self):
        state = {'model_state_dict': self.model.state_dict()}
        init_state_path = os.path.join(self.checkpoint_dir, 'initial_state.pytorch')
        self.logger.info(f"Saving initial state to '{init_state_path}'")
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        torch.save(state, init_state_path)

    def _save_checkpoint(self, is_best):
        utils.save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': self.model.state_dict(),
            'best_eval_score': self.best_eval_score,
            'eval_score_higher_is_better': self.eval_score_higher_is_better,
            #'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs,
        }, is_best, checkpoint_dir=self.checkpoint_dir,
            logger=self.logger)

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_new_best(self, eval_score):
        self.writer.add_scalar('best_val_loss', eval_score, self.num_iterations)

    def _log_stats(self, phase, loss_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
        }
        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        self.logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            #self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations) #not sure what this is 

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)