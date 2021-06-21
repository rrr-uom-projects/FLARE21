"""
Main training loops
"""
import os
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from .losses import exp_log_loss, RunningAverage


class FineSegmentation():
    def __init__(self, model, optimizer, train_loader, test_loader, writer, logger,
                    num_epochs, output_path, device = "cuda:0", **kwargs):
        self.device = torch.device(device)
        self.model = model.to(device)
        self.optimizer = optimizer
        self.writer = writer
        self.logger = logger
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', verbose=True)
        self.scaler = torch.cuda.amp.GradScaler()
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.best_loss = 10000 #* Init. Loss to large val.
        self.output_path = output_path
        #* Iters_to_accumulate used to simulate larger batch sizes
        self.iters_to_accumulate = kwargs["iters_to_accumulate"] if "iters_to_accumulate" in kwargs else 4
        self.patience = kwargs["patience"] if "patience" in kwargs else 10 #* For early stopping
        self.num_iterations = 1 #Skip first iteration (no point to accumulating gradients)

    def forward(self):
        #@ Main loop
        for epoch in range(self.num_epochs + 1):
            print(f'EPOCH: {epoch+1} / {self.num_epochs}')
            self.writer.epoch = epoch
            self.training(epoch)
            self.validation(epoch)
            self.save_best_model()

    def training(self, epoch, writer_step=20):
        #@ Training loop
        self.model.train()
        train_losses = RunningAverage() # Resets loss
        for idx, data in enumerate(tqdm(self.train_loader)):
            inputs = data['inputs'].to(self.device, dtype=torch.float32)
            targets = data['targets'].to(self.device, dtype=torch.float32)
            ignore_index = data['ignore_index'].to(self.device)

            with torch.cuda.amp.autocast():
                outputs= self.model(inputs)
                loss = exp_log_loss(outputs, targets, ignore_index)

            #* Update losses
            train_losses.update(loss.item(), self._batch_size(inputs))
            
            loss = loss/self.iters_to_accumulate
            self.scaler.scale(loss).backward()

            #* Step after accumulating gradients
            if self.num_iterations % self.iters_to_accumulate == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                #Log Stats
                self._log_stats('train', train_losses.avg)

            self.num_iterations += 1
            
            if epoch % writer_step == 0 and idx == 0:
                print('Plotting inputs...')
                self.writer.plot_inputs('Input data', inputs, targets=targets) #TODO
        self.writer.add_scalar('Training Loss (Epoch)', train_losses.avg, epoch)

    def validation(self, epoch, writer_step=5):
        #@ Validation loop
        self.model.eval()
        self.logger.info('Validation...')
        val_losses = RunningAverage()
        with torch.set_grad_enabled(False):
            for batch_idx, data in enumerate(self.val_loader):
                inputs = data['inputs'].to(self.device, dtype=torch.float32)
                targets = data['targets'].to(self.device, dtype=torch.float32)
                ignore_index = data['ignore_index'].to(self.device)

                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = exp_log_loss(outputs, targets, ignore_index)
            
                val_losses.update(loss.item(), self._batch_size(inputs))
                self._log_stats('val', val_losses.avg)
                #* --- PLOT TO TENSORBOARD ---#
                if epoch % writer_step == 0 and batch_idx == 0:
                    self.writer.plot_segmentation(
                        'Predictions', inputs, outputs, targets=None)
        self.logger.info('Validation Loss:', val_losses.avg)
        self.writer.add_scalar('Validation Loss (Epoch)', val_losses.avg, epoch)
        return val_losses.avg

    def save_best_model(self, model_name='best_model.pt'):
        #@ Save model w. lowest validation loss
        loss1 = np.mean(self.writer.metrics['val_loss'])
        is_best = loss1 < self.best_loss
        self.best_loss = min(loss1, self.best_loss)
        if is_best:
            print('Saving best model')
            torch.save(self.model.state_dict(),
                       self.output_path + model_name)

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)

    def _log_stats(self, phase, loss_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
        }
        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)
