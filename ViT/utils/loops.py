"""
Main training loops
"""
import os
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm

class FineSegmentation():
    def __init__(self, model, optimizer, train_loader, test_loader, writer, num_epochs, device = "cuda:0"):
        self.device = torch.device(device)
        self.model = model.to(device)
        self.optimizer = optimizer
        self.writer = writer
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', verbose=True)
        self.loss = nn.BCEWithLogitsLoss().to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.best_loss = 10000 #* Init. Loss to large val.
        self.output_path = './logs/'
        os.makedirs(self.output_path, exist_ok=True)

    def forward(self):
        for epoch in range(self.num_epochs + 1):
            print(f'EPOCH: {epoch} / {self.num_epochs}')
            self.writer.epoch = epoch
            self.training(epoch)
            self.validation(epoch)
            self.save_best_model()

    def training(self, epoch, writer_step=20):
        self.model.train()
        self.writer.reset_losses()

        for idx, data in enumerate(tqdm(self.train_loader)):
            inputs = data['inputs'].to(self.device, dtype=torch.float32)
            targets = data['targets'].to(self.device, dtype=torch.float32)

            self.optimizer.zero_grad()
            outputs= self.model(inputs)

            train_loss = self.loss(outputs, targets)

            self.writer.metrics['train_loss'].append(train_loss.item())
            train_loss.backward()
            self.optimizer.step()

            if epoch % writer_step == 0 and idx == 0:
                print('Plotting inputs...')
                self.writer.plot_inputs() #TODO
        print('Train Loss:', np.mean(self.writer.metrics['train_loss']))
        self.writer.add_scalar('Training Loss', np.mean(self.writer.metrics['train_loss']), epoch)

    def validation(self, epoch, writer_step=5):
        self.model.eval()

        with torch.set_grad_enabled(False):
            print('VALIDATION')
            for batch_idx, data in enumerate(self.val_loader):
                inputs = data['inputs'].to(self.device, dtype=torch.float32)
                targets = data['targets'].to(self.device, dtype=torch.float32)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                valid_loss = self.loss(outputs, targets)
                #* Writer
                self.writer.metrics['val_loss'].append(valid_loss.item())
                #* --- PLOT TENSORBOARD ---#
                if epoch % writer_step == 0 and batch_idx == 0:
                    self.writer.plot_segmentation(
                        'Predictions', inputs, outputs['out'], targets=None)
        print('Validation Loss:', np.mean(self.writer.metrics['val_loss']))
        self.writer.add_scalar('Validation Loss', np.mean(self.writer.metrics['val_loss']), epoch)

    def save_best_model(self, model_name='best_model.pt'):
        loss1 = np.mean(self.writer.metrics['val_loss'])
        is_best = loss1 < self.best_loss
        self.best_loss = min(loss1, self.best_loss)
        if is_best:
            print('Saving best model')
            torch.save(self.model.state_dict(),
                       self.output_path + model_name)
