"""
Main script for training fine-scale segmentation
"""
import torch
import cv2

from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import torch.optim as optim

from model.model import SegmentationModel
from utils.transforms import Resize, NormalizeImage, PrepareForNet
from utils.dataset import customDataset
from utils.writer import customWriter
from utils.loops import FineSegmentation
from ..multiStageSeg.utils


train_path = ''
test_path = ''

batch_size = 16
num_epochs=500
lr = 3e-4
net_width = net_height = 480  # !! This will need adapting



def main():
    train_transforms = Compose([
        #* From DTP - Resize = RandomCrops I think
        Resize(net_width, net_height,
            resize_target=None, keep_aspect_ratio=True, ensure_multiple_of=32,
            resize_method="minimal", image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        PrepareForNet(),
    ])
    test_transforms = Compose([
        Resize(net_width, net_height,
                resize_target=None, keep_aspect_ratio=True, ensure_multiple_of=32,
                resize_method="minimal", image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.5, 0.5, 0.5], std=[
                        0.5, 0.5, 0.5]),
        PrepareForNet(),
    ]) 

    #* Prepare data
    train_dataset = customDataset(train_path, train_transforms)
    test_dataset = customDataset(test_path, test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    #* Prepare model 
    model = SegmentationModel(num_classes=1, path=None,
                              backbone="vitb_rn50_384")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    writer = customWriter(batch_size)

    #* Train Object
    seg = FineSegmentation(model, optimizer, train_loader, 
                test_loader, writer, num_epochs=num_epochs, device="cuda:0")

    seg.forward()


if __name__ == '__main__':
    main()
