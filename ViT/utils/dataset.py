"""
Should be similar to Ed's custom Dataset in multistage seg

"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from ...multiStageSeg.utils import getFiles


class customDataset(Dataset):
    def __init__(self, image_path, mask_path, transforms, indices, window=400, level=50):
        self.indices = indices
        self.names = self.idx_2_names(image_path)
        self.images = self.load_data(image_path)
        self.masks = self.load_data(mask_path)
        self.WL_images = self.WL_norm(self.images, window=window, level=level)
        self.transforms = transforms
        self.ignore_oars = np.load("/data/FLARE21/training_data/labels_present.npy")

    def idx_2_names(self, image_dir):
        return [sorted(getFiles(image_dir))[idx] for idx in self.indices]

    def load_data(self, path):
        #* Expects contents of directory to be .npy (ID.npy)
        data_dict = {'slices': []}
        for file in os.listdir(path):
            name = file.split('.')[0]
            if file.endswith('.npy') and name in self.names: 
                data_dict['id'].append(name)
                slice_ = np.load(path + file)
                data_dict['slices'].append(slice_)
        data_dict['slices'] = np.array(data_dict['slices'])
        return data_dict

    @staticmethod
    def WL_norm(data, window, level):
        minval = level - window/2
        maxval = level + window/2
        wld = np.clip(data['slices'], minval, maxval)
        wld -= minval
        wld /= window
        return wld

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        if  torch.is_tensor(index):
            index = index.tolist()
        pid = self.names[index]
        ignore_index = self.ignore_oars[index]

        if self.normalise:
            img = self.WL_images[index, ..., np.newaxis]
        else:
            img = self.images[index, ..., np.newaxis]
        mask = self.masks['slices'][index]
        #* Convert mask to one-hot
        mask = (np.arange(6) == mask[..., None]).astype(int)

        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            sample = {'inputs': augmented["image"], 
            'mask': augmented["mask"],
            'id': pid, 
            'ignore_index': ignore_index}
            return sample

        else:
            print('Need some transforms - minimum ToTensor()') 
            raise
