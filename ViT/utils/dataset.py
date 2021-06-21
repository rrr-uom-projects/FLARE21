"""
Should be similar to Ed's custom Dataset in multistage seg

"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

class customDataset(Dataset):
    def __init__(self, image_path, mask_path, transforms, indices, apply_WL, window=400, level=50):
        self.indices = indices
        self.organ_to_idx = ["Background", "Liver",
                             "Kidney L", "Kidney R", "Spleen", "Pancreas"]
        self.names = self.idx_to_names(image_path)
        self.images = self.load_data(image_path, oar=None)
        self.masks = self.load_data(mask_path, oar='Liver')
        self.WL_images = self.WL_norm(self.images, window=window, level=level)
        self.transforms = transforms
        self.apply_WL = apply_WL
        self.ignore_oars = np.load("/data/FLARE21/training_data/labels_present.npy")
        
    def idx_to_names(self, image_dir):
        return [sorted(self.getFiles(image_dir))[idx] for idx in self.indices]

    def load_data(self, path, oar=None):
        #* Expects contents of directory to be .npy (ID.npy)
        data_dict = {'slices': [], 'id': []}
        for file in os.listdir(path):
            name = file.split('.')[0]
            if file.endswith('.npy') and name in self.names: 
                data_dict['id'].append(name)
                slice_ = np.load(path + file)
                if oar is None:
                    data_dict['slices'].append(slice_)
                else:
                    idx = self.organ_to_idx.index(oar)
                    #* Convert mask to one-hot
                    mask = (idx == slice_[..., None]).astype(int)
                    data_dict['slices'].append(mask)
        data_dict['slices'] = np.array(data_dict['slices'])
        return data_dict

    @staticmethod
    def getFiles(targetdir):
        ls = []
        for fname in os.listdir(targetdir):
            path = os.path.join(targetdir, fname)
            if os.path.isdir(path):
                continue
            ls.append(fname.split('.')[0])
        return ls

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

        if self.apply_WL:
            img = self.WL_images[index, ..., np.newaxis]
        else:
            img = self.images['slices'][index, ..., np.newaxis]
        mask = self.masks['slices'][index]
        if self.transforms:
            augmented = self.transforms({"image":img, "mask":mask})
            sample = {'inputs': augmented["image"], 
            'targets': augmented["mask"],
            'id': pid, 
            'ignore_index': ignore_index}
            return sample

        else:
            print('Need some transforms - minimum ToTensor()') 
            raise
