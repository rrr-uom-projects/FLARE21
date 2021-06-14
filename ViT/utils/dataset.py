"""
Should be similar to Ed's custom Dataset in multistage seg

"""
import os
import numpy as np
from torch.utils.data import Dataset

class customDataset(Dataset):
    def __init__(self, image_path, transforms):
        super().__init__()
        self.images = self.load_data(image_path + 'slices/')
        self.masks = self.load_data(image_path + 'masks/')
        self.transforms = transforms
        self.ids = self.images['id']

    @staticmethod
    def load_data(path):
        #* Expects contents of directory to be .npy (ID.npy)
        data_dict = {'slices': [], 'id': []}
        for file in os.listdir(path):
            if file.endswith('.npy'):  # !!
                name = file.split('.')[0]
                data_dict['id'].append(name)
                slice_ = np.load(path + file)
                data_dict['slices'].append(slice_)
        data_dict['slices'] = np.array(data_dict['slices'])
        return data_dict
