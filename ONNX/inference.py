"""
ONNX model inference

"""
import os
import torch
import sys
import numpy as np
sys.path.append('..')
import onnxruntime as ort
import time

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose


from ViT.utils.transforms import PrepareForNet
from roughSeg.utils import k_fold_split_train_val_test, getFiles


img_dir = '/data/FLARE21/training_data/scaled_ims/'  # * path to data
model_filename = './compiled_model.onnx'

test_workers = 2
batch_size=6

#~ Dataset class
class customDataset(Dataset):
    def __init__(self, image_path, transforms, indices, apply_WL, window=400, level=50):
        self.indices = indices
        self.organ_to_idx = ["Background", "Liver",
                             "Kidney L", "Kidney R", "Spleen", "Pancreas"]
        self.names = self.idx_to_names(image_path)
        self.images = self.load_data(image_path, oar=None)
        self.WL_images = self.WL_norm(self.images, window=window, level=level)
        self.transforms = transforms
        self.apply_WL = apply_WL
        self.ignore_oars = np.load(
            "/data/FLARE21/training_data/labels_present.npy")

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
        if torch.is_tensor(index):
            index = index.tolist()
        pid = self.names[index]

        if self.apply_WL:
            img = self.WL_images[index, ..., np.newaxis]
        else:
            img = self.images['slices'][index, ..., np.newaxis]
        if self.transforms:
            augmented = self.transforms({"image": img})
            sample = {'inputs': augmented["image"],
                      'id': pid}
            return sample

        else:
            print('Need some transforms - minimum ToTensor()')
            raise


def to_numpy(x):
    return x.detach().cpu().numpy() if x.requires_grad else x.cpu().numpy()

def main():
    test_transforms = Compose([
        PrepareForNet(), #*Channels first
    ])

    dataset_size = len(getFiles(img_dir))
    train_idx, val_idx, test_idx = k_fold_split_train_val_test(
        dataset_size, fold_num=1, seed=230597) #! Use test set from first fold for now
    
    test_dataset = customDataset(img_dir, test_transforms, indices=test_idx, apply_WL=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=False,
                             num_workers=test_workers, worker_init_fn=lambda _: np.random.seed(
                                 int(torch.initial_seed()) % (2**32-1)))
    #* ONNX inference session

    ort_session = ort.InferenceSession(model_filename)
    print("ONNX Available providers:", ort.get_available_providers())
    print(ort.get_device())
    t = time.time()
    for data in test_loader:
        ort.backend.run(ort_session, to_numpy(data['inputs']))

    # for data in test_loader:
    #     ort_inputs = ort.OrtValue.ortvalue_from_numpy(to_numpy(data['inputs']), 'cuda', 0) #* Place on cuda device id=0
    #     out = ort_session.run(None, ort_inputs)
    #     print(out.shape)
    #     break
    print(f'Execution time: {time.time() - t} for {len(test_idx)} examples.')

if __name__ == '__main__':
    main()