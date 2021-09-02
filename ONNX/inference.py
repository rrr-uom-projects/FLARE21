"""
ONNX model inference

"""
import albumentations as A
import os
import torch
import sys
import numpy as np
sys.path.append('..')
import onnxruntime as ort
import time
from multiprocessing import Pool
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch.transforms import ToTensor

from utils import k_fold_split_train_val_test, getFiles
from train import segmenter_Dataset

parser = ArgumentParser(prog="Run ONNX inference on test set")
parser.add_argument("img_dir", help="Path to test images", type=str)
parser.add_argument("model_path", help="Path to ONNX model", type=str)
# parser.add_argument("outsize_path", help="Path to CSV file with output sizes", type=str)
args = parser.parse_args()

test_workers = 1
batch_size = 1


#~ Dataset class
class customDataset(Dataset):
    def __init__(self, image_path, out_size_file, transforms, indices, apply_WL, window=[400,100], level=[50,60]):
        self.indices = indices
        self.image_path = image_path
        self.out_size = np.load(out_size_file)[indices]
        self.organ_to_idx = ["Background", "Liver", "Kidney", "Spleen", "Pancreas"]
        self.names = self.idx_to_names(image_path)
        self.availableImages = [sorted(getFiles(image_path))[ind] for ind in indices]
        #self.availableImages.remove('train_079.npy')
        self.transforms = transforms
        self.window = window
        self.level = level
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
                try:
                    slice_ = np.load(path + file)
                except ValueError:
                    print(name)
                    continue
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
    def WL_norm(img, window, level):
        minval = level - window/2
        maxval = level + window/2
        wld = np.clip(img, minval, maxval)
        wld -= minval
        wld /= window
        return wld

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        pid = self.names[index]
        imageToUse = self.availableImages[index]
        img = np.load(os.path.join(self.image_path, imageToUse))#[..., np.newaxis]
        if self.apply_WL:
            if type(self.window) is list and type(self.level) is list:
                arr = []
                for i in range(len(self.window)):
                    arr.append(self.WL_norm(img, self.window[i], self.level[i]))
                img = np.stack(arr, axis=-1)
            else:
                img = self.WL_norm(img, self.window, self.level)[..., np.newaxis]
        else:
            img = img[..., np.newaxis]
        if self.transforms:
            print(img.shape)
            augmented = self.transforms(image=img)
            print(augmented["image"].shape)
            sample = {'inputs': augmented["image"],
                      'out_size': (self.out_size[index], 512, 512),
                      'id': pid}
            return sample

        else:
            print('Need some transforms - minimum ToTensor()')
            raise

def sigmoid(x):
    return 1/(1+np.exp(-x))

def to_numpy(x):
    return x.detach().cpu().numpy() if x.requires_grad else x.cpu().numpy()

def main():
    test_transforms = A.Compose([
        ToTensor(), #*Channels first
    ])

    dataset_size = len(getFiles(args.img_dir))
    _, _, test_idx = k_fold_split_train_val_test(
        dataset_size, fold_num=1, seed=230597) #! Use test set from first fold for now
    test_dataset = customDataset(args.img_dir, args.outsize_path, test_transforms, window=[400, 100], level=[50, 60], indices=test_idx, apply_WL=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    print("ONNX Available providers:", ort.get_available_providers())
    print(ort.get_device())
    #* ONNX inference session
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL # ! or ORT_SEQUENTIAL
    sess_options.optimized_model_filepath = "./optimized_model.onnx"
    sess_options.log_severity_level = 0
    sess_options.enable_profiling = False
    sess_options.inter_op_num_threads = os.cpu_count() - 1 
    sess_options.intra_op_num_threads = os.cpu_count() - 1
    print(args.model_path)
    ort_session = ort.InferenceSession(
        args.model_path, sess_options=sess_options)
    
    t = time.time()
    output_dict = {}
    for data in test_loader:
        #out_size = [int(x) for x in data['out_size']]
        #inputs = [data['inputs'], *(torch.tensor(x) for x in out_size)]
        print(data['id'])
        inputs = [data['inputs']]
        ort_inputs = {key.name: to_numpy(x) for key, x in zip(ort_session.get_inputs(), inputs)}
        outputs = np.array(ort_session.run(None, ort_inputs))
        output_dict[data['id'][0]] = np.argmax(
            np.squeeze(outputs), axis=0).astype(np.int8)

    print(f'Execution time: {time.time() - t} for {len(test_idx)} examples.')
    for key, val in output_dict.items():
        np.save(
            f'/data/FLARE21/models/full_runs/nano_segmenter_192/fold1/outputs/{key}.npy', val)
    

if __name__ == '__main__':
    main()
