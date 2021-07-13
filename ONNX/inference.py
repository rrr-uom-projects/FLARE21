"""
ONNX model inference

"""
import os
from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession
import torch
import sys
import numpy as np
sys.path.append('..')
import onnxruntime as ort
import time
from multiprocessing import Pool
import itertools


from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose


from ViT.utils.transforms import PrepareForNet
from roughSeg.utils import k_fold_split_train_val_test, getFiles


img_dir = '/data/FLARE21/training_data_192_sameKidneys/scaled_ims/'  # * path to data
model_filename = './compiled_model.quant.onnx'

test_workers = 8
batch_size=1
num_threads = 16 #* Num threads to use (MAX on PEPITA: 24*2)

#~ Dataset class
class customDataset(Dataset):
    def __init__(self, image_path, transforms, indices, apply_WL, window=[400,100], level=[50,60]):
        self.indices = indices
        self.image_path = image_path
        self.organ_to_idx = ["Background", "Liver", "Kidney", "Spleen", "Pancreas"]
        self.names = self.idx_to_names(image_path)
        self.availableImages = [sorted(getFiles(image_path))[
            ind] for ind in indices]
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
                img = np.array(arr).reshape(*img.shape, len(self.window))
            else:
                img = self.WL_norm(img, self.window, self.level)[..., np.newaxis]
        else:
            img = img[..., np.newaxis]
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

class Inference(object):
    def __init__(self, data, inf_session, workers = os.cpu_count()-1):
        print("Constructor (in pid=%d)..." % os.getpid())

        self.count = data.__len__()
        #self.sess_options = sess_options
        self.inf_session = inf_session
        #* Pooling
        pool = Pool(processes=workers)
        self.results = pool.map(self.forward, data)
        pool.close()
        pool.join()

    def __del__(self):
        self.count -= 1
        print("... Destructor (in pid=%d) count=%d" % (os.getpid(), self.count))
    
    def process_obj(self, index):
        print ("object %d" % index)
        return "results"

    def forward(self, data):
        #~ Inference
        inputs = {self.inf_session.get_inputs()[0].name: to_numpy(data['inputs'])}
        outputs = np.array(self.inf_session.run(None, inputs))
        return outputs


def main():
    test_transforms = Compose([
        PrepareForNet(), #*Channels first
    ])

    dataset_size = len(getFiles(img_dir))
    _, _, test_idx = k_fold_split_train_val_test(
        dataset_size, fold_num=1, seed=230597) #! Use test set from first fold for now
    test_dataset = customDataset(img_dir, test_transforms, indices=test_idx, apply_WL=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=False,
                             num_workers=test_workers, worker_init_fn=lambda _: np.random.seed(
                                 int(torch.initial_seed()) % (2**32-1)))
    
    print("ONNX Available providers:", ort.get_available_providers())
    print(ort.get_device())
    #* ONNX inference session
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL  # ! or ORT_SEQUENTIAL
    sess_options.optimized_model_filepath = "./optimized_model.onnx"
    sess_options.log_severity_level = 1
    sess_options.enable_profiling = False
    sess_options.inter_op_num_threads = os.cpu_count() - 1 
    sess_options.intra_op_num_threads = os.cpu_count() - 1
    ort_session = ort.InferenceSession(
        model_filename, sess_options=sess_options)
    
    t = time.time()
    output_list = []
    for data in test_loader:
        inputs = {ort_session.get_inputs()[0].name: to_numpy(data['inputs'])}
        outputs = np.array(ort_session.run(None, inputs))
        output_list.append(outputs)
    #Inference(test_loader, ort_session)
    print(len(output_list))
    out = np.array(output_list).reshape(len(test_idx), 6, 96, 192, 192)
    print(out.shape)
    print(f'Execution time: {time.time() - t} for {len(test_idx)} examples.')

if __name__ == '__main__':
    main()
