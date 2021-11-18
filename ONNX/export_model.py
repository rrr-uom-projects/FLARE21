"""
Compile with ONNX and run on GPU
https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
"""

import os
from onnxruntime.quantization.quant_utils import quantize_data
from onnxruntime.quantization.quantize import quantize
import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static, CalibrationDataReader, QuantFormat, QuantType
import numpy as np
from argparse import ArgumentParser
import sys
import time
sys.path.append('..')

from roughSeg.utils import k_fold_split_train_val_test, getFiles
from models import yolo_transpose_plusplus, tiny_segmenter, tiny_inference_segmenter, nano_segmenter
from einops import rearrange

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

parser = ArgumentParser(prog="Compile trained model")
parser.add_argument("weights", help="Path to weights", type=str)
parser.add_argument("output", help="Output filename (+.onnx)", type=str)
parser.add_argument("--check_model", default=True, help="Check compiled model", type=str2bool)
parser.add_argument("--check_output", default=True, help="Check onnx model outputs against pytorch model's", type=str2bool)
# parser.add_argument("--quantize", default=True, help="Quantize model to decrease size (def. int8)", type=str2bool)
# parser.add_argument("--calibration_data", default="/ data/FLARE21/training_data_192_sameKidneys/scaled_ims/",
#                      help="Calibration dataset for static quantization", type=str)
args = parser.parse_args()


def sigmoid(x):
    return 1/(1+np.exp(-x))

def to_numpy(x):
    return x.detach().cpu().numpy() if x.requires_grad else x.cpu().numpy()


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def export():
    model  = nano_segmenter(n_classes=6, in_channels=2, p_drop=0)

    #* Load trained model
    model.load_best(args.weights) #* Will default to GPU if detected + defaults to best_checkpoint.pytorch in weights directory
    model.eval()

    #* Random input w. correct shape
    out_size = (96, 512, 512)
    x = torch.randn(1, 2, 96, 192, 192, requires_grad=True, dtype=torch.float32) #! B x C x H x W x D
    #output = model(x, *out_size)
    output = model(x)
    benchmark_pytorch(args.weights)
    print(output.shape)
    #TODO Optimise model https://www.onnxruntime.ai/docs/resources/graph-optimizations.html#python-api-example
    torch.onnx.export(model, 
                        args = x,
                        #args=(x, *out_size), 
                        f=args.output + '.onnx', 
                        export_params=True, opset_version=13,
                        do_constant_folding=True,
                        input_names = ['img', 'depth', 'height', 'width'],
                        output_names = ['output'],
                        dynamic_axes = {'img':{0: 'batch_size'}, #* Export w. variable batch axis
                        # 'depth': {0:'batch_size'},
                        # 'height': {0: 'batch_size'},
                        # 'width': {0: 'batch_size'},
                        'output': {0: 'batch_size', 1: 'depth'}}) #* Variable depth to account for resampling

    if args.check_model:
        onnx_model = onnx.load(args.output + '.onnx')
        onnx.checker.check_model(onnx_model)
    
    # if args.quantize:
    #     input_model = args.output + '.onnx'
    #     output_model = args.output + '.quant.onnx'
    #     calibration_data = args.calibration_data
    #     #* Get train image idx
    #     dataset_size = len(getFiles(calibration_data))
    #     print('Dataset size', dataset_size)
    #     train_idx, _, _ = k_fold_split_train_val_test(
    #         dataset_size, fold_num=1, seed=230597)  # ! Use test set from first fold for now
    #     print('# Calibration examples', len(train_idx))
    #     dataset = ONNXReader(calibration_data, train_idx)
    #     quantize_static(input_model, 
    #                     output_model, 
    #                     dataset, 
    #                     quant_format=QuantFormat.QOperator,
    #                     per_channel=False, 
    #                     weight_type=QuantType.QInt8)
    #     print("Calibrated and quantized model saved.")

        #Dynamic quantization
        #quantize_dynamic(args.output+'.onnx', args.output + ".quant.onnx", weight_type=QuantType.QUInt8)

    print('benchmarking fp32 model...')
    benchmark(args.output + '.onnx')

    
    # print('benchmarking int8 model...')
    # benchmark(args.output + '.quant.onnx')

    if args.check_output:
        #* Runs on CPU
        ort_session = ort.InferenceSession(args.output + '.onnx')
        #inputs = [x, *(torch.tensor(x) for x in out_size)]
        inputs = [x]
        print([x.shape for x in inputs])
        ort_inputs = {key.name: to_numpy(x) for key, x in zip(
            ort_session.get_inputs(), inputs)}
        ort_outputs = ort_session.run(None, ort_inputs)
        out = np.argmax(ort_outputs[0], axis=1)
        pytorch = np.argmax(to_numpy(output), axis=1)
        print(out.shape, pytorch.shape)
        # *rtol = relative tolerance; atol=absolute tolerance
        np.testing.assert_allclose(pytorch, out, rtol=1e-03, atol=1e-05)
        print('Outputs look good!')


class ONNXReader(CalibrationDataReader):
    def __init__(self, calibration_data, indices, augmented_model_path='augmented_model.onnx', window=[400, 100], level=[50,60]):
        self.image_folder = calibration_data
        self.indices = indices
        self.avail_images = [sorted(getFiles(self.image_folder))[
            idx] for idx in self.indices] 
        self.names = self.idx_to_names(calibration_data)
        self.window = window
        self.level = level
        self.augmented_model_path = augmented_model_path
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0
        self.apply_WL = True

    def get_next(self):
        #* Create batch and preprocess
        if self.preprocess_flag:
            self.preprocess_flag = False
            session = ort.InferenceSession(self.augmented_model_path, None)
            data_stack = self.prep_images(size_limit=0) #* 0 = all images
            input_name = session.get_inputs()[0].name
            self.datasize = data_stack.shape[0]
            self.enum_data_dicts = iter([{input_name: data_stack[i][np.newaxis]} for i in range(self.datasize)])
        return next(self.enum_data_dicts, None)

    def prep_images(self, size_limit=0):
        #* Preprocessing
        if size_limit > 0 and len(self.names) >= size_limit:
            batch_filenames = [self.names[i] for i in range(size_limit)]
        else:
            batch_filenames = self.names
        batch_data = []
        for image_name in batch_filenames:
            img = np.load(os.path.join(self.image_folder, image_name))
            if self.apply_WL:
                if type(self.window) is list and type(self.level) is list:
                    arr = []
                    for i in range(len(self.window)):
                        arr.append(self.WL_norm(
                            img, self.window[i], self.level[i]))
                    img = np.stack(arr, axis=-1)
                else:
                    img = self.WL_norm(
                        img, self.window, self.level)[..., np.newaxis]
            else:
                img = img[..., np.newaxis]

            img = rearrange(img, " h w d c -> c h w d")
            batch_data.append(img)
        return np.stack(batch_data, axis=0)

    def idx_to_names(self, image_dir):
        return [sorted(getFiles(image_dir))[idx] for idx in self.indices]        

    @staticmethod
    def WL_norm(img, window, level):
        minval = level - window/2
        maxval = level + window/2
        wld = np.clip(img, minval, maxval)
        wld -= minval
        wld /= window
        return wld


def benchmark(model_path):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    total = 0.0
    runs = 10
    input_data = np.zeros((1, 2, 96, 192, 192), np.float32)
    # Warming up
    _ = session.run([], {input_name: input_data})
    for i in range(runs):
        start = time.perf_counter()
        _ = session.run([], {input_name: input_data})
        end = (time.perf_counter() - start) * 1000
        total += end
        print(f"{end:.2f}ms")
    total /= runs
    print(f"Avg: {total:.2f}ms")
#
def benchmark_pytorch(model_path):
    model = nano_segmenter(n_classes=6, in_channels=2, p_drop=0)

    #* Load trained model
    # * Will default to GPU if detected + defaults to best_checkpoint.pytorch in weights directory
    model.load_best(args.weights)
    model.eval()
    x = torch.randn(1, 2, 96, 192, 192, requires_grad=True,
                    dtype=torch.float32)
    runs=10
    total = 0.0
    print('Pytorch benchmark')
    for i in range(runs):
        start = time.perf_counter()
        output = model(x)
        end = (time.perf_counter() - start) * 1000
        total += end
        print(f"{end:.2f}ms")
    total /= runs
    print(f"Avg: {total:.2f}ms")

if __name__ =='__main__':
    export()
