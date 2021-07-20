"""
Compile with ONNX and run on GPU
https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
"""

import os
import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np
from argparse import ArgumentParser
import sys
sys.path.append('..')


from models import yolo_transpose_plusplus, tiny_segmenter, tiny_inference_segmenter


def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

parser = ArgumentParser(prog="Compile trained model")
parser.add_argument("weights", help="Path to weights", type=str)
parser.add_argument("output", help="Output filename (+.onnx)", type=str)
parser.add_argument("--check_model", default=True, help="Check compiled model", type=bool)
parser.add_argument("--check_output", default=True, help="Check onnx model outputs against pytorch model's", type=bool)
parser.add_argument("--quantize", default=True, help="Quantize model to decrease size (def. int8)", type=bool)
args = parser.parse_args()


def sigmoid(x):
    return 1/(1+np.exp(-x))

def to_numpy(x):
    return x.detach().cpu().numpy() if x.requires_grad else x.cpu().numpy()

def export():
    model  = tiny_segmenter(n_classes=6, in_channels=2, p_drop=0.25)

    #* Load trained model
    model.load_best(args.weights) #* Will default to GPU if detected + defaults to best_checkpoint.pytorch in weights directory
    model.eval()

    #* Random input w. correct shape
    out_size = (96, 512, 512)
    x = torch.randn(1, 2, 96, 192, 192, requires_grad=True, dtype=torch.float32) #! B x C x H x W x D
    #output = model(x, *out_size)
    output = model(x)
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
    
    if args.check_output:
        #* Runs on CPU
        ort_session = ort.InferenceSession(args.output + '.onnx')
        #inputs = [x, *(torch.tensor(x) for x in out_size)]
        inputs = [x]
        print([x.shape for x in inputs])
        ort_inputs = {key.name: to_numpy(x) for key, x in zip(ort_session.get_inputs(), inputs)}
        ort_outputs = ort_session.run(None, ort_inputs)
        np.testing.assert_allclose(sigmoid(to_numpy(output)), 
            sigmoid(ort_outputs[0]), rtol=1e-03, atol=1e-05) #*rtol = relative tolerance; atol=absolute tolerance
        print('Outputs look good!')

    if args.quantize:
        quantize_dynamic(args.output+'.onnx', args.output + ".quant.onnx", weight_type=QuantType.QUInt8)

if __name__ =='__main__':
    export()
