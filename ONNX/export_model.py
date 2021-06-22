"""
Compile with ONNX and run on GPU
https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
"""

import os
import torch
import onnx
import onnxruntime as ort
from onnxruntime import quantize_dynamic, QuantType
import numpy as np
from argparse import ArgumentParser
import sys
sys.path.append('..')


from models import bottleneck_yolo_segmenter



parser = ArgumentParser(prog="Compile trained model")
parser.add_argument("weights", help="Path to weights", type=str)
parser.add_argument("output", help="Output filename (+.onnx)", type=str)
parser.add_argument("--check_model", default=True, help="Check compiled model", type=bool)
parser.add_argument("--check_output", default=True, help="Check onnx model outputs against pytorch model's", type=bool)
parser.add_argument("--quantize", default=True, help="Quantize model to decrease size (def. int8)", type=bool)
args = parser.parse_args()


def to_numpy(x):
    return x.detach().cpu().numpy() if x.requires_grad else x.cpu().numpy()

def export():
    model  = bottleneck_yolo_segmenter(n_classes=7, in_channels=1, p_drop=0.25)

    #* Load trained model
    model.load_best(args.weights) #* Will default to GPU if detected
    model.eval()

    #* Random input w. correct shape
    x = torch.randn(1, 1, 96, 256, 256, requires_grad=True, dtype=torch.float32) #! B x C x H x W x D
    output = model(x)
    print(output.shape)
    torch.onnx.export(model, x, args.output + '.onnx', 
                        export_params=True, opset_version=12,
                        do_constant_folding=True,
                        input_names = ['input'],
                        output_names = ['output'],
                        dynamic_axes = {'input':{0: 'batch_size'}, #* Export w. variable batch axis
                        'output': {0: 'batch_size'}})

    if args.check_model:
        onnx_model = onnx.load(args.output + '.onnx')
        onnx.checker.check_model(onnx_model)
    
    if args.check_output:
        #* Runs on CPU
        ort_session = ort.InferenceSession(args.output + '.onnx')
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
        ort_outputs = ort_session.run(None, ort_inputs)
        np.testing.assert_allclose(to_numpy(output), 
            ort_outputs[0], rtol=1e-03, atol=1e-05) #*rtol = relative tolerance; atol=absolute tolerance
        print('Outputs look good!')

    if args.quantize:
        quantize_dynamic(args.output+'.onnx', args.output + ".quant.onnx", weight_type=QuantType.QUInt8)

if __name__ =='__main__':
    export()
