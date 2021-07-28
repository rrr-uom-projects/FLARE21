"""
Frankenscript to input & output nii.gz images
"""

import SimpleITK as sitk

import onnxruntime as ort
import multiprocessing as mp
import argparse as ap
import os
import pathlib
import numpy as np
import numba
import time

## Global configuration variables

out_resolution = (96,192,192)
level_0 = 50
window_0 = 400
minval_0 = level_0 - window_0//2
maxval_0 = level_0 + window_0//2

level_1 = 60
window_1 = 100
minval_1 = level_1 - window_1//2
maxval_1 = level_1 + window_1//2

model_path = "./rrr-models/compiled_model_nano.onnx"

class InferenceRecord:
    """
    Class to store a record of something that is being inferenced on

    Data only class, just makes carting it around a bit easier
    """
    def __init__(self, image, flipped, origin, direction, original_size, original_spacing, filename):
        self.npy_image = image
        self.flipped = flipped
        self.origin = origin
        self.direction = direction
        self.original_size = original_size
        self.original_spacing = original_spacing
        self.filename = filename

        self.spacing = [sz*spc/nsz for nsz,sz,spc in zip(out_resolution[::-1], self.original_size, self.original_spacing)]




@numba.jit(parallel=True, cache=True)
def WL_norm(img):
    """
    Apply a window and level transformation to the image data, also expands up to 2 channels 

    Notes:
        - Uses global variables to reduce computation in the function
        - Profiling suggests this is about as fast as it can go
        - Fastest when image is kept as int16
    """
    wld = np.zeros((2,*img.shape), dtype=np.float32)
    wld[0,...] = np.minimum(maxval_0, np.maximum(img, minval_0)).astype(np.float32) #np.clip(img, minval, maxval)
    wld[0,...] -= minval_0
    wld[0,...] /= window_0

    wld[1,...] = np.minimum(maxval_1, np.maximum(img, minval_1)).astype(np.float32) #np.clip(img, minval, maxval)
    wld[1,...] -= minval_1
    wld[1,...] /= window_1

    return wld

def load_nifty(path):
    """
    Load a nifty file and apply all preprocessing steps, ready for inference.

    NB: The shape needs to be reversed to work in SITK - i.e. slices last

    Steps are:
        - Downsample to network input resolution
        - Convert to numpy array
        - Flip CC & LR if required
        - Window/level normalise & expand channels
    """
    filename = pathlib.PurePath(path).name

    read_start = time.time()
    sitk_im = sitk.ReadImage(path)
    read_end = time.time()

    ## immediately downsample to expected resolution
    resamp_start = time.time()
    reference_image = sitk.Image(out_resolution[::-1], sitk_im.GetPixelIDValue())
    reference_image.SetOrigin(sitk_im.GetOrigin())
    reference_image.SetDirection(sitk_im.GetDirection())
    reference_image.SetSpacing([sz*spc/nsz for nsz,sz,spc in zip(out_resolution[::-1], sitk_im.GetSize(), sitk_im.GetSpacing())])
    
    ## calculate smoothing sigma to match scikit-image:
    ## Standard deviation for Gaussian filtering to avoid aliasing artifacts. 
    ## By default, this value is chosen as (s - 1) / 2 where s is the down-scaling factor, where s > 1. 
    ## For the up-size case, s < 1, no anti-aliasing is performed prior to rescaling.

    factors = [old/new for old, new in zip(sitk_im.GetSize(), out_resolution[::-1] )]
    
    smoothing_sigma = np.clip([(f-1)/2 for f in factors], 0.00001, 100) ## assume f > 1 !
    
    sitk_im_resamp = sitk.Resample(sitk.SmoothingRecursiveGaussian(sitk_im, smoothing_sigma), reference_image, interpolator=sitk.sitkBSpline )
    resamp_end = time.time()

    
    im_o = sitk.GetArrayFromImage(sitk_im_resamp).astype(np.int16) ## cast back to short - some appear to be stored as float32?
    # check if flip required
    flipped = False
    if sitk_im.GetDirection()[-1] == -1:
        # print("Image upside down, CC flip required!")
        flip_start = time.time()
        im_o = np.flip(im_o, axis=0)        # flip CC
        im_o = np.flip(im_o, axis=2)        # flip LR --> this works, should be made more robust though (with sitk cosine matrix)
        flip_end = time.time()
        flipped = True
        
    ## perform normalisation
    wld_start = time.time()
    im_o_n = WL_norm(im_o) ## this actually takes the longest time!
    wld_end = time.time()

    ## Uncomment this to get some timing data
    # if flipped:
    #     print(f"reading: {read_end-read_start:.4f}\t\tresampling: {resamp_end - resamp_start:.4f}\t\tflipping{flip_end - flip_start:.4f}\t\twindowing:{wld_end-wld_start:.4f}")
    return InferenceRecord(im_o_n.astype(np.float32), 
                            flipped,
                            sitk_im.GetOrigin(),
                            sitk_im.GetDirection(),
                            sitk_im.GetSize(),
                            sitk_im.GetSpacing(),
                            filename)


def get_onnx_session():
    """
    Load the ONNX model and return an onnxruntime with the correct
    session options set

    Directly taken from Donal's onnx inference.py script

    1 change - I made ONNX shut up
    """
    print("ONNX Available providers:", ort.get_available_providers())
    print(ort.get_device())
    #* ONNX inference session
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL # ! or ORT_SEQUENTIAL
    sess_options.optimized_model_filepath = "./optimized_model.onnx"
    sess_options.log_severity_level = -1 ## make onnx shut up
    sess_options.enable_profiling = False
    sess_options.inter_op_num_threads = os.cpu_count() - 1 
    sess_options.intra_op_num_threads = os.cpu_count() - 1
    
    ort_session = ort.InferenceSession(model_path, sess_options=sess_options)
    return ort_session

def inference_one(session, image):
    """
    Use an ONNX session to run inference on a single image.

    The image comes in as a record, containing the image and all header data.

    We return a new record, where the 'image' is the segmentation and the header data 
    is copied; this allows easier writing of the nifty.
    """
    image_input = np.broadcast_to(image.npy_image, (1,*image.npy_image.shape))
    ort_inputs = {"img":image_input}
    outputs = np.array(session.run(None, ort_inputs)).squeeze()
    preds = np.argmax(outputs, axis=0).astype(np.int8)
    return InferenceRecord(preds, 
                    image.flipped, 
                    image.origin, 
                    image.direction, 
                    image.original_size, 
                    image.original_spacing, 
                    image.filename)

@numba.jit(parallel=True, cache=True)
def clip_body(seg):
    """
    Remove body segmentation from the final segmentation object

    This should be accelerated by numba and end up faster than np.clip (I think)
    """
    return np.minimum(4, np.maximum(seg-1, 0))


def mp_write_wrapper(args):
    """
    Wrapper function for multiprocessing writer.

    mp.Pool.map can't handle functions with multiple arguments, this is one way around that
    while not compromising the readability of the underlying function.
    """
    record, out_dir = args
    write_one(record, out_dir)

def write_one(record, out_dir):
    """
    Resample, unflip and write to output directory
    """
    output_image = clip_body(record.npy_image)
    ## first unflip if needed
    if record.flipped:
        output_image = np.flip(output_image, axis=0)        # flip CC
        output_image = np.flip(output_image, axis=2)        # flip LR --> this works, should be made more robust though (with sitk cosine matrix)
    
    ## create output image
    sitk_im = sitk.GetImageFromArray(output_image.astype(np.int8))
    sitk_im.SetOrigin(record.origin)
    sitk_im.SetDirection(record.direction)
    sitk_im.SetSpacing(record.spacing)

    ## Now resample, make sure to use nearest neighbour
    resamp_start = time.time()
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputOrigin(record.origin)
    resampler.SetOutputDirection(record.direction)
    resampler.SetOutputSpacing(record.original_spacing)
    resampler.SetSize(record.original_size)

    
    ## Should we do some smoothing here? 
    sitk_im_resamp = resampler.Execute(sitk_im)
    resamp_end = time.time()

    write_start = time.time()
    ## Now we can write
    sitk.WriteImage(sitk_im_resamp, os.path.join(out_dir, record.filename))
    write_end = time.time()

    

def main(args):
    """
    Run the full segmentation workflow:
        - List images in the input directory
        - Load those images into memory (in parallel)
            - Resample, flip and normalise the images as they are loaded
        - Run one-by-one inference (sequential, but onnx uses all cores)
        - Write segmentation masks to the output directory (in parallel)
            - unflip, resample on the way
    """
    start_all = time.time()
    images_2_segment = [os.path.join(args.input_dir, im) for im in os.listdir(args.input_dir)]
    print(f"Detected {len(images_2_segment)} images to segment...")

    # Load and transform images
    worker_pool = mp.Pool()
    start_read = time.time()
    targets  = worker_pool.map(load_nifty, images_2_segment)
    worker_pool.close()
    worker_pool.join()
    end_read = time.time()

    print(f"Total image loading time: {end_read-start_read} = {(end_read-start_read)/len(targets)} s/image")

    ## Load model

    inference_session = get_onnx_session()

    ## run inference
    start_inf = time.time()
    results = []
    for tgt in targets:
        results.append((inference_one(inference_session, tgt), args.output_dir))
    end_inf = time.time()
    print(f"Inference time: {end_inf - start_inf} or {(end_inf - start_inf)/len(targets)} s/image")

    ## Now figure out how to write all that...

    start_write = time.time()
    writer_pool = mp.Pool()
    writer_pool.map(mp_write_wrapper, results)
    writer_pool.close()
    writer_pool.join()

    end_all = time.time()
    print(f"Writing time: {end_all - start_write} = {(end_all-start_write)/len(targets)} s/image")

    print(f"Total time: {end_all - start_all} = {(end_all-start_all)/len(targets)} s/image")

    exit()








if __name__ == "__main__":
    """
    Handle argparse here
    """
    parser = ap.ArgumentParser()
    parser.add_argument("input_dir", help="Directory containing images to segment")
    parser.add_argument("output_dir", help="Directory in which to put the output")

    args = parser.parse_args()
    main(args)
