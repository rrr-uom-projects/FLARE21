<h1> MCR_RRR FLARE21 submission: Cpu-Only aBdominal oRgan segmentAtion (COBRA)</h1>
<h2> Training data preprocessing </h2>
Before training our model, the training data needs to be pre-processing. For this use <strong>train_preprocessing.py</strong> in the parent directory.<br>
This script will:<br>
<li> convert images and gold standard segmentations from .nii.gz to .npy format <br>
<li> correct orientation errors <br>
<li> segment the "body" using thresholding <br>
<li> resample all images to the same dimensions (96 x 192 x 192 voxels) <br>
<li> save the resulting data to a directory ready for training <br>

<strong>NOTE</strong> - Three directory paths need to be changed here:<br>
<li> imdir: directory containing the raw .nii.gz format CT images<br>
<li> maskdir: directory containing the raw .nii.gz format gold standard segmentations<br>
<li> out_dir: where the preprocessed images and masks will be saved<br>

<h2> Model training </h2>
Once pre-processed, the data is ready for model training!<br>
To train a model, use <code>python train.py --fold_num i</code> where <code>i</code> is an integer in [1,2,3,4,5]<br>
or <code>sh train_multi_folds.sh</code><br>
  
<strong>NOTE</strong> - Two directory paths need to be changed here:<br>
<li> source_dir: where the preprocessed images and masks are saved by the preprocessing script<br>
<li> checkpoint_dir: where model weights and training statitics will be saved<br>
  
Current hyper-parameter settings in <strong>train.py</strong> will reproduce our submitted model.<br>

<h2> ONNX compilation + Inference </h2>
<h3> Compile </h3>
Once the model has been trained, it can be compiled by running <code>cd ONNX && bash ./export.sh</code>. <br>
ARGS:<br>
<li> MODEL_PATH: path to trained model.<br>
<li> OUTPUT_NAME: compiled model will be named: <code>OUTPUT_NAME.onnx</code>.<br>
<li> check_model: checks model was compiled correctly. <br>
<li> check_output: validates compiled model outputs against original pytorch model. <br>
<strong>NOTE</strong>: If you get the following error <code>AssertionError: Not equal to tolerance rtol=0.001, atol=1e-05</code>. Re-running the command should fix the issue.
<br>
<h3> Inference </h3>  
Inference is performed by the <strong>inference.py</strong> script in the docker directory.<br>
This script operates end-to-end, reading .nii.gz format CTs and writing .nii.gz format segmentations (no preprocessing necessary).<br>
