<h1> FLARE21 </h1>

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
