from ubuntu:20.04

run apt update && apt upgrade -y
run apt install -y python3 python3-pip

run pip install onnx onnxruntime nvidia-ml-py3
run pip install numba  SimpleITK

run mkdir /rrr-models

copy ./docker/inference.py /rrr-models
copy ./docker/compiled_model_nano.onnx /rrr-models

copy ./docker/predict.sh .

