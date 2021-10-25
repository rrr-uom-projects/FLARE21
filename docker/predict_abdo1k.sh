#! /bin/sh

echo "Running MCR-RRR segmentation model (Abdo1k edition)"



python3 /rrr-models/inference.py /workspace/inputs /workspace/outputs --model_path=/rrr-models/compiled_model_AbdomenCT1K.onnx