#!/bin/bash
folds='1 2 3 4 5'
for fold_num in $folds
do
    CUDA_VISIBLE_DEVICES='0' python3 train_headHunter.py --fold_num $fold_num --GPUs 1
done