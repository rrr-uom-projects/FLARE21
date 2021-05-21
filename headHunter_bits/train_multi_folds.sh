#!/bin/bash
folds='2'
for fold_num in $folds
do
    CUDA_VISIBLE_DEVICES='1' python3 train_headHunter.py --fold_num $fold_num --GPUs 1 
done