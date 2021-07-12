#!/bin/bash
CUDA_VISIBLE_DEVICES='0' python3 train.py --fold_num 1 --GPUs 1
sleep 10m
CUDA_VISIBLE_DEVICES='0' python3 train2.py --fold_num 1 --GPUs 1
#folds='2 3 4 5'
#for fold_num in $folds
#do
    #sleep 10m
    #CUDA_VISIBLE_DEVICES='0' python3 train.py --fold_num $fold_num --GPUs 1
#done
