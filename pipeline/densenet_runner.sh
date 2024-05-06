#!/bin/bash
# DenseNet169 models
#bash ./pipeline/train_densenet.sh -t "train_all.txt" -g "4,5"
bash ./pipeline/train_densenet.sh -t "train_all_50.txt" -g "4,5"
bash ./pipeline/train_densenet.sh -t "train_all_10.txt" -g "4,5"
bash ./pipeline/train_densenet.sh -t "train_all_1.txt" -g "4,5"
