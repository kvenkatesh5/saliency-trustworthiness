#!/bin/bash
# use 1 GPU due to parallel split of batches (using batch size 32, there will be a final batch
# that contains 3 images. so you cannot split this onto 2 GPUs, otherwise one copy will get just 1 image
# (not allowed in training mode!).
bash ./pipeline/train_inception.sh -t "train_all_1.txt" -g "4"
bash ./pipeline/train_inception.sh -t "train_all_1.txt" -g "4"
bash ./pipeline/train_inception.sh -t "train_all_1.txt" -g "4"
