#!/bin/bash
# Pipelining LR/WD in hyperparameter search
while getopts ":a:b:d:g:" opt; do
  case $opt in
    a) ar="$OPTARG"
    ;;
    b) bs="$OPTARG"
    ;;
    d) dr="$OPTARG"
    ;;
    g) gpu="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

lr=(0.0001 0.001 0.01)
wd=(0 0.00001 0.0001 0.001)

for l in "${lr[@]}"
do
  for w in "${wd[@]}"
  do
    # Variable and fixed hparams
    python3 -m src.train --use-gpus $gpu --architecture $ar --batch-size $bs \
    --dropout $dr  --initial-lr $l --weight-decay $w \
    --frozen "n" --pretrained "y" --plateau-threshold 0.0001 \
    --break-patience 5 --plateau-patience 3 --optimizer-family "AdamW" --momentum "0.9" --scheduler-family "drop" \
    --results-dir /export/gaon2/data/kvenka10/explain-mura/results/hparam_search \
    --train-file "train_100.txt" --val-file "val_100.txt" \
    --image-size "320" --use-parallel "y" --num-workers "12" --train-transform "hvflip-rotate" \
    --max-epochs "75"
  done
done