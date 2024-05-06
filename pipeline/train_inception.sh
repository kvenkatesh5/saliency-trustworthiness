#!/bin/bash
# Pipelining final model training using opt hparams
while getopts ":t:g:" opt; do
  case $opt in
    t) tf="$OPTARG"
    ;;
    g) gpu="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

# Uses optimal hparams for InceptionV3
python3 -m src.train --use-gpus $gpu --architecture "InceptionV3" --batch-size "32" \
--dropout "0.5"  --initial-lr "0.0001" --weight-decay "0" \
--frozen "n" --pretrained "y" --plateau-threshold 0.0001 \
--break-patience 5 --plateau-patience 3 --optimizer-family "AdamW" --momentum "0.9" --scheduler-family "drop" \
--results-dir /export/gaon2/data/kvenka10/explain-mura/results/model_training \
--train-file $tf --val-file "test_100.txt" \
--image-size "320" --use-parallel "y" --num-workers "12" --train-transform "hvflip-rotate" \
--max-epochs "75"
