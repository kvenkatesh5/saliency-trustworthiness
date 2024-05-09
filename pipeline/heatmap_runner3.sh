#!/bin/bash
python3 -m src.heatmaps --model-name InceptionV3_3_100 --use-gpus 7
python3 -m src.heatmaps --model-name InceptionV3_3_50 --use-gpus 7
python3 -m src.heatmaps --model-name InceptionV3_3_10 --use-gpus 7
python3 -m src.heatmaps --model-name InceptionV3_3_1 --use-gpus 7