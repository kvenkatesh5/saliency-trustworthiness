#!/bin/bash
python3 -m src.heatmaps --model-name "InceptionV3_2_100" --use-gpus "3"
python3 -m src.heatmaps --model-name "InceptionV3_2_50" --use-gpus "3"
python3 -m src.heatmaps --model-name "InceptionV3_2_10" --use-gpus "3"
python3 -m src.heatmaps --model-name "InceptionV3_2_1" --use-gpus "3"