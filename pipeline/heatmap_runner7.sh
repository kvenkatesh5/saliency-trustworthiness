#!/bin/bash
python3 -m src.heatmaps --model-name InceptionV3_1_1 --cascade-layer fc --use-gpus 4
python3 -m src.heatmaps --model-name InceptionV3_1_1 --cascade-layer Mixed_7c --use-gpus 4
python3 -m src.heatmaps --model-name InceptionV3_1_1 --cascade-layer Mixed_7b --use-gpus 4
python3 -m src.heatmaps --model-name InceptionV3_1_1 --cascade-layer Mixed_7a --use-gpus 4
python3 -m src.heatmaps --model-name InceptionV3_1_1 --cascade-layer Mixed_6e --use-gpus 4
python3 -m src.heatmaps --model-name InceptionV3_1_1 --cascade-layer Mixed_6d --use-gpus 4
python3 -m src.heatmaps --model-name InceptionV3_1_1 --cascade-layer Mixed_6c --use-gpus 4
python3 -m src.heatmaps --model-name InceptionV3_1_1 --cascade-layer Mixed_6b --use-gpus 4
python3 -m src.heatmaps --model-name InceptionV3_1_1 --cascade-layer Mixed_6a --use-gpus 4
python3 -m src.heatmaps --model-name InceptionV3_1_1 --cascade-layer Mixed_5d --use-gpus 4
python3 -m src.heatmaps --model-name InceptionV3_1_1 --cascade-layer Mixed_5c --use-gpus 4
python3 -m src.heatmaps --model-name InceptionV3_1_1 --cascade-layer Mixed_5b --use-gpus 4
python3 -m src.heatmaps --model-name InceptionV3_1_1 --cascade-layer Conv2d_4a_3x3 --use-gpus 4
python3 -m src.heatmaps --model-name InceptionV3_1_1 --cascade-layer Conv2d_3b_1x1 --use-gpus 4
python3 -m src.heatmaps --model-name InceptionV3_1_1 --cascade-layer Conv2d_2b_3x3 --use-gpus 4
python3 -m src.heatmaps --model-name InceptionV3_1_1 --cascade-layer Conv2d_2a_3x3 --use-gpus 4
python3 -m src.heatmaps --model-name InceptionV3_1_1 --cascade-layer Conv2d_1a_3x3 --use-gpus 4
