#/bin/bash
python3 -m src.test --model-name "InceptionV3_1_100" --use-gpus "4"
python3 -m src.test --model-name "InceptionV3_2_100" --use-gpus "4"
python3 -m src.test --model-name "InceptionV3_3_100" --use-gpus "4"
python3 -m src.test --model-name "DenseNet169_1_100" --use-gpus "4"

python3 -m src.test --model-name "InceptionV3_1_50" --use-gpus "4"
python3 -m src.test --model-name "InceptionV3_2_50" --use-gpus "4"
python3 -m src.test --model-name "InceptionV3_3_50" --use-gpus "4"
python3 -m src.test --model-name "DenseNet169_1_50" --use-gpus "4"

python3 -m src.test --model-name "InceptionV3_1_10" --use-gpus "4"
python3 -m src.test --model-name "InceptionV3_2_10" --use-gpus "4"
python3 -m src.test --model-name "InceptionV3_3_10" --use-gpus "4"
python3 -m src.test --model-name "DenseNet169_1_10" --use-gpus "4"

python3 -m src.test --model-name "InceptionV3_1_1" --use-gpus "4"
python3 -m src.test --model-name "InceptionV3_2_1" --use-gpus "4"
python3 -m src.test --model-name "InceptionV3_3_1" --use-gpus "4"
python3 -m src.test --model-name "DenseNet169_1_1" --use-gpus "4"
