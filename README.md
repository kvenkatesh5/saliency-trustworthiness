# ExplainMURA: Trustworthiness Experiments

This document outlines how to use/adapt our code for preparing and running the trustworthiness experiments presented in our manuscript. If you use this repository, please cite our publication:

**Venkatesh, K., Mutasa, S., Moore, F. et al. Gradient-Based Saliency Maps Are Not Trustworthy Visual Explanations of Automated AI Musculoskeletal Diagnoses. J Digit Imaging. Inform. med. (2024). https://doi.org/10.1007/s10278-024-01136-4**

Please feel free to contact Kesavan for questions about this repository. Email is kvenka10 at alumni dot jh dot edu. Questions about overall study should be directed to ***REMOVED*** author Paul Yi (pyi at som dot umaryland dot edu). _This repository is being actively uploaded to GitHub as of May 6, 2024. Complete code will be available soon._

## Requirements
- Python 3.6.9
```
python3 -m venv tenv
source tenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Model Training
- Hyperparameter search can be done by pipelining the ```train.py``` script. See ```pipeline/hparam_lrwd.sh``` and its runners ```pipeline/hparam_runner[X]```.
- After hyperparameter search and selection, use the ```train.py``` script for training final models (i.e., models trained on the training and validation splits). See ```pipeline/train_densenet.sh``` and ```pipeline/train_inception.sh``` and their associated runners (```pipeline/densenet_runner.sh``` and ```pipeline/inception_runner[X]```) for sample runs.

## Model Evaluation
- Generate model predictions using the ```test.py``` script. See ```pipeline/test_final_models.sh``` for sample runs.
- Use ```eval.py``` and ```eval_stats_rev1.py``` and ```eval_plot.py``` for statistically evaluating these testing set predictions. An AUROC plot will be made by ```eval_plot.py``` and saved locally. A numpy zipfile with AUC and Kappa scores will be saved to the temp directory for further statistical evaluation by ```eval_stats_rev1.py```.

## Credit
We would like to specifically thank Zach Murphy (a former colleague and collaborator of Kesavan's, see https://doi.org/10.1148/ryai.220012) and Nisanth Arun (co-first author of https://doi.org/10.1148/ryai.2021200267) for their gracious help.
