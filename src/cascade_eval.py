"""
Evaluate the cascading randomized CNNs used in study.
"""

import os, sys, shutil, json
import pandas as pd
import numpy as np
import time
import argparse
import warnings

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, cohen_kappa_score
import matplotlib.pyplot as plt
import random
import math
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from .image_utils import bootstrap

warnings.filterwarnings("ignore", category=UserWarning)

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--cfg-dir", default='cfg.json', type=str, help='')
parser.add_argument("--use-gpus", default='all', type=str, help='')
args = parser.parse_args()

# Set GPU vis
if args.use_gpus != 'all':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.use_gpus

# Get cfg
with open(args.cfg_dir.replace('~', os.path.expanduser('~')), 'r') as f:
    cfg = json.load(f)
cfg['data_dir'] = cfg['data_dir'].replace('~', os.path.expanduser('~'))
image_dir = os.path.join(cfg['data_dir'], "mura", "images")

# Parse args
labels = cfg['labels_mura'][1:]
regions = ['shoulder', 'humerus', 'elbow', 'forearm', 'wrist', 'hand', 'finger']

# Layer randomization order names
layer_randomization_order_names = [
    'fc',
    'Mixed_7c', 'Mixed_7b', 'Mixed_7a',
    'Mixed_6e', 'Mixed_6d', 'Mixed_6c', 'Mixed_6b', 'Mixed_6a',
    'Mixed_5d', 'Mixed_5c', 'Mixed_5b',
    'Conv2d_4a_3x3', 'Conv2d_3b_1x1', 'Conv2d_2b_3x3', 'Conv2d_2a_3x3', 'Conv2d_1a_3x3'
]

# Models to evaluate
to_analyze = []
base_model_names = ["InceptionV3_1_100", "InceptionV3_2_100", "InceptionV3_3_100"]
base_models = [cfg["final_models"][x] for x in base_model_names]
for ii in range(len(base_model_names)):
    base_model_name = base_model_names[ii]
    base_model = base_models[ii]
    for i, layer in enumerate(layer_randomization_order_names):
        casc_model = os.path.join(
            cfg["cascade_model_dir"],
            base_model.replace("model.pt", layer_randomization_order_names[i] + ".pt")
        )
        to_analyze.append({
            "file": casc_model,
            "name": base_model_name + "_" + layer
        })

# Set seeds
random.seed(5)
np.random.seed(5)

# Get bootstrapped metrics
results = {}
for t in to_analyze:
    path = t["file"].replace(cfg["cascade_model_dir"], cfg["cascade_model_evaluation_dir"])\
        .replace(".pt", ".pkl")
    with open(path, "rb") as f:
        m = pickle.load(f)

    # x[1] ~ Positive/Abnormality probability
    dat = pd.DataFrame().from_dict({'y': [int(x[1]) for x in m['mura_test_100']['y']],
                                    'yhat': [x[1] for x in m['mura_test_100']['yhat']],
                                    'region': m['mura_test_100']['region'],
                                    'study': m['mura_test_100']['study']})
    assert len(dat) == 1311
    print(t['name'])
    results[t['name']] = bootstrap(dat, n=1000)

# Get results table
results_table = pd.DataFrame()
for model_name, model_results in results.items():
    for metric in ['auc_weighted',
                   'micro_precision',
                   'micro_recall',
                   'micro_f1_score',
                   'micro_accuracy',
                   'macro_precision',
                   'macro_recall',
                   'macro_f1_score',
                   'macro_accuracy',
                   'kappa_score']:
        row = pd.DataFrame.from_dict({
            'Model': [model_name],
            'Data set': 'mura_test_100',
            'Finding': ['Overall'],
            'Metric': [metric],
            'Mean': model_results[metric]['mean'][0],
            'LCI': model_results[metric]['lci'][0],
            'UCI': model_results[metric]['uci'][0]})
        results_table = pd.DataFrame.append(results_table, row, ignore_index=True)

    for metric in ['auc', 'precision', 'recall', 'f1_score', 'accuracy', 'threshold']:
        row = pd.DataFrame.from_dict({
            'Model': [model_name] * len(regions),
            'Data set': ['mura_test_100'] * len(regions),
            'Finding': regions,
            'Metric': [metric] * len(regions),
            'Mean': model_results[metric]['mean'],
            'LCI': model_results[metric]['lci'],
            'UCI': model_results[metric]['uci']
        })
        results_table = pd.DataFrame.append(results_table, row, ignore_index=True)

results_table.to_csv(os.path.join(cfg["cascade_model_evaluation_dir"], 'results_table.csv'), index=False)
