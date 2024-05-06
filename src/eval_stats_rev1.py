"""
Stats compare the model performances.
"""

import argparse
import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--cfg-dir", default='cfg.json', type=str, help='')
parser.add_argument("--results-file", default='/export/gaon2/data/kvenka10/explain-mura/results/model_evaluation/results_table.csv',
                    type=str, help='results table stored from eval.py')
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

# Load auc_weighted
# Load the .npz file
npzfile = np.load(os.path.join(cfg["tmp_dir"], 'auc_weighted_lists.npz'), allow_pickle=True)
# Extract the dictionary of arrays
auc_weighted_lists = dict(npzfile['d'].item())  # Convert to regular dictionary
# Close the file
npzfile.close()

# Load kappa_scores
# Load the .npz file
npzfile = np.load(os.path.join(cfg["tmp_dir"], 'kappa_score_lists.npz'), allow_pickle=True)
# Extract the dictionary of arrays
kappa_score_lists = dict(npzfile['d'].item())  # Convert to regular dictionary
# Close the file
npzfile.close()

# Pairwise tukeyHSD to compare models
keys = ["InceptionV3_1_100", "InceptionV3_2_100", "InceptionV3_3_100", "DenseNet169_1_100"]
all_auc_weighted = np.hstack([auc_weighted_lists[k] for k in keys])
# n=1000 bootstrap
groups = []
for i in range(len(keys)):
    groups += [keys[i] for _ in range(1000)]
auc_weighted_test = pairwise_tukeyhsd(endog=all_auc_weighted, groups=groups)
print(auc_weighted_test)

all_kappa_score = np.hstack([kappa_score_lists[k] for k in keys])
kappa_score_test = pairwise_tukeyhsd(endog=all_kappa_score, groups=groups)
print(kappa_score_test)