"""
Model eval plots.
"""

import argparse
import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Load results table
results_table = pd.read_csv(args.results_file)

# Evaluation plot
# change model list as needed for plotting
# model = ['InceptionV3_1_100', 'InceptionV3_1_50', 'InceptionV3_1_10', 'InceptionV3_1_1']
model = ['InceptionV3_1_100', 'InceptionV3_2_100', 'InceptionV3_3_100', 'DenseNet169_1_100']

dataset = 'mura_test_100'
dataset_nicenames = "MURA"
finding = regions
metric = ['auc']
dat_plot = results_table[
       (results_table["Model"].isin(model)) & \
       (results_table["Data set"] == dataset) & \
       (results_table['Finding'].isin(finding)) & \
       (results_table["Metric"].isin(metric))
]

# By test set
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
shift = np.arange(-((len(model) - 1) / 2), (len(model) - 1) / 2 + 1, 1)
width = 0.8 / len(model)
labels_sub = dat_plot[(dat_plot['Data set'] == dataset) & (~dat_plot['Mean'].isna())]['Finding'].drop_duplicates()
fig, ax = plt.subplots(figsize=(9,12))

for i, h in enumerate(model):
    dat_sub = dat_plot[(dat_plot['Model'] == h) & (dat_plot['Data set'] == dataset) & (~dat_plot['Mean'].isna())]
    plt.bar(x=[x + shift[i] * width for x in range(len(labels_sub))],
            height=dat_sub['Mean'],
            yerr=[dat_sub['Mean'] - dat_sub['LCI'], dat_sub['UCI'] - dat_sub['Mean']],
            width=width,
            zorder=5,
            color=colors[i],
            label=h)

    weighted_mean = results_table[(results_table['Model'] == h) &
                                    (results_table['Data set'] == dataset) &
                                    (results_table['Finding'] == 'Overall') &
                                    (results_table['Metric'] == 'auc_weighted')
                                    ]
    plt.axhline(weighted_mean['Mean'].tolist()[0], zorder=4, color=colors[i], label=None, linewidth=1)

ax.set_xticks(range(len(labels_sub)))
ax.set_xticklabels([x.replace('_', ' ').capitalize() for x in labels_sub], rotation=45)

ax.set_ylabel('AURPC')
ax.set_xlabel('Anatomy')
ax.set_ylim(0.5, 1)
ax.legend([m.replace("_100", "") for m in model], loc='lower left')
ax.grid(True, axis='y', zorder=0)

plt.tight_layout()
# fig.savefig(os.path.join(cfg["tmp_dir"], "TT_rev1.png"))
fig.savefig(os.path.join(cfg["trustworthiness_dir"], "eval_models_plot1.png"), dpi=1200)
