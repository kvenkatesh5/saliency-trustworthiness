"""
Heatmap sensitivity plots.
"""
# Imports
import argparse
import json
import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .image_utils import SSIM, IoU

warnings.filterwarnings("ignore", category=UserWarning)

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--cfg-dir", default='cfg.json', type=str, help='')
parser.add_argument("--use-gpus", default='all', type=str, help='')
args = parser.parse_args()

# Get cfg
with open(args.cfg_dir.replace('~', os.path.expanduser('~')), 'r') as f:
    cfg = json.load(f)
cfg['data_dir'] = cfg['data_dir'].replace('~', os.path.expanduser('~'))

# Load localization scores
df = pd.read_csv(
    os.path.join(cfg["trustworthiness_dir"], "heatmap_sensitivity_tests.csv"),
)
assert len(df) == (4 * 6 * 3 * 17)

methods = [('gcam', 'GCAM'), ('grad', 'GRAD'), ('ig', 'IG'), ('sg', 'SG'), ('sig', 'SIG'), ('xrai', 'XRAI')]
layer_randomization_order_names = [
    'fc',
    'Mixed_7c', 'Mixed_7b', 'Mixed_7a',
    'Mixed_6e', 'Mixed_6d', 'Mixed_6c', 'Mixed_6b', 'Mixed_6a',
    'Mixed_5d', 'Mixed_5c', 'Mixed_5b',
    'Conv2d_4a_3x3', 'Conv2d_3b_1x1', 'Conv2d_2b_3x3', 'Conv2d_2a_3x3', 'Conv2d_1a_3x3'
]
layer_randomization_order_names_plotting = ['Non-randomized', 'FC', 'Mixed_7c', 'Mixed_7b', 'Mixed_7a', 'Mixed_6e',
                                            'Mixed_6d', 'Mixed_6c', 'Mixed_6b', 'Mixed_6a', 'Mixed_5d', 'Mixed_5c', 'Mixed_5b',
                                            'Conv2d_4a', 'Conv2d_3b', 'Conv2d_2b', 'Conv2d_2a', 'Conv2d_1a']

"""
Plot of cascading randomization scores on 100% subset with bounding box preprocessing.
"""

subdf = df[
    ((df["CNN Training Subset %"]==100) & (df["Mode"]=="_bbox"))
]
fig = plt.figure()
colors = ['r', 'b', 'y', 'g', 'c', 'm']
for j, mth in enumerate(methods):
    assert len(np.unique(subdf[(subdf["Method"] == mth[1])]["Threshold SSIM"])) == 1
    threshold = subdf[(subdf["Method"] == mth[1])]["Threshold SSIM"].iloc[0]
    plt.axhline(threshold, color=colors[j], linestyle='--')

# shift over by 2 (+1 for index, +1 because the first should be 'Original')
x = [i+1 for i in range(len(layer_randomization_order_names_plotting))]
for j, mth in enumerate(methods):
    # populate with 'Original' mean/std
    mean_scores = [1.0]
    std_scores = [0.0]
    for i, layer in enumerate(layer_randomization_order_names):
        mean_score = subdf[(subdf["Method"] == mth[1]) & (subdf["Cascade Layer"]==layer)]["Mean SSIM"].iloc[0]
        std_score = subdf[(subdf["Method"] == mth[1]) & (subdf["Cascade Layer"]==layer)]["Std SSIM"].iloc[0]
        mean_scores.append(mean_score)
        std_scores.append(std_score)
    mean_scores = np.array(mean_scores)
    std_scores = np.array(std_scores)
    plt.plot(x, mean_scores, color=colors[j], label=mth[1])
    # omit for clarity
    # plt.fill_between(x, mean_scores-std_scores, mean_scores+std_scores, color=colors[j], alpha=0.1)

# figure
plt.gcf().subplots_adjust(bottom=0.25)
plt.xticks([i+1 for i in range(len(layer_randomization_order_names_plotting))],\
           layer_randomization_order_names_plotting)
plt.xticks(rotation=90)
# optional vertical line to indicate where randomization starts
# plt.axvline(x=2, color='k', linestyle='--')
plt.xlim([0, len(layer_randomization_order_names_plotting)+1])
# plt.xlabel('Cascade Layer')
plt.ylim([0, 1])
plt.ylabel('SSIM')
# plt.legend(fontsize=5, loc='lower left')
fig.tight_layout()
# plt.savefig(os.path.join(cfg["tmp_dir"], "plot3.png"), dpi=1200)
plt.savefig(os.path.join(cfg["trustworthiness_dir"], "sensitivity_plot1.png"), dpi=1200)

"""
Plot of cascading randomization scores ALONGSIDE average wAUC (mean, LCI, UCI) over InceptionV3s.
Sample case: 100% subset with bounding box preprocessing, GCAM.
"""
# Retrieve wAUCs
settings = {
    "subset": 100,
    "mode": "_bbox",
    "method": "GCAM",
    "metric": "auc_weighted"
}
eval_mean_scores = []
eval_lci_scores = []
eval_uci_scores = []
# get eval scores
for i, layer in enumerate(layer_randomization_order_names_plotting):
    if i==0:
        models = [f"InceptionV3_{i+1}_{settings['subset']}" for i in range(3)]
        # get from model evaluation dir
        df = pd.read_csv(os.path.join(cfg["model_evaluation_dir"], "results_table.csv"))
        subdf = df[(df["Metric"]==settings["metric"]) & (df["Model"].isin(models))]
        eval_mean_scores.append(np.mean(subdf["Mean"]))
        eval_lci_scores.append(np.mean(subdf["LCI"]))
        eval_uci_scores.append(np.mean(subdf["UCI"]))
    else:
        models = [f"InceptionV3_{ii+1}_{settings['subset']}_{layer_randomization_order_names[i-1]}" for ii in range(3)]
        # get from cascade evaluation dir
        df = pd.read_csv(os.path.join(cfg["cascade_model_evaluation_dir"], "results_table.csv"))
        subdf = df[(df["Metric"]==settings["metric"]) & (df["Model"].isin(models))]
        eval_mean_scores.append(np.mean(subdf["Mean"]))
        eval_lci_scores.append(np.mean(subdf["LCI"]))
        eval_uci_scores.append(np.mean(subdf["UCI"]))
# sensitivity scores
# populate with 'Original' mean/std
df = pd.read_csv(
    os.path.join(cfg["trustworthiness_dir"], "heatmap_sensitivity_tests.csv"),
)
subdf = df[
    ((df["CNN Training Subset %"]==100) & (df["Mode"]=="_bbox"))
]
sensitivity_mean_scores = [1.0]
sensitivity_std_scores = [0.0]
for i, layer in enumerate(layer_randomization_order_names):
    mean_score = subdf[(subdf["Method"] == settings["method"]) & (subdf["Cascade Layer"]==layer)]["Mean SSIM"].iloc[0]
    std_score = subdf[(subdf["Method"] == settings["method"]) & (subdf["Cascade Layer"]==layer)]["Std SSIM"].iloc[0]
    sensitivity_mean_scores.append(mean_score)
    sensitivity_std_scores.append(std_score)

threshold = subdf[(subdf["Method"] == settings["method"])]["Threshold SSIM"].iloc[0]

# move arrays to np
eval_mean_scores = np.array(eval_mean_scores)
eval_lci_scores = np.array(eval_lci_scores)
eval_uci_scores = np.array(eval_uci_scores)
sensitivity_mean_scores = np.array(sensitivity_mean_scores)
sensitivity_std_scores = np.array(sensitivity_std_scores)

fig, ax1 = plt.subplots()
fig.subplots_adjust(bottom=0.25)
color1 = 'tab:red'
color2 = 'tab:blue'
# ax1.set_xlabel('Cascade Layer')
ax1.set_ylabel('SSIM', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
x = [i+1 for i in range(len(layer_randomization_order_names_plotting))]

ax1.plot(x, sensitivity_mean_scores, color=color1)
ax1.fill_between(x, sensitivity_mean_scores-sensitivity_std_scores, \
                 sensitivity_mean_scores+sensitivity_std_scores, color=color1, alpha=0.1)
plt.sca(ax1)
plt.xticks(
    ticks=[i+1 for i in range(len(layer_randomization_order_names_plotting))],
    labels=layer_randomization_order_names_plotting,
    rotation=90
)
ax1.set_xlim([0, len(layer_randomization_order_names_plotting)+1])
ax1.set_ylim([0, 1])
ax1.axhline(threshold, color=color1, linestyle='--')


ax2 = ax1.twinx()
ax2.plot(x, eval_mean_scores, color=color2, label="DCNN wAUROC")
ax2.fill_between(x, eval_lci_scores, eval_uci_scores, color=color2, alpha=0.1)
ax2.set_ylabel('wAUROC', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim([0, 1])

# fig.legend(fontsize=5, loc='lower left')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig(os.path.join(cfg["tmp_dir"], "plot4.png"), dpi=1200)
plt.savefig(os.path.join(cfg["trustworthiness_dir"], "sensitivity_plot2.png"), dpi=1200)
