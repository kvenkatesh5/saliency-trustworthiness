"""
Similarity plots (figure 4 in manuscript)
"""
# Imports
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
parser.add_argument("--use-gpus", default='all', type=str, help='')
args = parser.parse_args()

# Get cfg
with open(args.cfg_dir.replace('~', os.path.expanduser('~')), 'r') as f:
    cfg = json.load(f)
cfg['data_dir'] = cfg['data_dir'].replace('~', os.path.expanduser('~'))

# Load similarity scores
df = pd.read_csv(
    os.path.join(cfg["trustworthiness_dir"], "heatmap_similarity.csv"),
)
assert len(df) == (8 * 6 * 3 * 588 + 3 * 588)
# InceptionV3 models
methods = [('gcam', 'GCAM'), ('grad', 'GRAD'), ('ig', 'IG'), ('sg', 'SG'), ('sig', 'SIG'), ('xrai', 'XRAI')]
# here, raw mode is not empty string
baselines = ["BASELINE", "BASELINE_OTSU", "BASELINE_SQUARE"]
subgroups = ["arthritis", "hardware/fracture", "other"]

# Calculate threshold df
tdf = df[(df["Model-2"] == "GroundTruth")].groupby("Image", as_index=False).mean()

"""
Plot of repeatability scores.
"""
s_repeat = df[
    ((df["Model-1"]=="InceptionV3_1_100") & (df["Model-2"]=="InceptionV3_2_100") &\
     (df["Preprocessing Mode"]=="_bbox"))
][["Saliency Method", "SSIM", "IoU", "Pixel Error"]]
fig, ax = plt.subplots(1,3, figsize=(16,9))
# SSIM
sns.boxplot(data=s_repeat, x="Saliency Method", y="SSIM", showfliers=False, ax=ax[0])
ax[0].axhline(y=0.5, linestyle="--", color="black")
# IoU
sns.boxplot(data=s_repeat, x="Saliency Method", y="IoU", showfliers=False, ax=ax[1])
ax[1].axhline(y=0.5, linestyle="--", color="black")
# Pixel Error
sns.boxplot(data=s_repeat, x="Saliency Method", y="Pixel Error", showfliers=False, ax=ax[2])
ax[2].axhline(y=0.5, linestyle="--", color="black")
# beautify
for axis in ax:
    axis.xaxis.label.set_visible(False)
plt.tight_layout()
# plt.savefig(os.path.join(cfg["tmp_dir"], "plot4m-1.png"), dpi=1200)
plt.savefig(os.path.join(cfg["trustworthiness_dir"], "repeatability_plot.png"), dpi=1200)

"""
Plot of reproducibility scores.
"""
s_reproduce = df[
    ((df["Model-1"]=="InceptionV3_1_100") & (df["Model-2"]=="DenseNet169_1_100") &\
     (df["Preprocessing Mode"]=="_bbox"))
][["Saliency Method", "SSIM", "IoU", "Pixel Error"]]
fig, ax = plt.subplots(1,3, figsize=(16,9))
# SSIM
sns.boxplot(data=s_reproduce, x="Saliency Method", y="SSIM", showfliers=False, ax=ax[0])
ax[0].axhline(y=tdf["SSIM"].mean(), linestyle="--", color="black")
# IoU
sns.boxplot(data=s_reproduce, x="Saliency Method", y="IoU", showfliers=False, ax=ax[1])
ax[1].axhline(y=tdf["IoU"].mean(), linestyle="--", color="black")
# Pixel Error
sns.boxplot(data=s_reproduce, x="Saliency Method", y="Pixel Error", showfliers=False, ax=ax[2])
ax[2].axhline(y=tdf["Pixel Error"].mean(), linestyle="--", color="black")
# beautify
for axis in ax:
    axis.xaxis.label.set_visible(False)
plt.tight_layout()
# plt.savefig(os.path.join(cfg["tmp_dir"], "plot4m-2.png"), dpi=1200)
plt.savefig(os.path.join(cfg["trustworthiness_dir"], "reproducibility_plot.png"), dpi=1200)