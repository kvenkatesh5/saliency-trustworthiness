"""
Heatmap sensitivity experiments.
"""
# Imports
import argparse
import json
import os
import sys
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import ttest_1samp

from .image_utils import auprc, SSIM

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

# Load threshold scores
df = pd.read_csv(
    os.path.join(cfg["trustworthiness_dir"], "heatmap_sensitivity_threshold_scores.csv"),
)

# Load cascading randomization scores
df2 = pd.read_csv(
    os.path.join(cfg["trustworthiness_dir"], "heatmap_cascading_randomization_scores.csv"),
)
alpha = 0.05
subsets = [100, 50, 10, 1]
methods = [('gcam', 'GCAM'), ('grad', 'GRAD'), ('ig', 'IG'), ('sg', 'SG'), ('sig', 'SIG'), ('xrai', 'XRAI')]
modes = ["_raw", "_otsu", "_bbox"]
layer_randomization_order_names = [
    'fc',
    'Mixed_7c', 'Mixed_7b', 'Mixed_7a',
    'Mixed_6e', 'Mixed_6d', 'Mixed_6c', 'Mixed_6b', 'Mixed_6a',
    'Mixed_5d', 'Mixed_5c', 'Mixed_5b',
    'Conv2d_4a_3x3', 'Conv2d_3b_1x1', 'Conv2d_2b_3x3', 'Conv2d_2a_3x3', 'Conv2d_1a_3x3'
]

sensitivity_tests = {
    "CNN Training Subset %": [],
    "Method": [],
    "Mode": [],
    "Cascade Layer": [],
    "Mean SSIM": [],
    "Std SSIM": [],
    "Threshold SSIM": [],
    "Layer-wise P-value": [],
    "Sensitivity": [],
}

for subset in subsets:
    for mth in methods:
        for mode in modes:
            # if subset != 100 or mode != "_bbox": continue
            # get degradation threshold
            subdf = df[((df["Subset"]==subset) & \
                                    (df["Saliency Method"]==mth[1]) & \
                                    (df["Preprocessing Mode"]==mode))].copy()
            assert len(subdf) == (3*50)
            s1_ssim = subdf[["Cascade Test Set Image Pair", "SSIM"]]\
                .groupby("Cascade Test Set Image Pair", as_index=False).mean()
            threshold = s1_ssim["SSIM"].mean()
            print(threshold, mth, subset, mode)
            # continue
            for layer in tqdm(layer_randomization_order_names):
                subdf2 = df2[((df2["Subset"]==subset) & \
                                        (df2["Saliency Method"]==mth[1]) & \
                                        (df2["Preprocessing Mode"]==mode) & \
                                            (df2["Cascade Layer"]==layer))].copy()
                assert len(subdf2) == (3*100)
                s2_ssim = subdf2[["Image", "SSIM"]]\
                    .groupby("Image", as_index=False).mean()
                sensitivity_tests["CNN Training Subset %"].append(subset)
                sensitivity_tests['Method'].append(mth[1])
                sensitivity_tests["Mode"].append(mode)
                sensitivity_tests["Cascade Layer"].append(layer)
                sensitivity_tests["Mean SSIM"].append(s2_ssim["SSIM"].mean())
                sensitivity_tests['Std SSIM'].append(s2_ssim["SSIM"].std())
                sensitivity_tests["Threshold SSIM"].append(threshold)
                # Make comparison: is SSIM distribution less than degradation threshold
                s_ssim_test = ttest_1samp(s2_ssim["SSIM"], threshold)
                sensitivity_tests["Layer-wise P-value"].append(
                    s_ssim_test.pvalue/2
                )
                s_ssim_test_result = ((s_ssim_test.pvalue/2 < alpha) & (s_ssim_test.statistic < 0))
                if s_ssim_test_result:
                    sensitivity_tests["Sensitivity"].append(
                        "Pass"
                    )
                else:
                    sensitivity_tests["Sensitivity"].append(
                        "Fail"
                    )
            
df_sensitivity = pd.DataFrame().from_dict(sensitivity_tests)
assert len(df_sensitivity) == (4*6*3*17)
df_sensitivity.to_csv(
     os.path.join(cfg["trustworthiness_dir"], "heatmap_sensitivity_tests.csv"),
     index=False
)
