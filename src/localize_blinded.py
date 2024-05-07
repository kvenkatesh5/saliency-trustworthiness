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

from .image_utils import auprc

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

# Device
device = 'cpu'
ncpus = os.cpu_count()
dev_n = ncpus
if torch.cuda.is_available():
    device = 'cuda'
    dev_n = torch.cuda.device_count()
print('\nDevice: {} #: {} #cpus: {}\n'.format(device, dev_n, ncpus))

# Image paths
subgroups = {}
with open(cfg['saliency_test_images'], 'r') as f:
        # Extract the image basename and subgroup label
        unpacked = [x.strip('\n').split(",") for x in f.readlines()]
        unpacked = [(x[0], x[2].strip(" ")) for x in unpacked]
        image_paths, categories = zip(*unpacked)
        image_paths = list(image_paths)
        categories = list(categories)
        im_to_category = {
            image_paths[i]: categories[i] for i in range(len(image_paths))
        }

# Dictionary to store results
"""
Model: Best performing InceptionV3 at each subset %
Saliency Method: GCAM, GRAD, ..., XRAI
Preprocessing Mode: raw, otsu, bbox (this includes binary opening)
Image: [image base name]
AUPRC
"""
scores = {"Model": [], "Saliency Method": [], "Preprocessing Mode": [], "Image": [], "Subgroup": [], "AUPRC": []}

# Best performing InceptionV3 models
cnn_list = ["InceptionV3_1_100", "InceptionV3_1_50", "InceptionV3_1_10", "InceptionV3_2_1"]
methods = [('gcam', 'GCAM'), ('grad', 'GRAD'), ('ig', 'IG'), ('sg', 'SG'), ('sig', 'SIG'), ('xrai', 'XRAI')]
# empty string is raw mode
modes = ["", "_otsu", "_bbox"]

for cnn in cnn_list:
    for mth in methods:
        for mode in modes:
            sal_maps = np.load(os.path.join(cfg["heatmap_dir"], cnn, "{}{}.npy".format(mth[0], mode)))
            print("CNN: {} | MTH: {} | MODE: {}".format(cnn, mth[1], ("_raw" if mode == "" else mode)))
            for i, im in enumerate(tqdm(image_paths)):
                gt = np.load(os.path.join(cfg["annotations_dir"], "RAD_CONSENSUS/InceptionV3", im + ".npy"))
                s = auprc(gt.flatten(), sal_maps[i].flatten())
                scores["Model"].append(cnn)
                scores["Saliency Method"].append(mth[1])
                scores["Preprocessing Mode"].append(("_raw" if mode == "" else mode))
                scores["Image"].append(im)
                scores["Subgroup"].append(im_to_category[im])
                scores["AUPRC"].append(s)

# Baselines
baseline_list = ["BASELINE", "BASELINE_OTSU", "BASELINE_SQUARE"]
# recall that baseline SQUARE does not include binary opening, see get_gt and unittest/binary_opening

for j, bsl in enumerate(baseline_list):
    print("{}".format(bsl))
    for i, im in enumerate(tqdm(image_paths)):
        bsl_map = np.load(os.path.join(cfg["annotations_dir"], bsl, "InceptionV3", im + ".npy"))
        gt = np.load(os.path.join(cfg["annotations_dir"], "RAD_CONSENSUS/InceptionV3", im + ".npy"))
        s = auprc(gt.flatten(), bsl_map.flatten())
        mode = modes[j]
        scores["Model"].append(bsl)
        scores["Saliency Method"].append("n/a")
        scores["Preprocessing Mode"].append(("_raw" if mode == "" else mode))
        scores["Image"].append(im)
        scores["Subgroup"].append(im_to_category[im])
        scores["AUPRC"].append(s)

# Inter-radiologist comparisons
rad_list = ["rad1", "rad2", "rad3"]
for j, rad in enumerate(rad_list):
    print("RAD {}".format(rad))
    for i, im in enumerate(tqdm(image_paths)):
        rad_map = np.load(os.path.join(cfg["annotations_dir"], "RAD_{}".format(rad), "InceptionV3", im + ".npy"))
        gt = np.load(os.path.join(cfg["annotations_dir"], "RAD_CONSENSUS/InceptionV3", im + ".npy"))
        s = auprc(gt.flatten(), rad_map.flatten())
        scores["Model"].append(rad)
        scores["Saliency Method"].append("n/a")
        scores["Preprocessing Mode"].append("n/a")
        scores["Image"].append(im)
        scores["Subgroup"].append(im_to_category[im])
        scores["AUPRC"].append(s)

df = pd.DataFrame().from_dict(data=scores)
assert len(df) == (4 * 6 * 3 * 588 + 3 * 588 + 3 * 588)
df.to_csv(
     os.path.join(cfg["trustworthiness_dir"], "heatmap_localizations.csv"),
     index=False
)
