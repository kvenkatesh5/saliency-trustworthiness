"""
Compute similarity scores over cascading randomization.
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
with open(cfg['cascade_test_images'], 'r') as f:
    # Extract the image basename and subgroup label
    unpacked = [x.strip('\n').split(",") for x in f.readlines()]
    cascade_image_paths = [x[0] for x in unpacked]
    # Index lookup: pair index --> cascade index (i.e., index in cascade_test_set.csv file)
    u2 = [(int(x[1]), int(x[3])) for x in unpacked]
    index_lookup = {i: [] for i in range(50)}
    for j in range(len(u2)):
        pair_index = u2[j][1]
        index_lookup[pair_index].append(j)
    for k in range(50):
        assert len(index_lookup[k]) == 2
    # Index map: cascade index --> saliency test set index
    index_map = {}
    for j in range(len(u2)):
        index_map[j] = u2[j][0]

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
Scores for computing the degradation threshold
Model: Three InceptionV3 models at the subset %
Saliency Method: GCAM, GRAD, ..., XRAI
Preprocessing Mode: raw, otsu, bbox (this includes binary opening)
Image: [image base name]
SSIM
"""
internal_scores = {"Subset": [], "Model": [], "Saliency Method": [], "Preprocessing Mode": [], \
          "Cascade Test Set Image Pair": [], "SSIM": []}

subsets = [100, 50, 10, 1]
methods = [('gcam', 'GCAM'), ('grad', 'GRAD'), ('ig', 'IG'), ('sg', 'SG'), ('sig', 'SIG'), ('xrai', 'XRAI')]
# empty string is raw mode
modes = ["", "_otsu", "_bbox"]
layer_randomization_order_names = [
    'fc',
    'Mixed_7c', 'Mixed_7b', 'Mixed_7a',
    'Mixed_6e', 'Mixed_6d', 'Mixed_6c', 'Mixed_6b', 'Mixed_6a',
    'Mixed_5d', 'Mixed_5c', 'Mixed_5b',
    'Conv2d_4a_3x3', 'Conv2d_3b_1x1', 'Conv2d_2b_3x3', 'Conv2d_2a_3x3', 'Conv2d_1a_3x3'
]

# baseline (fully trained, non randomized) model
# used to compute degradation thresholds
for subset in subsets:
    cnn_list = ["InceptionV3_{}_{}".format(i+1, subset) for i in range(0,3)]
    for base_cnn in cnn_list:
        for mth in methods:
            for mode in modes:
                # note this saliency map array is of size 588 x imsize x imsize
                sal_maps = np.load(os.path.join(cfg["heatmap_dir"], base_cnn, "{}{}.npy".format(mth[0], mode)))
                print("CNN: {} | MTH: {} | MODE: {}".format(base_cnn, mth[1], ("_raw" if mode == "" else mode)))
                for pair_index in tqdm(range(50)):
                    # look up index using pair index
                    index1, index2 = index_lookup[pair_index]
                    # map index to saliency test set index
                    sts_index1 = index_map[index1]
                    sts_index2 = index_map[index2]
                    # retrieve image name from saliency test set
                    im1 = image_paths[sts_index1]
                    im2 = image_paths[sts_index2]
                    s_ssim = SSIM(sal_maps[sts_index1], sal_maps[sts_index2])
                    internal_scores["Subset"].append(subset)
                    internal_scores["Model"].append(base_cnn)
                    internal_scores["Saliency Method"].append(mth[1])
                    internal_scores["Preprocessing Mode"].append(("_raw" if mode == "" else mode))
                    internal_scores["Cascade Test Set Image Pair"].append(pair_index)
                    internal_scores["SSIM"].append(s_ssim)

df = pd.DataFrame().from_dict(data=internal_scores)
df.to_csv(
     os.path.join(cfg["trustworthiness_dir"], "heatmap_sensitivity_threshold_scores.csv"),
     index=False
)

"""
Scores for computing model sensitivity
Model: Three InceptionV3 models at the subset %
Saliency Method: GCAM, GRAD, ..., XRAI
Preprocessing Mode: raw, otsu, bbox (this includes binary opening)
Image: [image base name]
SSIM
"""

scores = {"Subset": [], "Model": [], "Cascade Layer": [], "Saliency Method": [], "Preprocessing Mode": [], \
          "Image": [], "SSIM": []}
# cascaded models
for subset in subsets:
    cnn_list = ["InceptionV3_{}_{}".format(i+1, subset) for i in range(0,3)]
    for mth in methods:
        for mode in modes:
            for base_cnn in cnn_list:
                # note this saliency map array is of size 588 x imsize x imsize
                base_sal_maps = np.load(os.path.join(cfg["heatmap_dir"], base_cnn, "{}{}.npy".format(mth[0], mode)))
                for layer in layer_randomization_order_names:
                    cnn = base_cnn + "_" + layer
                    # note this saliency map array is of size 100 x imsize x imsize
                    sal_maps = np.load(os.path.join(cfg["heatmap_dir"], cnn, "{}{}.npy".format(mth[0], mode)))
                    print("CNN: {} | LAYER: {} | MTH: {} | MODE: {}".format(cnn, layer, mth[1], ("_raw" if mode == "" else mode)))
                    for index in tqdm(range(sal_maps.shape[0])):
                        sts_index = index_map[index]
                        im = cascade_image_paths[index]
                        assert im == image_paths[sts_index]
                        s_ssim = SSIM(base_sal_maps[sts_index], sal_maps[index])
                        scores["Subset"].append(subset)
                        scores["Model"].append(base_cnn)
                        scores["Cascade Layer"].append(layer)
                        scores["Saliency Method"].append(mth[1])
                        scores["Preprocessing Mode"].append(("_raw" if mode == "" else mode))
                        scores["Image"].append(im)
                        scores["SSIM"].append(s_ssim)

df2 = pd.DataFrame().from_dict(data=scores)
df2.to_csv(
     os.path.join(cfg["trustworthiness_dir"], "heatmap_cascading_randomization_scores.csv"),
     index=False
)