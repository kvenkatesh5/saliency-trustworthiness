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

from .image_utils import resize_batch, SSIM, IoU, dice, binary_error

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
Model1: InceptionV3_1_100, InceptionV3_1_50, InceptionV3_1_10, InceptionV3_1_1
Model2: InceptionV3_2_100, InceptionV3_2_50, InceptionV3_2_10, InceptionV3_2_1
Saliency Method: GCAM, GRAD, ..., XRAI
Preprocessing Mode: raw, otsu, bbox (this includes binary opening)
Image: [image base name]
SSIM
IoU
Dice
Binary Pixel Error
"""
scores = {"Model-1": [], "Model-2": [], "Saliency Method": [], "Preprocessing Mode": [], "Image": [], "Subgroup": [], 
          "SSIM": [], "IoU": [], "Dice": [], "Pixel Error": []}

cnn_list = [
    # repeatability set
    # compare best two performing Inception models
    ("InceptionV3_1_100", "InceptionV3_2_100"),
    ("InceptionV3_1_50", "InceptionV3_3_50"),
    ("InceptionV3_1_10", "InceptionV3_2_10"),
    ("InceptionV3_2_1", "InceptionV3_1_1"),
    # reproducibility set
    # compare best Inception to the DenseNet169
    ("InceptionV3_1_100", "DenseNet169_1_100"),
    ("InceptionV3_1_50", "DenseNet169_1_50"),
    ("InceptionV3_1_10", "DenseNet169_1_10"),
    ("InceptionV3_2_1", "DenseNet169_1_1"),
]
methods = [('gcam', 'GCAM'), ('grad', 'GRAD'), ('ig', 'IG'), ('sg', 'SG'), ('sig', 'SIG'), ('xrai', 'XRAI')]
# empty string is raw mode
modes = ["", "_otsu", "_bbox"]

for cnn in cnn_list:
    for mth in methods:
        for mode in modes:
            cnn1, cnn2 = cnn
            sal_maps1 = np.load(os.path.join(cfg["heatmap_dir"], cnn1, "{}{}.npy".format(mth[0], mode)))
            sal_maps2 = np.load(os.path.join(cfg["heatmap_dir"], cnn2, "{}{}.npy".format(mth[0], mode)))
            # resize
            if np.shape(sal_maps2) != np.shape(sal_maps1):
                to_size = np.shape(sal_maps1)[1]
                sal_maps2 = resize_batch(sal_maps1, to_size)
            print("CNNs: {} | MTH: {} | MODE: {}".format(cnn, mth[1], ("_raw" if mode == "" else mode)))
            for i, im in enumerate(tqdm(image_paths)):
                s_ssim = SSIM(sal_maps1[i], sal_maps2[i])
                s_iou = IoU(sal_maps1[i], sal_maps2[i])
                s_dice = dice(sal_maps1[i], sal_maps2[i])
                s_pxlerr = binary_error(sal_maps1[i], sal_maps2[i])
                scores["Model-1"].append(cnn1)
                scores["Model-2"].append(cnn2)
                scores["Saliency Method"].append(mth[1])
                scores["Preprocessing Mode"].append(("_raw" if mode == "" else mode))
                scores["Image"].append(im)
                scores["Subgroup"].append(im_to_category[im])
                scores["SSIM"].append(s_ssim)
                scores["IoU"].append(s_iou)
                scores["Dice"].append(s_dice)
                scores["Pixel Error"].append(s_pxlerr)

# Inter-radiologist comparisons
rad_list = ["rad1", "rad2", "rad3"]
for j, rad in enumerate(rad_list):
    print("RAD {}".format(rad))
    for i, im in enumerate(tqdm(image_paths)):
        rad_map = np.load(os.path.join(cfg["annotations_dir"], "RAD_{}".format(rad), "InceptionV3", im + ".npy"))
        rad_map = rad_map.astype(np.float64)
        gt = np.load(os.path.join(cfg["annotations_dir"], "RAD_CONSENSUS/InceptionV3", im + ".npy"))
        s_ssim = SSIM(rad_map, gt)
        s_iou = IoU(rad_map, gt)
        s_dice = dice(rad_map, gt)
        s_pxlerr = binary_error(rad_map, gt)
        scores["Model-1"].append(rad)
        scores["Model-2"].append("GroundTruth")
        scores["Saliency Method"].append("n/a")
        scores["Preprocessing Mode"].append("n/a")
        scores["Image"].append(im)
        scores["Subgroup"].append(im_to_category[im])
        scores["SSIM"].append(s_ssim)
        scores["IoU"].append(s_iou)
        scores["Dice"].append(s_dice)
        scores["Pixel Error"].append(s_pxlerr)

df = pd.DataFrame().from_dict(data=scores)
assert len(df) == (8*6*3*588 + 3*588)
df.to_csv(
     os.path.join(cfg["trustworthiness_dir"], "heatmap_similarity.csv"),
     index=False
)
