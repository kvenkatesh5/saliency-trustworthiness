"""
S.2 section of paper
"""
import argparse
import json
import os
import sys
from tqdm import tqdm
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image


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

# Image paths for saliency test set
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

# Training transforms (no normalization)
training_tfms = transforms.Compose([transforms.Resize(320),
                                    transforms.CenterCrop(320)])

"""Plot 1: sample saliency map comparison"""
random.seed(2021)
saliency_method = ("gcam", "GCAM")
subset = 100
cnn = f"InceptionV3_1_{subset}"
sample_image_index = random.randint(0, len(image_paths)-1)
sample_image = image_paths[sample_image_index]
print(sample_image)
raw_sal_maps = np.load(os.path.join(cfg["heatmap_dir"], cnn, "{}.npy".format(saliency_method[0])))
# range for ground truth and raw saliency map is 0-1
raw_saliency_map = raw_sal_maps[sample_image_index]
ground_truth = np.load(os.path.join(cfg["annotations_dir"], "RAD_CONSENSUS/InceptionV3", sample_image + ".npy"))
# transform the original radiograph to match training (resize + center crop)
radiograph = Image.open(os.path.join(cfg["data_dir"], "mura", "images", \
                                     sample_image))
tsfm_radiograph = training_tfms(radiograph)
fig, ax = plt.subplots(1,3)
fig.tight_layout()
im0 = ax[0].imshow(tsfm_radiograph)
im1 = ax[1].imshow(ground_truth, vmin=0, vmax=1)
im2 = ax[2].imshow(raw_saliency_map, vmin=0, vmax=1)
# remove the x and y ticks
for axis in ax:
    axis.set_xticks([])
    axis.set_yticks([])
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = fig.add_axes([ax[2].get_position().x1+0.05,ax[2].get_position().y0,0.02,ax[2].get_position().height])
fig.colorbar(im1, cax=cax)
# plt.savefig(os.path.join(cfg["tmp_dir"], "plotS1.png"), dpi=1200)
plt.savefig(os.path.join(cfg["trustworthiness_dir"], "supplemental_plotS1.png"), dpi=1200)

"""Plot 2: saliency map preprocessing routine, using same image as above"""
otsu_sal_maps = np.load(os.path.join(cfg["heatmap_dir"], cnn, "{}_otsu.npy".format(saliency_method[0])))
bbox_sal_maps = np.load(os.path.join(cfg["heatmap_dir"], cnn, "{}_bbox.npy".format(saliency_method[0])))
# range for ground truth and raw saliency map is 0-1
otsu_saliency_map = otsu_sal_maps[sample_image_index]
bbox_saliency_map = bbox_sal_maps[sample_image_index]
fig, ax = plt.subplots(1,5)
im0 = ax[0].imshow(tsfm_radiograph)
ax[0].set_xlabel("Image")
im1 = ax[1].imshow(ground_truth, vmin=0, vmax=1)
ax[1].set_xlabel("Annotation")
im2 = ax[2].imshow(raw_saliency_map, vmin=0, vmax=1)
ax[2].set_xlabel("Raw")
im3 = ax[3].imshow(otsu_saliency_map, vmin=0, vmax=1)
ax[3].set_xlabel("Otsu")
im4 = ax[4].imshow(bbox_saliency_map, vmin=0, vmax=1)
ax[4].set_xlabel("Boxed")
# remove the x and y ticks
for axis in ax:
    axis.set_xticks([])
    axis.set_yticks([])
    axis.xaxis.label.set_visible(False)
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = fig.add_axes([ax[4].get_position().x1+0.05,ax[2].get_position().y0,0.02,ax[2].get_position().height])
fig.colorbar(im1, cax=cax)
# plt.savefig(os.path.join(cfg["tmp_dir"], "plotS2.png"), dpi=1200)
plt.savefig(os.path.join(cfg["trustworthiness_dir"], "supplemental_plotS2.png"), dpi=1200)
