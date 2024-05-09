"""
This test checks to see if the saliency processing transforms are correctly applied.
To keep this computationally less-demanding, we query a random image.
"""

# Imports
import argparse
import json
import random
import os
import warnings

import numpy as np
import saliency.core as saliency
import torch

from ..image_utils import otsu, bbox, binary_open

warnings.filterwarnings("ignore", category=UserWarning)

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--cfg-dir", default='cfg.json', type=str, help='')
parser.add_argument("--images-file", default="test_100_pos.txt", type=str, help='file with image paths to analyze, relative path')
parser.add_argument("--model-name", default='InceptionV3_1_100', type=str, help='display name')
parser.add_argument("--from-dir", default="/export/gaon1/data/kvenka10/explain-mura/results/mapsP", type=str, help='directory from where maps will be loaded')
parser.add_argument("--use-gpus", default='all', type=str, help='')
args = parser.parse_args()

# Set GPU vis
if args.use_gpus != 'all':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.use_gpus

# Device
device = 'cpu'
ncpus = os.cpu_count()
dev_n = ncpus
if torch.cuda.is_available():
    device = 'cuda'
    dev_n = torch.cuda.device_count()
print('\nDevice: {} #: {} #cpus: {}\n'.format(device, dev_n, ncpus))

# Get cfg
with open(args.cfg_dir.replace('~', os.path.expanduser('~')), 'r') as f:
    cfg = json.load(f)
cfg['data_dir'] = cfg['data_dir'].replace('~', os.path.expanduser('~'))
cfg['model_dir'] = cfg['model_dir'].replace('~', os.path.expanduser('~'))

# Set target dir
dataset_dir = os.path.join(cfg['data_dir'], "mura")
image_dir = os.path.join(dataset_dir, 'images')

# Get test images
with open(os.path.join(dataset_dir, args.images_file), 'r') as f:
    image_paths = [x.strip('\n') for x in f.readlines()]
    image_paths = [os.path.join(image_dir, x) for x in image_paths]

# Random image index
random.seed(42)
index = random.randrange(0,len(image_paths))

# Load maps
methods = [('gcam', 'GCAM'), ('grad', 'GRAD'), ('ig', 'IG'), ('sg', 'SG'), ('sig', 'SIG'), ('xrai', 'XRAI')]
raw_maps = [
    np.load(os.path.join(args.from_dir, args.model_name, 'gcam.npy')),
    np.load(os.path.join(args.from_dir, args.model_name, 'grad.npy')),
    np.load(os.path.join(args.from_dir, args.model_name, 'ig.npy')),
    np.load(os.path.join(args.from_dir, args.model_name, 'sg.npy')),
    np.load(os.path.join(args.from_dir, args.model_name, 'sig.npy')),
    np.load(os.path.join(args.from_dir, args.model_name, 'xrai.npy'))
]

otsu_maps = [
    np.load(os.path.join(args.from_dir, args.model_name, 'gcam_otsu.npy')),
    np.load(os.path.join(args.from_dir, args.model_name, 'grad_otsu.npy')),
    np.load(os.path.join(args.from_dir, args.model_name, 'ig_otsu.npy')),
    np.load(os.path.join(args.from_dir, args.model_name, 'sg_otsu.npy')),
    np.load(os.path.join(args.from_dir, args.model_name, 'sig_otsu.npy')),
    np.load(os.path.join(args.from_dir, args.model_name, 'xrai_otsu.npy'))
]

bbox_maps = [
    np.load(os.path.join(args.from_dir, args.model_name, 'gcam_bbox.npy')),
    np.load(os.path.join(args.from_dir, args.model_name, 'grad_bbox.npy')),
    np.load(os.path.join(args.from_dir, args.model_name, 'ig_bbox.npy')),
    np.load(os.path.join(args.from_dir, args.model_name, 'sg_bbox.npy')),
    np.load(os.path.join(args.from_dir, args.model_name, 'sig_bbox.npy')),
    np.load(os.path.join(args.from_dir, args.model_name, 'xrai_bbox.npy'))
]

# Test the transforms
print("...Testing transforms on image index {} from {}...".format(index, args.images_file))
for i in range(len(methods)):
    m1 = raw_maps[i][index]
    m2 = otsu_maps[i][index]
    m3 = bbox_maps[i][index]
    assert np.all(otsu(m1) == m2), "Otsu transform did not match for method {}".format(methods[i][1])
    assert np.all(binary_open(bbox(m2)) == m3), "Bbox transform did not match for method {}".format(methods[i][1])
print("Test passed")