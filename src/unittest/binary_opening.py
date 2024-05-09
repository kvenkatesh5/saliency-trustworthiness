"""
Visualize impact of binary opening on baseline edge maps.
Generally, the transform will discard a lot of the edge information.
"""

# Imports
import argparse
import json
import os
import random
import warnings

from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from ..image_utils import edge_canny, otsu, bbox, binary_open

warnings.filterwarnings("ignore", category=UserWarning)

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--cfg-dir", default='cfg.json', type=str, help='')
parser.add_argument("--edge-low-thresh", default=30, type=int, help='low baseline for Canny edge detection algorithm')
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

# Set baseline parameter
edge_low_threshold = args.edge_low_thresh

# Model transforms (using nearest interpolation to maintain binary masks)
# Mimics image transformations that original image undergoes in pre-ML Model pipeline (excludes normalization)
InceptionV3_mask_tfms = transforms.Compose([transforms.ToTensor(), transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.NEAREST), transforms.CenterCrop((299, 299))])
DenseNet121_mask_tfms = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST), transforms.CenterCrop((224, 224))])
InceptionV3_mask_tfms2 = transforms.Compose([transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.NEAREST), transforms.CenterCrop((299, 299))])
DenseNet121_mask_tfms2 = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST), transforms.CenterCrop((224, 224))])

# Get annotation images for saliency test set
with open(cfg['saliency_test_images'], 'r') as f:
        # Ignore the index + subgroup label + pair index in file
        annot_imgs = [x.strip('\n').split(",")[0] for x in f.readlines()]
assert len(annot_imgs) == 588  

random.seed(4)
index = random.randrange(0, len(annot_imgs))

image_3d = Image.open(os.path.join(image_dir, annot_imgs[index]))
# Transform similar to image in CNN (except no mean normalization)
InceptionV3_img_3d = InceptionV3_mask_tfms2(image_3d)
InceptionV3_img_3d = np.asarray(InceptionV3_img_3d)
DenseNet121_img_3d = DenseNet121_mask_tfms2(image_3d)
DenseNet121_img_3d = np.asarray(DenseNet121_img_3d)
# Edge detection
InceptionV3_edges = edge_canny(InceptionV3_img_3d, low_threshold=edge_low_threshold)
DenseNet121_edges = edge_canny(DenseNet121_img_3d, low_threshold=edge_low_threshold)
InceptionV3_edges = InceptionV3_edges * 1/255
DenseNet121_edges = DenseNet121_edges * 1/255
# Otsu edges
InceptionV3_edges_otsu = otsu(InceptionV3_edges)
DenseNet121_edges_otsu = otsu(DenseNet121_edges)
# Square edges
InceptionV3_edges_square = bbox(InceptionV3_edges_otsu)
DenseNet121_edges_square = bbox(DenseNet121_edges_otsu)

plt.imshow(InceptionV3_edges)
plt.savefig("./tmp/edge_or_unittest.png")
plt.clf()

plt.imshow(InceptionV3_edges_otsu)
plt.savefig("./tmp/edge_ot_unittest.png")
plt.clf()

plt.imshow(InceptionV3_edges_square)
plt.savefig("./tmp/edge_sq_unittest.png")
plt.clf()

plt.imshow(binary_open(InceptionV3_edges))
plt.savefig("./tmp/edge_or_open_unittest.png")
plt.clf()

plt.imshow(binary_open(InceptionV3_edges_otsu))
plt.savefig("./tmp/edge_ot_open_unittest.png")
plt.clf()

plt.imshow(binary_open(InceptionV3_edges_square))
plt.savefig("./tmp/edge_sq_open_unittest.png")
plt.clf()
