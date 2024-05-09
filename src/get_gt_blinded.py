"""
Create radiologist annotations and edge-detection baseline for saliency test set.
Radiologist names are blinded. Please update for your use.
"""

# Imports
import argparse
import json
import os
import warnings

from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from .image_utils import edge_canny, otsu, bbox

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

# Make radiologist folders
dir_rad1 = os.path.join(cfg["annotations_dir"], "RAD_rad1")
dir_rad2 = os.path.join(cfg["annotations_dir"], "RAD_rad2")
dir_rad3 = os.path.join(cfg["annotations_dir"], "RAD_rad3")
dir_consensus = os.path.join(cfg["annotations_dir"], "RAD_CONSENSUS")
dir_edge = os.path.join(cfg["annotations_dir"], "BASELINE")
dir_edge_otsu = os.path.join(cfg["annotations_dir"], "BASELINE_OTSU")
dir_edge_square = os.path.join(cfg["annotations_dir"], "BASELINE_SQUARE")

os.makedirs(dir_rad1)
os.makedirs(dir_rad2)
os.makedirs(dir_rad3)
os.makedirs(dir_consensus)
os.makedirs(dir_edge)
os.makedirs(dir_edge_otsu)
os.makedirs(dir_edge_square)

os.makedirs(os.path.join(dir_rad1, "InceptionV3"))
os.makedirs(os.path.join(dir_rad2, "InceptionV3"))
os.makedirs(os.path.join(dir_rad3, "InceptionV3"))
os.makedirs(os.path.join(dir_consensus, "InceptionV3"))
os.makedirs(os.path.join(dir_edge, "InceptionV3"))
os.makedirs(os.path.join(dir_edge_otsu, "InceptionV3"))
os.makedirs(os.path.join(dir_edge_square, "InceptionV3"))

os.makedirs(os.path.join(dir_rad1, "DenseNet121"))
os.makedirs(os.path.join(dir_rad2, "DenseNet121"))
os.makedirs(os.path.join(dir_rad3, "DenseNet121"))
os.makedirs(os.path.join(dir_consensus, "DenseNet121"))
os.makedirs(os.path.join(dir_edge, "DenseNet121"))
os.makedirs(os.path.join(dir_edge_otsu, "DenseNet121"))
os.makedirs(os.path.join(dir_edge_square, "DenseNet121"))

# Model transforms (using nearest interpolation to maintain binary masks)
# Mimics image transformations that original image undergoes in pre-ML Model pipeline (excludes normalization)
img_size = cfg["image_size"]
InceptionV3_mask_tfms = transforms.Compose([
     transforms.ToTensor(), # converts to 0-1
     transforms.Resize((img_size, img_size),interpolation=transforms.InterpolationMode.NEAREST),
     transforms.CenterCrop((img_size, img_size))])
DenseNet121_mask_tfms = transforms.Compose([
     transforms.ToTensor(), # converts to 0-1
     transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
     transforms.CenterCrop((img_size, img_size))])

# Load radiologist jsons as extracted 
# from_dir = "/export/gaon1/data/kvenka10/explain-mura/annotations"
from_dir = cfg["subgroup_labels_dir"]
with open(os.path.expanduser(os.path.join(from_dir, "rad1_annotations.json"))) as f:
	rad1 = json.load(f)
with open(os.path.expanduser(os.path.join(from_dir, "rad2_annotations.json"))) as f:
	rad2 = json.load(f)
with open(os.path.expanduser(os.path.join(from_dir, "rad3_annotations.json"))) as f:
	rad3 = json.load(f)

def get_maps(imgs, rad, rad_json, c_InceptionV3_masks=None, c_DenseNet121_masks=None):
    '''Note: coordinate conventions are tricky.
    In NumPy,
    --------y
    |
    |
    |
    |
    x
    In VGG Annotator and CV2,
    --------x
    |
    |
    |
    |
    y
    '''
    dir_rad = ""
    if rad == "rad1":
        dir_rad = dir_rad1
    elif rad == "rad2":
        dir_rad = dir_rad2
    elif rad == "rad3":
        dir_rad = dir_rad3
    else:
        exit("Invalid radiologist provided.")
    for im in tqdm(imgs):
        image = Image.open(os.path.join(image_dir, im)).convert("1")
        image = np.asarray(image)
        mask = np.zeros(image.shape)
        if len(rad_json[im]["regions"]) != 0:
            for region in rad_json[im]["regions"]:
                shape = region["shape_attributes"]
                if shape["name"] == "polygon":
                    # Note coordinate flip
                    ys = list(shape["all_points_x"])
                    xs = list(shape["all_points_y"])
                    # Note coordinate flip
                    cv2_vertices = [[y, x] for x, y in zip(xs, ys)]
                    cv2.fillConvexPoly(mask, np.array(cv2_vertices), 1)
                elif shape["name"] == "rect":
                    # Note coordinate flip
                    y0 = shape["x"]
                    y1 = y0 + shape["width"]
                    x0 = shape["y"]
                    x1 = x0 + shape["height"]
                    mask[x0:x1, y0:y1] = 1
        # Binarize the mask
        mask[mask == 1] = 255
        mask[mask != 255] = 0
        # Mask shape: C x H x W
        mask = np.stack([mask, mask, mask])
        # Mask shape: H x W x C (for transforms)
        mask = mask.transpose(1, 2, 0)
        # Run through transforms, shape: Img_size x Img_size
        InceptionV3_mask = InceptionV3_mask_tfms(mask)[0]
        InceptionV3_mask_ndarr = InceptionV3_mask.numpy()
        DenseNet121_mask = DenseNet121_mask_tfms(mask)[0]
        DenseNet121_mask_ndarr = DenseNet121_mask.numpy()
        # Unique values are 0,1
        InceptionV3_mask_ndarr = InceptionV3_mask_ndarr * 1/255
        InceptionV3_mask_ndarr = InceptionV3_mask_ndarr.astype(np.uint8)
        DenseNet121_mask_ndarr = DenseNet121_mask_ndarr * 1/255
        DenseNet121_mask_ndarr = DenseNet121_mask_ndarr.astype(np.uint8)
        np.save(os.path.join(dir_rad, "InceptionV3", im), InceptionV3_mask_ndarr)
        np.save(os.path.join(dir_rad, "DenseNet121", im), DenseNet121_mask_ndarr)
        if not c_InceptionV3_masks is None:
            c_InceptionV3_masks[im] += InceptionV3_mask_ndarr
        if not c_DenseNet121_masks is None:
            c_DenseNet121_masks[im] += DenseNet121_mask_ndarr

# Get annotation images for saliency test set
with open(cfg['saliency_test_images'], 'r') as f:
        # Ignore the index + subgroup label + pair index in file
        annot_imgs = [x.strip('\n').split(",")[0] for x in f.readlines()]
assert len(annot_imgs) == 588  

# Create consensus maps (c_masks)
c_InceptionV3_masks = {x: np.zeros((img_size,img_size)) for x in annot_imgs}
c_DenseNet121_masks = {x: np.zeros((img_size,img_size)) for x in annot_imgs}

# Save rad1 masks
print("Starting to save rad1 masks.")
# clean json
imgs = list(rad1.keys())
for x in imgs:
    rad1[x[:x.find(".png")+4]] = rad1.pop(x)
imgs = [x[:x.find(".png")+4] for x in imgs]
# only look at saliency test set
get_maps(annot_imgs, "rad1", rad1, c_InceptionV3_masks=c_InceptionV3_masks, c_DenseNet121_masks=c_DenseNet121_masks)

# Save rad2 masks
print("Starting to save rad2 masks.")
# clean json
imgs = list(rad2.keys())
for x in imgs:
    rad2[x[:x.find(".png")+4]] = rad2.pop(x)
imgs = [x[:x.find(".png")+4] for x in imgs]
# only look at saliency test set
get_maps(annot_imgs, "rad2", rad2, c_InceptionV3_masks=c_InceptionV3_masks, c_DenseNet121_masks=c_DenseNet121_masks)

# Save rad3 masks
print("Starting to save rad3 masks.")
# clean json
imgs = list(rad3.keys())
for x in imgs:
    rad3[x[:x.find(".png")+4]] = rad3.pop(x)
imgs = [x[:x.find(".png")+4] for x in imgs]
# only look at saliency test set
get_maps(annot_imgs, "rad3", rad3, c_InceptionV3_masks=c_InceptionV3_masks, c_DenseNet121_masks=c_DenseNet121_masks)

# Majority vote for consensus maps
print("Starting to save CONSENSUS masks.")
for im in tqdm(annot_imgs):
    mI = c_InceptionV3_masks[im]
    mI[mI < 2] = 0
    mI[mI >= 2] = 1
    c_InceptionV3_masks[im] = mI
    mD = c_DenseNet121_masks[im]
    mD[mD < 2] = 0
    mD[mD >= 2] = 1
    c_DenseNet121_masks[im] = mD
    np.save(os.path.join(dir_consensus, "InceptionV3", im), mI)
    np.save(os.path.join(dir_consensus, "DenseNet121", im), mD)

# Store edge detection baseline masks
print("Starting to save BASELINE masks.")
for im in tqdm(annot_imgs):
    image = Image.open(os.path.join(image_dir, im)).convert('RGB')
    # Apply training transforms (but scale by 255 so that image is between 0-255)
    InceptionV3_img = InceptionV3_mask_tfms(image)
    InceptionV3_img = np.asarray(InceptionV3_img)[0] * 255.0
    InceptionV3_img = InceptionV3_img.astype(np.float64)
    DenseNet121_img_3d = DenseNet121_mask_tfms(image)
    DenseNet121_img_3d = np.asarray(DenseNet121_img_3d)[0] * 255.0
    # Edge detection (default low threshold = 30)
    InceptionV3_edges = edge_canny(InceptionV3_img, low_threshold=edge_low_threshold)
    DenseNet121_edges = edge_canny(DenseNet121_img_3d, low_threshold=edge_low_threshold)
    # Scale to 0,1
    InceptionV3_edges = InceptionV3_edges * 1/255.0
    DenseNet121_edges = DenseNet121_edges * 1/255.0
    np.save(os.path.join(dir_edge, "InceptionV3", im), InceptionV3_edges)
    np.save(os.path.join(dir_edge, "DenseNet121", im), DenseNet121_edges)
    # Otsu edges (redundant because images are binary, only done for consistency)
    InceptionV3_edges_otsu = otsu(InceptionV3_edges)
    DenseNet121_edges_otsu = otsu(DenseNet121_edges)
    np.save(os.path.join(dir_edge_otsu, "InceptionV3", im), InceptionV3_edges_otsu)
    np.save(os.path.join(dir_edge_otsu, "DenseNet121", im), DenseNet121_edges_otsu)
    # Square edges
    InceptionV3_edges_square = bbox(InceptionV3_edges_otsu)
    DenseNet121_edges_square = bbox(DenseNet121_edges_otsu)
    np.save(os.path.join(dir_edge_square, "InceptionV3", im), InceptionV3_edges_square)
    np.save(os.path.join(dir_edge_square, "DenseNet121", im), DenseNet121_edges_square)
