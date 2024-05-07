"""
Individual images/etc needed in overview figure.
"""
import argparse
import json
import os
import sys
from tqdm import tqdm
import random

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
from sklearn.metrics import precision_recall_curve, auc

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

# fxn to plot individual heatmap
def plot_heatmap(image, save_name):
    fig = plt.figure()
    plt.imshow(image, vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["tmp_dir"], save_name), dpi=1200)

# Choose an image (that is in both STS and Cascade TS)
sample_image_index = 524
cascade_sample_image_index = 2
saliency_method = ("sig", "SIG")

sample_image = image_paths[sample_image_index]
print(os.path.join(cfg["data_dir"], "mura", "images", sample_image))
radiograph = Image.open(os.path.join(cfg["data_dir"], "mura", "images", \
                                     sample_image)).convert("L")

# Training transforms (no normalization)
training_tfms = transforms.Compose([transforms.Resize(320),
                                    transforms.CenterCrop(320)])
tsfm_radiograph = training_tfms(radiograph)

# Plot radiograph in grayscale
fig = plt.figure()
plt.imshow(tsfm_radiograph, cmap="gray")
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig(os.path.join(cfg["tmp_dir"], "overview_rad.png"), dpi=1200)

# Annotations
annotations = [
    np.load(os.path.join(cfg["annotations_dir"], "RAD_PAUL/InceptionV3", sample_image + ".npy")),
    np.load(os.path.join(cfg["annotations_dir"], "RAD_FLET/InceptionV3", sample_image + ".npy")),
    np.load(os.path.join(cfg["annotations_dir"], "RAD_SIMI/InceptionV3", sample_image + ".npy"))
]
plot_heatmap(annotations[0], "overview_annot0.png")
plot_heatmap(annotations[1], "overview_annot1.png")
plot_heatmap(annotations[2], "overview_annot2.png")

consensus_annotation = np.load(os.path.join(cfg["annotations_dir"], "RAD_CONSENSUS/InceptionV3", sample_image + ".npy"))
plot_heatmap(consensus_annotation, "overview_consensus.png")

# annotation + radiograph
k = np.logical_and(tsfm_radiograph,consensus_annotation) * 100 + tsfm_radiograph
fig = plt.figure()
plt.imshow(k)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig(os.path.join(cfg["tmp_dir"], "overview_annot_overlap.png"), dpi=1200)

# Load predictions
df_master = pd.read_csv(
    os.path.join(cfg["trustworthiness_dir"], "heatmap_localizations.csv"),
)
assert len(df_master) == (4 * 6 * 3 * 588 + 3 * 588 + 3 * 588)
to_analyze = [{"file": cfg["final_models"][x], "name": x} for x in ["InceptionV3_1_100"]]
correctly_classified = {x: [] for x in ["InceptionV3_1_100"]}
for t in to_analyze:
    path = os.path.join(cfg["model_evaluation_dir"], t["file"])
    try:
        with open(path.replace("_model.pt", ".pkl"), "rb") as f:
            m = pickle.load(f)
    except:
        with open(path.replace(".pt", ".pkl"), "rb") as f:
            m = pickle.load(f)

    # x[1] ~ Positive/Abnormality probability
    dat = pd.DataFrame().from_dict({'y': [int(x[1]) for x in m['mura_test_100']['y']],
                                    'yhat': [x[1] for x in m['mura_test_100']['yhat']],
                                    'region': m['mura_test_100']['region'],
                                    'study': m['mura_test_100']['study'],
                                    'file': m['mura_test_100']['file']})
    assert len(dat) == 1311
    # print(t['name'])

    # Make a list of the correctly classified radiographs
    for i in range(len(dat)):
        if ((dat['y'][i] == 1.0) and (dat['yhat'][i] > 0.5)) or ((dat['y'][i] == 0.0) and (dat['yhat'][i] < 0.5)):
            correctly_classified[t['name']].append(os.path.basename(dat['file'][i]))

# Localization
bbox_sal_maps = np.load(os.path.join(cfg["heatmap_dir"], "InceptionV3_1_100", "{}_bbox.npy".format(saliency_method[0])))
# image is in correctly classified subset
assert (os.path.basename(sample_image) in (correctly_classified["InceptionV3_1_100"])) == True
# range for ground truth and raw saliency map is 0-1
bbox_saliency_map = bbox_sal_maps[sample_image_index]
plot_heatmap(bbox_saliency_map, "overview_heatmap.png")

precision, recall, thresholds = precision_recall_curve(
     consensus_annotation.flatten(),
     bbox_saliency_map.flatten()
)
print(f"AUPRC: {auc(recall, precision)}")
fig = plt.figure()
plt.plot(recall, precision, color="red")
plt.fill_between(recall, precision, color="red", alpha=0.5)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig(os.path.join(cfg["tmp_dir"], "overview_auprc_curve.png"), dpi=1200)

z = bbox_saliency_map + consensus_annotation
fig = plt.figure()
plt.imshow(z, vmin=0, vmax=2)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig(os.path.join(cfg["tmp_dir"], "overview_overlap.png"), dpi=1200)

# Heatmap consistency
bbox_sal_maps2 = np.load(os.path.join(cfg["heatmap_dir"], "DenseNet169_1_100", "{}_bbox.npy".format(saliency_method[0])))
# range for ground truth and raw saliency map is 0-1
bbox_saliency_map2 = bbox_sal_maps2[sample_image_index]
plot_heatmap(bbox_saliency_map2, "overview_heatmap2.png")

# Heatmap sensitivity
bbox_sal_maps3 = np.load(os.path.join(cfg["heatmap_dir"], "InceptionV3_1_100_Mixed_7c", "{}_bbox.npy".format(saliency_method[0])))
bbox_saliency_map3 = bbox_sal_maps3[cascade_sample_image_index]
plot_heatmap(bbox_saliency_map3, "overview_heatmap3.png")

bbox_sal_maps4 = np.load(os.path.join(cfg["heatmap_dir"], "InceptionV3_1_100_Mixed_7b", "{}_bbox.npy".format(saliency_method[0])))
bbox_saliency_map4 = bbox_sal_maps4[cascade_sample_image_index]
plot_heatmap(bbox_saliency_map4, "overview_heatmap4.png")

bbox_sal_maps5 = np.load(os.path.join(cfg["heatmap_dir"], "InceptionV3_1_100_Mixed_5c", "{}_bbox.npy".format(saliency_method[0])))
bbox_saliency_map5 = bbox_sal_maps5[cascade_sample_image_index]
plot_heatmap(bbox_saliency_map5, "overview_heatmap5.png")
