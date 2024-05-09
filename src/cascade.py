"""
Perform cascading randomization on a (final) DCNN.
"""

# Imports
import argparse
import json
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision

# Custom
from .mura_modules import MURADataset, MURAmodel

warnings.filterwarnings("ignore", category=UserWarning)

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--cfg-dir", default='cfg.json', type=str, help='')
parser.add_argument("--dataset", default='mura', type=str, help='')
parser.add_argument("--model-name", default='', type=str, help='')
parser.add_argument("--use-parallel", default='y', type=str, help='y | n')
parser.add_argument("--num-workers", default=12, type=int, help='')
parser.add_argument("--use-gpus", default='all', type=str, help='')
args = parser.parse_args()

# Set GPU vis
if args.use_gpus != 'all':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.use_gpus

# Get cfg
with open(args.cfg_dir.replace('~', os.path.expanduser('~')), 'r') as f:
    cfg = json.load(f)

# Model params
model_name = args.model_name
model_state = cfg["final_models"][model_name]
model_type = None
if 'InceptionV3' in model_state:
    model_type = 'InceptionV3'
assert ('InceptionV3' == model_type),\
      'Cascading randomization only supports InceptionV3'
image_size = cfg["image_size"]
labels = cfg['labels_mura']

model_args = {
    'model_state': model_state,
    'model_type': model_type,
    'model_name': args.model_name,
    'img_size': image_size,
    'use_parallel': args.use_parallel,
    'num_workers': args.num_workers,
    'dataset': args.dataset,
    'labels_set': "mura",
    'labels': labels,
    'n_labels': len(labels),
    'pretrained': True,
    'frozen': False,
}

# Setup
model = MURAmodel(model_args)

# Temporary loading measure, must match train.py arg
if model_args['use_parallel']:
    model = nn.DataParallel(model)

# Device
device = 'cpu'
ncpus = os.cpu_count()
dev_n = ncpus
if torch.cuda.is_available():
    device = 'cuda'
    dev_n = torch.cuda.device_count()
print('\nDevice: {} #: {} #cpus: {}\n'.format(device, dev_n, ncpus))

# Load model
model.load_state_dict(torch.load(os.path.join(cfg["model_dir"], model_state), map_location=torch.device(device)))
model = model.to(device)
model.eval()

# Inception Layers
layer_randomization_order = [
    model.module.fc,
    model.module.Mixed_7c, model.module.Mixed_7b, model.module.Mixed_7a,
    model.module.Mixed_6e, model.module.Mixed_6d, model.module.Mixed_6c, model.module.Mixed_6b, model.module.Mixed_6a,
    model.module.Mixed_5d, model.module.Mixed_5c, model.module.Mixed_5b,
    model.module.Conv2d_4a_3x3, model.module.Conv2d_3b_1x1, model.module.Conv2d_2b_3x3, model.module.Conv2d_2a_3x3,
    model.module.Conv2d_1a_3x3
]
layer_randomization_order_names = [
    'fc',
    'Mixed_7c', 'Mixed_7b', 'Mixed_7a',
    'Mixed_6e', 'Mixed_6d', 'Mixed_6c', 'Mixed_6b', 'Mixed_6a',
    'Mixed_5d', 'Mixed_5c', 'Mixed_5b',
    'Conv2d_4a_3x3', 'Conv2d_3b_1x1', 'Conv2d_2b_3x3', 'Conv2d_2a_3x3', 'Conv2d_1a_3x3'
]

# Reinitialize function on a layer
def reinit(m):
    # Reset weights for Linear, Conv2d, and BatchNorm2D layers
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
        with torch.no_grad():
            m.reset_parameters()


# Recursively apply reinitialization
def recurse(layer):
    if len(list(layer.children())) == 0:
        layer.apply(reinit)
        return

    for children in layer.children():
        recurse(children)

entries = []
# Cascading randomization (start with fc and work top-->down)
for i, layer_ in enumerate(layer_randomization_order):
    recurse(layer_)
    new_file_path = os.path.join(
        cfg["cascade_model_dir"],
        model_state.replace("model.pt", layer_randomization_order_names[i] + ".pt")
    )
    new_model_name = model_name + "_{}".format(layer_randomization_order_names[i])
    torch.save(model.state_dict(), new_file_path)
    entries.append("{},{}".format(new_file_path, new_model_name))
    print(new_file_path)

# Make ledger for cascading randomization models
model_aliases_path = os.path.join(cfg["cascade_model_dir"], "casc_rand_model_aliases.txt")
if not os.path.exists(model_aliases_path):
    f = open(model_aliases_path, "w")
else:
    f = open(model_aliases_path, "a")
for e in entries:
    f.write(e + "\n")
f.close()
