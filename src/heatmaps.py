"""
Create heatmaps using saliency_test_set.txt
"""

# Imports
import argparse
import json
import os
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
import saliency.core as saliency
from skimage.filters.thresholding import try_all_threshold
import torch
import torch.nn as nn
import torchvision

from .image_utils import preprocess_images, load_image, normalize_image, otsu, bbox, binary_open
from .mura_modules import MURAmodel

warnings.filterwarnings("ignore", category=UserWarning)

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--cfg-dir", default='cfg.json', type=str, help='')
parser.add_argument("--model-name", default='', type=str, help='display name')
parser.add_argument("--cascade-layer", default='n', type=str, help='n means not a casacded model, otherwise provide layer')
parser.add_argument("--use-parallel", default='y', type=str, help='y | n')
parser.add_argument("--use-gpus", default='all', type=str, help='')
args = parser.parse_args()

# Set GPU vis
if args.use_gpus != 'all':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.use_gpus

# Get cfg / setup dirs
with open(args.cfg_dir.replace('~', os.path.expanduser('~')), 'r') as f:
    cfg = json.load(f)
cfg['data_dir'] = cfg['data_dir'].replace('~', os.path.expanduser('~'))
cfg['model_dir'] = cfg['model_dir'].replace('~', os.path.expanduser('~'))
cfg['cascade_model_dir'] = cfg['cascade_model_dir'].replace('~', os.path.expanduser('~'))
cfg['heatmap_dir'] = cfg['heatmap_dir'].replace('~', os.path.expanduser('~'))
cfg['saliency_test_images'] = cfg['saliency_test_images'].replace('~', os.path.expanduser('~'))
image_dir = os.path.join(cfg['data_dir'], "mura", "images")

# Get test images
# If not cascading, use saliency test set
# If cascaded, use cascade test set
# This order (is the same order as the data files) defines the heatmap order in the saved numpy array
is_cascade = (args.cascade_layer != "n")
if is_cascade:
    with open(cfg['cascade_test_images'], 'r') as f:
        image_paths = [x.strip('\n') for x in f.readlines()]
        # Ignore the index + subgroup label in file
        image_paths = [os.path.join(image_dir, x.split(",")[0]) for x in image_paths]
else:
    with open(cfg['saliency_test_images'], 'r') as f:
        image_paths = [x.strip('\n') for x in f.readlines()]
        # Ignore the index + subgroup label + pair index in file
        image_paths = [os.path.join(image_dir, x.split(",")[0]) for x in image_paths]

# Model params
model_name = args.model_name
arch = None
if "InceptionV3" in model_name:
    arch = "InceptionV3"
elif "DenseNet121" in model_name:
    arch = "DenseNet121"
    arch = "InceptionV3"
elif "DenseNet169" in model_name:
    arch = "DenseNet169"
assert ('DenseNet169' == arch) or ('DenseNet121' == arch) or ('InceptionV3' == arch), 'Provided model is neither InceptionV3 or DenseNet121 or DenseNet169'
args.use_parallel = args.use_parallel == 'y'
model_args = {
    'model_type': arch,
    'n_labels': 2,
    'use_parallel': args.use_parallel,
    # for purposes of MURAmodel
    'pretrained': False,
    'frozen': False
}

# Setup
model = MURAmodel(model_args)

# Temporary loading measure, must match train.py arg (from ~/my_documents/explain-mura)
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
if is_cascade:
    model_state = os.path.join(cfg["cascade_model_dir"], cfg["final_models"][model_name])
    model_state = model_state.replace("_model.pt", "_"+args.cascade_layer+".pt")
else:
    model_state = os.path.join(cfg["model_dir"], cfg["final_models"][model_name])
model.load_state_dict(torch.load(model_state, map_location=torch.device(device)))
model = model.to(device)
model.eval()
print('Model loaded!')

# Register hooks
if model_args['model_type'] == 'InceptionV3':
    conv_layer = model.module.Mixed_7c if model_args['use_parallel'] else model.Mixed_7c
else:
    conv_layer = model.module.features if model_args['use_parallel'] else model.features
conv_layer_outputs = {}


def conv_layer_forward(m, i, o):
    # output has shape N x C x W x H
    conv_layer_outputs[saliency.base.CONVOLUTION_LAYER_VALUES] = torch.movedim(o, 1, 3).detach().cpu().numpy()

def conv_layer_backward(m, i, o):
    conv_layer_outputs[saliency.base.CONVOLUTION_OUTPUT_GRADIENTS] = torch.movedim(o[0], 1, 3).detach().cpu().numpy()

conv_layer.register_forward_hook(conv_layer_forward)
conv_layer.register_backward_hook(conv_layer_backward)


# Call model function
def call_model_function(images, call_model_args=None, expected_keys=None):
    images = preprocess_images(images)
    images.requires_grad = True
    images = images.to(device)
    target_class_idx = call_model_args['class_idx_str']
    output = model(images)
    if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
        outputs = output[:, target_class_idx]
        grads = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs))
        grads = torch.movedim(grads[0], 1, 3)
        gradients = grads.detach().cpu().numpy()
        return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
    else:
        one_hot = torch.zeros_like(output)
        one_hot[:, target_class_idx] = 1
        model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)
        return conv_layer_outputs


# sz = (299, 299) if model_args['model_type'] == 'InceptionV3' else (224, 224)
sz = (320,320) # taken from cfg

# Results log
meta_results = {
    'model path': model_state,
    'model name': model_name,
    'image paths': image_paths
}
grad_results = np.empty([len(image_paths), *sz])
sg_results = np.empty([len(image_paths), *sz])
ig_results = np.empty([len(image_paths), *sz])
sig_results = np.empty([len(image_paths), *sz])
xrai_results = np.empty([len(image_paths), *sz])
gcam_results = np.empty([len(image_paths), *sz])

grad_otsu_results = np.empty([len(image_paths), *sz])
sg_otsu_results = np.empty([len(image_paths), *sz])
ig_otsu_results = np.empty([len(image_paths), *sz])
sig_otsu_results = np.empty([len(image_paths), *sz])
xrai_otsu_results = np.empty([len(image_paths), *sz])
gcam_otsu_results = np.empty([len(image_paths), *sz])

grad_bbox_results = np.empty([len(image_paths), *sz])
sg_bbox_results = np.empty([len(image_paths), *sz])
ig_bbox_results = np.empty([len(image_paths), *sz])
sig_bbox_results = np.empty([len(image_paths), *sz])
xrai_bbox_results = np.empty([len(image_paths), *sz])
gcam_bbox_results = np.empty([len(image_paths), *sz])

if is_cascade:
    assert len(image_paths) == 100
else:
    assert len(image_paths) == 588

compute_time_start = time.time()
print('Starting saliency computations.\n')
# Per image
for i, image_path in enumerate(image_paths):
    iteration_time_start = time.time()
    print(
        'Computing saliency maps for {} on image {} ({}/{})\n'.format(model_state, image_path, i + 1, len(image_paths)))
    # Load images and make prediction manually
    im_orig = load_image(image_path, sz)
    im_tensor = preprocess_images([im_orig])
    im_tensor = im_tensor.to(device)
    predictions = model(im_tensor)
    predictions = predictions.detach().cpu().numpy()
    prediction_class = np.argmax(predictions[0])
    call_model_args_dict = {'class_idx_str': prediction_class}

    # Compute masks
    output = '\tComputed {} saliency masks. 3d shape: {} | grayscale shape: {} | any nans: {} | all nans: {}'
    # Gradient saliency methods
    gradient_saliency = saliency.GradientSaliency()
    vanilla_mask_3d = gradient_saliency.GetMask(im_orig, call_model_function, call_model_args_dict)
    vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_mask_3d)
    vanilla_mask_norm = normalize_image(vanilla_mask_3d)
    print(output.format('vanilla gradient', vanilla_mask_3d.shape, vanilla_mask_grayscale.shape,
                        np.any(np.isnan(vanilla_mask_grayscale)), np.all(np.isnan(vanilla_mask_grayscale))))
    smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(im_orig, call_model_function, call_model_args_dict)
    smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)
    smoothgrad_mask_norm = normalize_image(smoothgrad_mask_3d)
    print(output.format('smooth gradient', smoothgrad_mask_3d.shape, smoothgrad_mask_grayscale.shape,
                        np.any(np.isnan(smoothgrad_mask_grayscale)), np.all(np.isnan(smoothgrad_mask_grayscale))))
    # Integrated gradients (path-based) methods
    integrated_gradients = saliency.IntegratedGradients()
    baseline = np.zeros(im_orig.shape)
    vanilla_integrated_gradients_mask_3d = integrated_gradients.GetMask(im_orig, call_model_function,
                                                                        call_model_args_dict, x_steps=25,
                                                                        x_baseline=baseline, batch_size=20)
    vanilla_integrated_gradients_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_integrated_gradients_mask_3d)
    vanilla_integrated_gradients_mask_norm = normalize_image(vanilla_integrated_gradients_mask_3d)
    print(output.format('vanilla integrated gradient', vanilla_integrated_gradients_mask_3d.shape,
                        vanilla_integrated_gradients_mask_grayscale.shape,
                        np.any(np.isnan(vanilla_integrated_gradients_mask_grayscale)),
                        np.all(np.isnan(vanilla_integrated_gradients_mask_grayscale))))
    smoothgrad_integrated_gradients_mask_3d = integrated_gradients.GetSmoothedMask(im_orig, call_model_function,
                                                                                   call_model_args_dict, x_steps=25,
                                                                                   x_baseline=baseline, batch_size=20)
    smoothgrad_integrated_gradients_mask_grayscale = saliency.VisualizeImageGrayscale(
        smoothgrad_integrated_gradients_mask_3d)
    smoothgrad_integrated_gradients_mask_norm = normalize_image(smoothgrad_integrated_gradients_mask_3d)
    print(output.format('smooth integrated gradient', smoothgrad_integrated_gradients_mask_3d.shape,
                        smoothgrad_integrated_gradients_mask_grayscale.shape,
                        np.any(np.isnan(smoothgrad_integrated_gradients_mask_grayscale)),
                        np.all(np.isnan(smoothgrad_integrated_gradients_mask_grayscale))))
    # XRAI method
    xrai_saliency = saliency.XRAI()
    xrai_attributions = xrai_saliency.GetMask(im_orig, call_model_function, call_model_args_dict, batch_size=20)
    xrai_attributions_norm = normalize_image(xrai_attributions)
    print('\tComputed XRAI saliency masks. XRAI attributions shape: {} | XRAI attributions norm shape: {}'.format(
        xrai_attributions.shape,
        xrai_attributions_norm.shape, np.any(np.isnan(xrai_attributions_norm)),
        np.all(np.isnan(xrai_attributions_norm))))
    # GradCAM method
    grad_cam = saliency.GradCam()
    grad_cam_mask_3d = grad_cam.GetMask(im_orig, call_model_function, call_model_args_dict)
    grad_cam_mask_grayscale = saliency.VisualizeImageGrayscale(grad_cam_mask_3d)
    grad_cam_mask_norm = normalize_image(grad_cam_mask_3d)
    print(output.format('grad cam', grad_cam_mask_3d.shape, grad_cam_mask_grayscale.shape,
                        np.any(np.isnan(grad_cam_mask_grayscale)), np.all(np.isnan(grad_cam_mask_grayscale))))

    # Store raw results
    grad_results[i] = vanilla_mask_grayscale
    sg_results[i] = smoothgrad_mask_grayscale
    ig_results[i] = vanilla_integrated_gradients_mask_grayscale
    sig_results[i] = smoothgrad_integrated_gradients_mask_grayscale
    xrai_results[i] = xrai_attributions_norm
    gcam_results[i] = grad_cam_mask_grayscale

    # Set all nans to zero (and all infs to finite values)
    gcam_results[i] = np.nan_to_num(gcam_results[i])
    grad_results[i] = np.nan_to_num(grad_results[i])
    ig_results[i] = np.nan_to_num(ig_results[i])
    sg_results[i] = np.nan_to_num(sg_results[i])
    sig_results[i] = np.nan_to_num(sig_results[i])
    xrai_results[i] = np.nan_to_num(xrai_results[i])
    
    # Store otsu results
    grad_otsu_results[i] = otsu(grad_results[i])
    sg_otsu_results[i] = otsu(sg_results[i])
    ig_otsu_results[i] = otsu(ig_results[i])
    sig_otsu_results[i] = otsu(sig_results[i])
    xrai_otsu_results[i] = otsu(xrai_results[i])
    gcam_otsu_results[i] = otsu(gcam_results[i])

    # Store bbox results (after bbox, do a binary opening to remove noise)
    grad_bbox_results[i] = binary_open(bbox(grad_otsu_results[i]))
    sg_bbox_results[i] = binary_open(bbox(sg_otsu_results[i]))
    ig_bbox_results[i] = binary_open(bbox(ig_otsu_results[i]))
    sig_bbox_results[i] = binary_open(bbox(sig_otsu_results[i]))
    xrai_bbox_results[i] = binary_open(bbox(xrai_otsu_results[i]))
    gcam_bbox_results[i] = binary_open(bbox(gcam_otsu_results[i]))

    # Logging
    print('\tAll masks have been stored!')
    print("\tIteration Time (mins): {:4.4f}\n".format((time.time() - iteration_time_start) / 60))

compute_time_delta = (time.time() - compute_time_start) / 60
print("\nCompute Time (mins): {:4.4f}".format(compute_time_delta))

# Save
if is_cascade:
    maps_dir = os.path.join(cfg["heatmap_dir"], model_name + "_" + args.cascade_layer)
else:
    maps_dir = os.path.join(cfg["heatmap_dir"], model_name)
# If directory exists, alert that to the user!
os.makedirs(maps_dir, exist_ok=False)
with open(os.path.join(maps_dir, 'meta.json'), 'w') as f:
    json.dump(meta_results, f)

# Raw results
np.save(os.path.join(maps_dir, 'grad'), grad_results)
np.save(os.path.join(maps_dir, 'sg'), sg_results)
np.save(os.path.join(maps_dir, 'ig'), ig_results)
np.save(os.path.join(maps_dir, 'sig'), sig_results)
np.save(os.path.join(maps_dir, 'xrai'), xrai_results)
np.save(os.path.join(maps_dir, 'gcam'), gcam_results)
print('Raw results saved.')
# Otsu results
np.save(os.path.join(maps_dir, 'grad_otsu'), grad_otsu_results)
np.save(os.path.join(maps_dir, 'sg_otsu'), sg_otsu_results)
np.save(os.path.join(maps_dir, 'ig_otsu'), ig_otsu_results)
np.save(os.path.join(maps_dir, 'sig_otsu'), sig_otsu_results)
np.save(os.path.join(maps_dir, 'xrai_otsu'), xrai_otsu_results)
np.save(os.path.join(maps_dir, 'gcam_otsu'), gcam_otsu_results)
print('Otsu results saved.')
# Bbox results
np.save(os.path.join(maps_dir, 'grad_bbox'), grad_bbox_results)
np.save(os.path.join(maps_dir, 'sg_bbox'), sg_bbox_results)
np.save(os.path.join(maps_dir, 'ig_bbox'), ig_bbox_results)
np.save(os.path.join(maps_dir, 'sig_bbox'), sig_bbox_results)
np.save(os.path.join(maps_dir, 'xrai_bbox'), xrai_bbox_results)
np.save(os.path.join(maps_dir, 'gcam_bbox'), gcam_bbox_results)
print('Bbox results saved.')
