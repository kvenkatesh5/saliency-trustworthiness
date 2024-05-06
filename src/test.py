# Imports
import argparse
import json
import os
import pickle
import warnings

# PyTorch
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Custom
from .mura_modules import MURADataset, MURAmodel

warnings.filterwarnings("ignore", category=UserWarning)

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--cfg-dir", default='cfg.json', type=str, help='')
parser.add_argument("--model-name", default='', type=str, help='')
parser.add_argument("--batch-size", default=16, type=int, help='')
parser.add_argument("--dataset", default='mura', type=str, help='')
parser.add_argument("--test-file", default='test_100.txt', type=str, help='')
parser.add_argument("--use-parallel", default='y', type=str, help='y | n')
parser.add_argument("--num-workers", default=12, type=int, help='')
parser.add_argument("--print-batches", default='n', type=str, help='y | n')
parser.add_argument("--results-dir", default='/export/gaon2/data/kvenka10/explain-mura/results/model_evaluation', type=str, help='')
parser.add_argument("--use-gpus", default='all', type=str, help='')
parser.add_argument("--casc-model", default='n', type=str)
args = parser.parse_args()

# Set GPU vis
if args.use_gpus != 'all':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.use_gpus

# Get cfg
with open(args.cfg_dir.replace('~', os.path.expanduser('~')), 'r') as f:
    cfg = json.load(f)
cfg['data_dir'] = cfg['data_dir'].replace('~', os.path.expanduser('~'))

# Parse args
labels = cfg['labels_mura']
args.use_parallel = args.use_parallel == 'y'
args.print_batches = args.print_batches == 'y'
results_dir = args.results_dir
results_dir = results_dir.replace('~', os.path.expanduser('~'))
# Currently assumes you are working with a final model
casc_model = (args.casc_model != "n")
model_name = args.model_name
if casc_model:
    layer_name = args.casc_model
    model_state = os.path.join(cfg["cascade_model_dir"], cfg["final_models"][model_name])\
        .replace("model.pt", layer_name + ".pt")
else:
    model_state = os.path.join(cfg["model_dir"], cfg["final_models"][model_name])
model_type = None
if 'InceptionV3' in model_state:
    model_type = 'InceptionV3'
elif 'DenseNet121' in model_state:
    model_type = 'DenseNet121'
elif 'DenseNet169' in model_state:
    model_type = 'DenseNet169'
image_size = cfg["image_size"]

# Model params
model_args = {
    'model_state': model_state,
    'model_type': model_type,
    'labels_set': "mura",
    'labels': labels,
    'n_labels': len(labels),
    'batch_size': args.batch_size,
    'dropout': 0, # DEBUG
    'pretrained': 'y',
    'frozen': 'n',
    'data_dir': cfg['data_dir'],
    'dataset': args.dataset,
    'test_file': args.test_file,
    'use_parallel': args.use_parallel,
    'num_workers': args.num_workers,
    'img_size': image_size,
    'print_batches': args.print_batches,
    'results_dir': results_dir
}

# Setup
model = MURAmodel(model_args)

# Set dropout
for m in model.modules():
    if isinstance(m, nn.Dropout):
        m.p = model_args['dropout']

# Must match train.py arg
if model_args['use_parallel']:
    model = nn.DataParallel(model)

# Datasets
dataset_root = os.path.join(model_args['data_dir'], model_args['dataset'])
test_data = MURADataset(images_file=os.path.join(dataset_root, model_args['test_file']),
                             dataset=model_args['dataset'],
                             images_dir=os.path.join(dataset_root, 'images'),
                             image_paths_file=None,
                             labels_file=os.path.join(dataset_root, 'labels.csv'),
                             labels=model_args['labels'],
                             transform='none',
                             op='test',
                             img_size=model_args['img_size'],
                             return_index=True)

# Get device
device = 'cpu'
ncpus = os.cpu_count()
dev_n = ncpus
if torch.cuda.is_available() and args.use_gpus != 'none':
    device = 'cuda'
    dev_n = torch.cuda.device_count()
print('Device: {} #: {} #cpus: {}\n'.format(device, dev_n, ncpus))
model.load_state_dict(torch.load(model_args['model_state'], map_location=torch.device(device)))

# Data loaders
testLoader = DataLoader(test_data, batch_size=model_args['batch_size'],
                        pin_memory=True, shuffle=True,
                        num_workers=model_args['num_workers'])

# Loss functions
loss_fxn = nn.BCELoss()

if model_args['use_parallel']:
    model = nn.DataParallel(model)

# Model to device
model = model.to(device)

# Test
model.eval()

test_loss = 0
batch_counter = 0
test_ys = []
test_yhats = []
test_regions = []
test_studies = []
test_files = []

# For each batch
for x, y, study, region, file in testLoader:
    if model_args['print_batches']:
        print('Batch {}/{}'.format(batch_counter, len(testLoader)))
    batch_counter += 1
    with torch.no_grad():
        x = x.to(device)
        y = y.to(device)

        yhat = model(x)
        loss = loss_fxn(yhat, y)

        test_loss += loss.item() / len(testLoader)
        test_ys.extend(y.to('cpu').numpy().tolist())
        test_yhats.extend(yhat.to('cpu').numpy().tolist())
        test_regions.extend(region)
        test_studies.extend(study)
        test_files.extend(file)

# Save
results_path = os.path.join(model_args['results_dir'],
                            model_args['model_state'][
                            model_args['model_state'].rfind('/') + 1:model_args['model_state'].rfind(
                                '.pt')] + '.pkl')
if os.path.exists(results_path):
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
else:
    results = {}

results['{}_{}'.format(model_args['dataset'],
                       model_args['test_file'].replace('.txt', ''))] = {'y': test_ys,
                                                                        'yhat': test_yhats,
                                                                        'region': test_regions,
                                                                        'study': test_studies,
                                                                        'file': test_files}
print(test_loss)
print(results.keys())
with open(results_path, 'wb') as f:
    pickle.dump(results, f)
