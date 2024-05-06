# Standard
import argparse
import json
import os
import time
import warnings
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from torch.utils.data import DataLoader
import numpy as np

# Custom
from .mura_modules import MURADataset, MURAmodel

warnings.filterwarnings("ignore", category=UserWarning)

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--cfg-dir", default='cfg.json', type=str, help='')
parser.add_argument("--architecture", default='InceptionV3', type=str, help='InceptionV3 | DenseNet121')
parser.add_argument("--pretrained", default='y', type=str, help='y | n')
parser.add_argument("--frozen", default='n', type=str, help='y | n')
parser.add_argument("--initial-lr", default=1e-2, type=float, help='')
parser.add_argument("--batch-size", default=16, type=int, help='')
parser.add_argument("--max-epochs", default=75, type=int, help='')
parser.add_argument("--optimizer-family", default='SGD', type=str, help='SGD | AdamW')
parser.add_argument("--weight-decay", default=1e-4, type=float, help='')
parser.add_argument("--momentum", default=0.9, type=float, help='')
parser.add_argument("--scheduler-family", default='step', type=str, help='step | drop')
parser.add_argument("--drop-factor", default=0.1, type=float, help='')
parser.add_argument("--plateau-patience", default=3, type=int, help='')
parser.add_argument("--plateau-threshold", default=1e-4, type=float, help='')
parser.add_argument("--break-patience", default=5, type=int, help='')
parser.add_argument("--dataset", default='mura', type=str, help='')
parser.add_argument("--train-file", default='train_100.txt', type=str, help='')
parser.add_argument("--val-file", default='val_100.txt', type=str, help='')
parser.add_argument("--use-parallel", default='y', type=str, help='y | n')
parser.add_argument("--train-transform", default='hflip-rotate', type=str, help='hflip | rotate | hflip-rotate | hvflip-rotate')
parser.add_argument("--num-workers", default=12, type=int, help='')
parser.add_argument("--dropout", default=0, type=float, help='')
parser.add_argument("--image-size", default=0, type=int, help='')
parser.add_argument("--print-batches", default='n', type=str, help='y | n')
parser.add_argument("--scratch-dir", default='./tmp', type=str, help='')
parser.add_argument("--results-dir", default='/export/gaon2/data/kvenka10/explain-mura/model_training', type=str, help='')
parser.add_argument("--results-file", default='', type=str, help='txt file')
parser.add_argument("--hparam-model", default="n", type=str, help='is this an hparam model?')
parser.add_argument("--final-model", default="n", type=str, help='is this a final model for saliency experiments?')
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

# Parse args
args.pretrained = args.pretrained == 'y'
labels = cfg['labels_mura']
args.frozen = args.frozen == 'y'
args.use_parallel = args.use_parallel == 'y'
args.print_batches = args.print_batches == 'y'
args.scratch_dir = args.scratch_dir.replace('~', os.path.expanduser('~'))
args.results_dir = args.results_dir.replace('~', os.path.expanduser('~'))
if args.results_file == '':
    args.results_file = '{}_lr{}_bs{}_opt{}_wd{}_sch_{}_pp{}_bp{}_tr{}_va{}_tf{}_do{}_{}.txt'.format(
        args.architecture, args.initial_lr, args.batch_size, args.optimizer_family,
        args.weight_decay, args.scheduler_family, args.plateau_patience, args.break_patience,
        args.train_file, args.val_file, args.train_transform, args.dropout, int(time.time()))
image_size = args.image_size
if image_size == 0:
    if args.architecture == 'InceptionV3':
        image_size = 299
    elif args.architecture == 'DenseNet121':
        image_size = 224
args.hparam_model = args.hparam_model == 'y'
args.final_model = args.final_model == 'y'

# Model params
model_args = {
    'model_type': args.architecture,
    'pretrained': args.pretrained,
    'labels_set': "mura",
    'labels': labels,
    'n_labels': len(labels),
    'frozen': args.frozen,
    'initial_lr': args.initial_lr,
    'batch_size': args.batch_size,
    'max_epochs': args.max_epochs,
    'optimizer_family': args.optimizer_family,
    'weight_decay': args.weight_decay,
    'momentum': args.momentum,
    'scheduler_family': args.scheduler_family,
    'drop_factor': args.drop_factor,
    'plateau_patience': args.plateau_patience,
    'plateau_threshold': args.plateau_threshold,
    'break_patience': args.break_patience,
    'data_dir': cfg['data_dir'],
    'dataset': args.dataset,
    'train_file': args.train_file,
    'val_file': args.val_file,
    'use_parallel': args.use_parallel,
    'train_transform': args.train_transform,
    'num_workers': args.num_workers,
    'dropout': args.dropout,
    'img_size': image_size,
    'print_batches': args.print_batches,
    'scratch_dir': args.scratch_dir,
    'results_dir': args.results_dir,
    'hparam_model': args.hparam_model,
    'final_model': args.final_model,
    'results_file': args.results_file
}


# Print fxn
def print_results(to_print, file_name):
    with open(file_name, 'a') as f:
        f.write(to_print)


print(model_args)

# Setup
model = MURAmodel(model_args)

# Set dropout
for m in model.modules():
    if isinstance(m, nn.Dropout):
        m.p = model_args['dropout']

# Datasets
dataset_root = os.path.join(model_args['data_dir'], model_args['dataset'])
train_data = MURADataset(images_file=os.path.join(dataset_root, model_args['train_file']),
                              dataset=model_args['dataset'],
                              images_dir=os.path.join(dataset_root, 'images'),
                              image_paths_file=None,
                              labels_file=os.path.join(dataset_root, 'labels.csv'),
                              labels=model_args['labels'],
                              transform=model_args['train_transform'],
                              op='train',
                              img_size=model_args['img_size'])
val_data = MURADataset(images_file=os.path.join(dataset_root, model_args['val_file']),
                            dataset=model_args['dataset'],
                            images_dir=os.path.join(dataset_root, 'images'),
                            image_paths_file=None,
                            labels_file=os.path.join(dataset_root, 'labels.csv'),
                            labels=model_args['labels'],
                            transform='none',
                            op='test',
                            img_size=model_args['img_size'])

# Get device
device = 'cpu'
ncpus = os.cpu_count()
dev_n = ncpus
if torch.cuda.is_available():
    device = 'cuda'
    dev_n = torch.cuda.device_count()
print('\nDevice: {} #: {} #cpus: {}\n'.format(device, dev_n, ncpus))

# Data loaders
trainLoader = DataLoader(train_data, batch_size=model_args['batch_size'],
                         pin_memory=True, shuffle=True,
                         num_workers=model_args['num_workers'])

valLoader = DataLoader(val_data, batch_size=model_args['batch_size'],
                       pin_memory=True, shuffle=True,
                       num_workers=model_args['num_workers'])

# Loss functions
loss_fxn = nn.BCELoss()
soft_loss_fxn = nn.BCELoss()

# Optimizer
optimizer = None
if model_args['optimizer_family'] == 'SGD':
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=model_args['initial_lr'],
        momentum=model_args['momentum'],
        weight_decay=model_args['weight_decay']
    )
elif model_args['optimizer_family'] == 'AdamW':
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_args['initial_lr'],
        weight_decay=model_args['weight_decay']
    )

# Scheduler
scheduler = None
if model_args['scheduler_family'] == 'step':
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=model_args['plateau_patience'],
        gamma=model_args['drop_factor'],
    )
elif model_args['scheduler_family'] == 'drop':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        factor=model_args['drop_factor'],
        patience=model_args['plateau_patience'],
        verbose=False
    )

# Train logs
best_log = {'epoch': -1,
            'loss': {'train': 999999, 'val': 999999},
            'auc': {'train': 0, 'val': 0},
            'kappa': {'train': -2, 'val': -2},
            'points': {'train': {'y': [], 'yhat': []}, 'val': {'y': [], 'yhat': []}},
            'timer': 0
            }
train_log = {'epoch': -1,
             'loss': {'train': [], 'val': []},
             'auc': {'train': [], 'val': []},
             'kappa': {'train': [], 'val': []},
             'timer': []
             }

if model_args['use_parallel']:
    model = nn.DataParallel(model)

# Model to device
model = model.to(device)

# Epoch loop
for epoch in range(model_args['max_epochs']):
    time_start = time.time()
    train_log['epoch'] = epoch

    # Train
    model.train()

    train_loss = 0
    batch_counter = 0
    train_ys = []
    train_yhats = []

    for i, (x, y) in enumerate(tqdm(trainLoader)):
        if model_args['print_batches']:
            print('Epoch {}\t{} batch {}/{}'.format(epoch, 'train', batch_counter, len(trainLoader)))
        batch_counter += 1

        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = None
        if model_args['model_type'] == 'InceptionV3':
            yhat = output.logits
            yhat_aux = output.aux_logits
            # weight of auxiliary classifier is set to 0.4
            loss = loss_fxn(yhat, y) + 0.4 * soft_loss_fxn(yhat_aux, y)
        else:
            yhat = output
            loss = loss_fxn(yhat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_loss += loss.item() / len(trainLoader)
            train_ys.extend(y.to('cpu').numpy().tolist())
            train_yhats.extend(yhat.to('cpu').numpy().tolist())

    # Val
    model.eval()

    val_loss = 0
    batch_counter = 0
    val_ys = []
    val_yhats = []
    val_studies = []
    val_regions = []
    val_files = []

    # For each batch
    for x, y, s, r, f in tqdm(valLoader):
        if model_args['print_batches']:
            print('Epoch {}\t{} batch {}/{}'.format(epoch, 'val', batch_counter, len(valLoader)))
        batch_counter += 1
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)

            yhat = model(x)
            loss = loss_fxn(yhat, y)

            val_loss += loss.item() / len(valLoader)
            val_ys.extend(y.to('cpu').numpy().tolist())
            val_yhats.extend(yhat.to('cpu').numpy().tolist())
            val_studies.extend(s)
            val_regions.extend(r)
            val_files.extend(f)

    # Add to train_log
    epoch_time = (time.time() - time_start) / 60
    train_log['timer'].append(epoch_time)
    train_log['loss']['train'].append(train_loss)
    train_log['loss']['val'].append(val_loss)

    train_log['auc']['train'].append(roc_auc_score(train_ys, train_yhats, average='weighted'))
    train_log['auc']['val'].append(roc_auc_score(val_ys, val_yhats, average='weighted'))
    
    train_log['kappa']['train'].append(cohen_kappa_score(
        np.argmax(train_ys, axis=1),
        np.argmax(train_yhats,axis=1)
    ))
    train_log['kappa']['val'].append(cohen_kappa_score(
        np.argmax(val_ys, axis=1),
        np.argmax(val_yhats,axis=1)
    ))

    # Best
    # Update best
    if train_log['auc']['val'][-1] - best_log['auc']['val'] >= model_args['plateau_threshold']:
        # Print
        print('New best!')

        # Update best arg
        best_log = {'epoch': epoch,
                    'loss': {'train': train_loss, 'val': val_loss},
                    'auc': {'train': train_log['auc']['train'][-1], 'val': train_log['auc']['val'][-1]},
                    'kappa': {'train': train_log['kappa']['train'][-1], 'val': train_log['kappa']['val'][-1]},
                    'points': {'train': {'y': train_ys, 'yhat': train_yhats},
                               'val': {'y': val_ys, 'yhat': val_yhats, 'file': val_files}},
                    'timer': sum(train_log['timer'])
                    }

        # Save
        torch.save(model.state_dict(),
                   os.path.join(model_args['results_dir'], model_args['results_file'][:-4] + '_model.pt'))
        torch.save(best_log, os.path.join(model_args['results_dir'], model_args['results_file'][:-4] + '_stats.pt'))

    # Print
    print(
        'Epoch {}\tTrain loss: {:.4f} Val loss: {:.4f} Train auc: {:.4f} Val auc: {:.4f} Val kappa: {:.4f} Time (min): {:.2f} Total '
        'time: {:.2f}'.format(
            epoch,
            train_log['loss']['train'][-1],
            train_log['loss']['val'][-1],
            train_log['auc']['train'][-1],
            train_log['auc']['val'][-1],
            train_log['kappa']['val'][-1],
            epoch_time,
            sum(train_log['timer'])))

    # Early breaking if not a final model
    if not model_args["final_model"]:
        if epoch - best_log['epoch'] > model_args['break_patience']:
            print('Breaking epoch loop')
            break

    # LR Scheduler step
    if model_args['scheduler_family'] == 'no-scheduler':
        pass
    elif model_args['scheduler_family'] == 'drop':
        scheduler.step(train_log['loss']['val'][-1])
    else:
        scheduler.step()

epoch = best_log['epoch']
train_loss = best_log['loss']['train']
val_loss = best_log['loss']['val']
val_auc = best_log['auc']['val']
val_kappa = best_log['kappa']['val']

results = {
    'File': model_args['results_file'],
    'Architecture': model_args['model_type'],
    'Dataset': model_args['dataset'],
    '% Data': model_args['train_file'][model_args['train_file'].rfind('_') + 1:model_args['train_file'].rfind('.')],
    'Use Parallel': model_args['use_parallel'],
    'Initial LR': model_args['initial_lr'],
    'Optimizer': model_args['optimizer_family'],
    'Scheduler': model_args['scheduler_family'],
    'Scheduler Patience': model_args['plateau_patience'],
    'Break Patience': model_args['break_patience'],
    'Scheduler Drop Factor': model_args['drop_factor'],
    'Batch Size': model_args['batch_size'],
    'Weight Decay': model_args['weight_decay'],
    'Frozen': model_args['frozen'],
    'Transform': model_args['train_transform'],
    'Dropout': model_args['dropout'],
    'Epoch': epoch,
    'Loss Train': train_loss,
    'Loss Val': val_loss,
    'AUC Val': val_auc,
    'Kappa Val': val_kappa,
    'Total Time': best_log['timer']
}

print(','.join([str(x) for x in results.values()]))
print_results(','.join([str(x) for x in results.values()]) + '\n',
              os.path.join(model_args['results_dir'], 'results.csv'))
