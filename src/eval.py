"""
Evaluate the CNNs used in study.
"""

import os, sys, shutil, json
import pandas as pd
import numpy as np
import time
import argparse
import warnings

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, cohen_kappa_score
import matplotlib.pyplot as plt
import random
import math
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

warnings.filterwarnings("ignore", category=UserWarning)

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

# Parse args
labels = cfg['labels_mura'][1:]
regions = ['shoulder', 'humerus', 'elbow', 'forearm', 'wrist', 'hand', 'finger']

# Models to evaluate
to_analyze = cfg["final_models"]
to_analyze = [{"file": to_analyze[x], "name": x} for x in to_analyze]

# Set seeds
random.seed(5)
np.random.seed(5)

# Bootstrap fxn
def bootstrap(dat, n=1000, regions=regions):
    def getMeanCI(metric, ci=0.95):
        metric = np.array(metric)
        ci_lower = ((1.0 - ci) / 2.0) * 100
        ci_upper = (ci + ((1.0 - ci) / 2.0)) * 100

        mean = []
        lci = []
        uci = []

        if len(metric.shape) == 2:
            for c in range(metric.shape[1]):
                mean.append(np.nanmean(metric[:, c]))
                lci.append(max(0.0, np.percentile(metric[:, c], ci_lower)))
                uci.append(min(1.0, np.percentile(metric[:, c], ci_upper)))
        else:
            mean.append(np.nanmean(metric))
            lci.append(max(0.0, np.percentile(metric, ci_lower)))
            uci.append(min(1.0, np.percentile(metric, ci_upper)))

        return {'mean': mean, 'lci': lci, 'uci': uci}

    auc = []
    auc_weighted = []
    precision = []
    recall = []
    f1_score = []
    accuracy = []
    thresholds_list = []

    micro_precision = []
    micro_recall = []
    micro_f1_score = []
    micro_accuracy = []

    # Including Cohen's kappa score
    kappa_score_list = []

    # Bootstrap loop
    for i in tqdm(range(n)):
        # Resample
        resampled_idxs = random.choices(range(dat.shape[0]), k=dat.shape[0])
        dat_rs = dat.iloc[resampled_idxs]

        aucs = []
        weights = []
        # print(len(dat_rs))
        # print(dat_rs['y'].sum())
        for li, l in enumerate(regions):
            dat_sub = dat_rs[dat_rs['region'] == l]
            if (dat_sub['y'].mean() > 0) and (dat_sub['yhat'].mean() < 1):
                aucs.append(roc_auc_score(dat_sub['y'], dat_sub['yhat']))
                weights.append(dat_sub['y'].sum() / dat_rs['y'].sum())
            else:
                aucs.append(np.NaN)
                weights.append(np.NaN)
        # print([x * dat_rs['y'].sum() for x in weights])
        # exit()
        auc.append(aucs)

        auc_weighted.append(
            np.average([x for x in aucs if not math.isnan(x)], weights=[x for x in weights if not math.isnan(x)]))

        # Other metrics by class
        pr = []
        re = []
        f1 = []
        acc = []
        thresh_list = []
        confusion_matrix_total = np.zeros((2, 2))

        # For each region
        for li, l in enumerate(regions):
            dat_sub = dat_rs[dat_rs['region'] == l]

            if (dat_sub['y'].mean() > 0) and (dat_sub['yhat'].mean() < 1):
                # Get optimal threshold
                fpr, tpr, thresholds = roc_curve(dat_sub['y'], dat_sub['yhat'])
                fnr = 1 - tpr
                op_idx = np.nanargmin(np.absolute(((tpr) - (1-fpr))))
                op_thresh = thresholds[op_idx]
                thresh_list.append(op_thresh)

                # Confusion matrix
                confusion_matrix = np.zeros((2, 2))
                for j in range(dat_sub.shape[0]):
                    pred = 0
                    if dat_sub['yhat'].iloc[j] >= op_thresh:
                        pred = 1
                    confusion_matrix[pred, dat_sub['y'].iloc[j]] += 1

                # Calculate confusion matrix metrics
                pr.append(confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 0]))
                re.append(confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[0, 1]))
                f1.append(2 * pr[-1] * re[-1] / (pr[-1] + re[-1]))
                acc.append((confusion_matrix[0, 0] + confusion_matrix[1, 1]) / confusion_matrix.sum())

                # Add to total confusion matrix
                confusion_matrix_total = np.add(confusion_matrix_total, confusion_matrix)
            else:
                pr.append(np.NaN)
                re.append(np.NaN)
                f1.append(np.NaN)
                acc.append(np.NaN)
                thresh_list.append(np.NaN)

        # By class
        precision.append(pr)
        recall.append(re)
        f1_score.append(f1)
        accuracy.append(acc)
        thresholds_list.append(thresh_list)

        # Micro
        micro_precision.append(
            confusion_matrix_total[1, 1] / (confusion_matrix_total[1, 1] + confusion_matrix_total[1, 0]))
        micro_recall.append(
            confusion_matrix_total[1, 1] / (confusion_matrix_total[1, 1] + confusion_matrix_total[0, 1]))
        micro_f1_score.append(2 * micro_precision[-1] * micro_recall[-1] / (micro_precision[-1] + micro_recall[-1]))
        micro_accuracy.append(
            (confusion_matrix_total[0, 0] + confusion_matrix_total[1, 1]) / confusion_matrix_total.sum())
        
        # Kappa score
        preds = dat_rs['yhat'].apply(lambda x: 1 if x > 0.5 else 0)
        kappa_score_list.append(
            cohen_kappa_score(dat_rs['y'], preds)
        )

    # Store full dist of auc_weighted
    auc_weighted_list = auc_weighted
    
    # Get CIs
    auc = getMeanCI(auc)
    # pd.Series(auc_weighted).to_csv(os.path.join(args.dir_name, '{}_auc_weighted_samples.csv'.format(s)), index=False,
    #                                header=False)
    auc_weighted = getMeanCI(auc_weighted)

    # Macro
    macro_precision = getMeanCI(np.mean(precision))
    macro_recall = getMeanCI(np.mean(recall))
    macro_f1_score = getMeanCI(np.mean(f1_score))
    macro_accuracy = getMeanCI(np.mean(accuracy))

    # By class
    precision = getMeanCI(precision)
    recall = getMeanCI(recall)
    f1_score = getMeanCI(f1_score)
    accuracy = getMeanCI(accuracy)
    thresholds_list = getMeanCI(thresholds_list)

    # Micro
    micro_precision = getMeanCI(micro_precision)
    micro_recall = getMeanCI(micro_recall)
    micro_f1_score = getMeanCI(micro_f1_score)
    micro_accuracy = getMeanCI(micro_accuracy)

    # Kappa
    kappa_score = getMeanCI(kappa_score_list)

    return {
        'auc': auc,
        'auc_weighted': auc_weighted,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1_score': micro_f1_score,
        'micro_accuracy': micro_accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1_score': macro_f1_score,
        'macro_accuracy': macro_accuracy,
        'threshold': thresholds_list,
        'kappa_score': kappa_score,
    }, auc_weighted_list, kappa_score_list

# Get bootstrapped metrics
results = {}
auc_weighted_lists = {}
kappa_score_lists = {}
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
                                    'study': m['mura_test_100']['study']})
    assert len(dat) == 1311
    print(t['name'])
    results[t['name']], auc_weighted_lists[t['name']], kappa_score_lists[t['name']] = bootstrap(dat, n=1000)

# Save full dists
np.savez(os.path.join(cfg["tmp_dir"], 'auc_weighted_lists.npz'), d=auc_weighted_lists)
np.savez(os.path.join(cfg["tmp_dir"], 'kappa_score_lists.npz'), d=kappa_score_lists)

# Get results table
results_table = pd.DataFrame()
for model_name, model_results in results.items():
    for metric in ['auc_weighted',
                   'micro_precision',
                   'micro_recall',
                   'micro_f1_score',
                   'micro_accuracy',
                   'macro_precision',
                   'macro_recall',
                   'macro_f1_score',
                   'macro_accuracy',
                   'kappa_score']:
        row = pd.DataFrame.from_dict({
            'Model': [model_name],
            'Data set': 'mura_test_100',
            'Finding': ['Overall'],
            'Metric': [metric],
            'Mean': model_results[metric]['mean'][0],
            'LCI': model_results[metric]['lci'][0],
            'UCI': model_results[metric]['uci'][0]})
        results_table = pd.DataFrame._append(results_table, row, ignore_index=True)

    for metric in ['auc', 'precision', 'recall', 'f1_score', 'accuracy', 'threshold']:
        row = pd.DataFrame.from_dict({
            'Model': [model_name] * len(regions),
            'Data set': ['mura_test_100'] * len(regions),
            'Finding': regions,
            'Metric': [metric] * len(regions),
            'Mean': model_results[metric]['mean'],
            'LCI': model_results[metric]['lci'],
            'UCI': model_results[metric]['uci']
        })
        results_table = pd.DataFrame._append(results_table, row, ignore_index=True)

results_table.to_csv(os.path.join(cfg["model_evaluation_dir"], 'results_table.csv'), index=False)
# results_table.to_csv(os.path.join(cfg["tmp_dir"], 'results_table.csv'), index=False)
