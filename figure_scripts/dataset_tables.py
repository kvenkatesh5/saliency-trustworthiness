"""
Table 1 in manuscript. Printing out the number of studies / patients separately.
"""

import os
import json
import pandas as pd
import numpy as np
from tableone import TableOne

# Get cfg
cfg_dir = "cfg.json"
with open(cfg_dir.replace('~', os.path.expanduser('~')), 'r') as f:
    cfg = json.load(f)
cfg['data_dir'] = cfg['data_dir'].replace('~', os.path.expanduser('~'))

# Regions & labels
regions = ["SHOULDER", "HUMERUS", "ELBOW", "FOREARM", "WRIST", "HAND", "FINGER"]
labels = pd.read_csv(os.path.join(cfg['data_dir'], "mura", "labels.csv"))

# Training set
training_set = pd.read_csv(os.path.join(cfg["data_dir"], "mura", "train_100.txt"), header=None)
training_set.columns = ["Image"]
training_set = pd.merge(training_set, labels, how="left", on="Image")
training_set["Region"] = training_set["Image"].apply(lambda x: x[x.find("_")+1:x.find("_p")])
training_set["Study"] = training_set["Image"].apply(lambda x: x[:x.find("_image")])
training_set["Patient"] = training_set["Image"].apply(lambda x: x[x.find("_p")+1:x.find("_s")])
training_table = TableOne(training_set, columns=["Region"])
print(training_table.tabulate(tablefmt="fancy_grid"))
print(f"Number of Studies: {len(np.unique(training_set['Study']))} \
      | Number of Patients: {len(np.unique(training_set['Patient']))}")

# Validation set
validation_set = pd.read_csv(os.path.join(cfg["data_dir"], "mura", "val_100.txt"), header=None)
validation_set.columns = ["Image"]
validation_set = pd.merge(validation_set, labels, how="left", on="Image")
validation_set["Region"] = validation_set["Image"].apply(lambda x: x[x.find("_")+1:x.find("_p")])
validation_set["Study"] = validation_set["Image"].apply(lambda x: x[:x.find("_image")])
validation_set["Patient"] = validation_set["Image"].apply(lambda x: x[x.find("_p")+1:x.find("_s")])
validation_table = TableOne(validation_set, columns=["Region"])
print(validation_table.tabulate(tablefmt="fancy_grid"))
print(f"Number of Studies: {len(np.unique(validation_set['Study']))} \
      | Number of Patients: {len(np.unique(validation_set['Patient']))}")

# Test set
testing_set = pd.read_csv(os.path.join(cfg["data_dir"], "mura", "test_100.txt"), header=None)
testing_set.columns = ["Image"]
testing_set = pd.merge(testing_set, labels, how="left", on="Image")
testing_set["Region"] = testing_set["Image"].apply(lambda x: x[x.find("_")+1:x.find("_p")])
testing_set["Study"] = testing_set["Image"].apply(lambda x: x[:x.find("_image")])
testing_set["Patient"] = testing_set["Image"].apply(lambda x: x[x.find("_p")+1:x.find("_s")])
testing_table = TableOne(testing_set, columns=["Region"])
print(testing_table.tabulate(tablefmt="fancy_grid"))
print(f"Number of Studies: {len(np.unique(testing_set['Study']))} \
      | Number of Patients: {len(np.unique(testing_set['Patient']))}")

# Saliency test set
saliency_set = pd.read_csv(os.path.join(cfg["saliency_test_images"]), header=None)
saliency_set.columns = ["Image", "Index", "Subgroup"]
saliency_set = saliency_set[["Image", "Subgroup"]]
saliency_set = pd.merge(saliency_set, labels, how="left", on="Image")
saliency_set["Region"] = saliency_set["Image"].apply(lambda x: x[x.find("_")+1:x.find("_p")])
saliency_set["Study"] = saliency_set["Image"].apply(lambda x: x[:x.find("_image")])
saliency_set["Patient"] = saliency_set["Image"].apply(lambda x: x[x.find("_p")+1:x.find("_s")])
saliency_table = TableOne(saliency_set, columns=["Region", "Subgroup"])
print(saliency_table.tabulate(tablefmt="fancy_grid"))
print(f"Number of Studies: {len(np.unique(saliency_set['Study']))} \
      | Number of Patients: {len(np.unique(saliency_set['Patient']))}")
