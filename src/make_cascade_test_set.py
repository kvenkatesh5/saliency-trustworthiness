"""
This file saves the images in the cascading randomization test set.
This is a subset of the saliency test set that contains 100 images.
These images are paired into 50 pairs. The pairing index (0-49) determines that.
This file makes "cascade_test_set.txt" that stores in rows:
    [Image base name] [Index in saliency_test_set.txt] [abnormality subgroup] [Pairing index]
The printed table (after adding patient and study counts) is in suppl materials
"""

# Imports
import os
import random
import json
import pandas as pd
import numpy as np
from tableone import TableOne

# Get cfg
cfg_dir = "cfg.json"
with open(cfg_dir.replace('~', os.path.expanduser('~')), 'r') as f:
    cfg = json.load(f)
cfg['data_dir'] = cfg['data_dir'].replace('~', os.path.expanduser('~'))

regions = ["SHOULDER", "HUMERUS", "ELBOW", "FOREARM", "WRIST", "HAND", "FINGER"]
labels = pd.read_csv(os.path.join(cfg['data_dir'], "mura", "labels.csv"))

dat = pd.read_csv(os.path.join(cfg['data_dir'], "mura", "saliency_test_set.txt"), header=None)
dat.columns = ["Image", "Original Index", "Subgroup"]
dat = pd.merge(dat, labels, how="left", on="Image")
dat["Region"] = dat["Image"].apply(lambda x: x[x.find("_")+1:x.find("_p")])
dat["Study"] = dat["Image"].apply(lambda x: x[:x.find("_image")])
dat["Patient"] = dat["Image"].apply(lambda x: x[x.find("_p")+1:x.find("_s")])
assert len(dat) == 588

random.seed(5)
subset_rows = random.sample(range(len(dat)), k=100)
# the order of the pairing indices will be correlated to the order of subset rows
pairing_indices = [x for x in range(50)] * 2
random.shuffle(pairing_indices)
assert len(pairing_indices) == 100
# select the subset (in its order) from the saliency test set
dat2 = dat.iloc[subset_rows, :]
assert len(dat2) == 100

table = TableOne(dat2, columns=["Region", "Subgroup"])
print(table.tabulate(tablefmt="fancy_grid"))

with open(os.path.join(cfg['data_dir'], "mura", "cascade_test_set.txt"), "w") as f:
    for i, (j, r) in enumerate(dat2.iterrows()):
        assert subset_rows[i] == j
        # Write the image name, its index in saliency_test_set.txt, its subgroup, and its pairing index
        f.write("{}, {}, {}, {}".format(r["Image"], subset_rows[i], r["Subgroup"], pairing_indices[i]))
        f.write("\n")
