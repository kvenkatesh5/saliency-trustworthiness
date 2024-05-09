"""
This file saves the images (w/ their appropriate index and abnormality subgroup) in the saliency test set.
The issue was previously, trustworthiness experiments used the entire 638 images ("test_100_pos.txt") in analysis.
However, radiologist consensus found that 50 of these were ambiguous.
This file makes "saliency_test_set.txt" that stores in rows:
    [Image base name] [Index in test_100_pos.txt] [abnormality subgroup]
Subgroup labels were manually corrected (on the non-empty consensus annotation images) as follows:
    If PHY annotations were nonambiguous, this was assigned.
    Else:
        Corroborating SM and FM annotations were used to make decision.
        If corroborating annotations are unclear:
            Match annotation in other image of study.
The printed table (after adding patient and study counts) is Table 1b in manuscript
"""

# Imports
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

regions = ["SHOULDER", "HUMERUS", "ELBOW", "FOREARM", "WRIST", "HAND", "FINGER"]
labels = pd.read_csv(os.path.join(cfg['data_dir'], "mura", "labels.csv"))

# Categories
# This file (see cfg["subgroup_labels_dir"]) only contains subgroup labels for the non-Inconclusive images
# The old file subanalysis_clean.csv in gaon1 is not up-to-date.
df_subanalysis = pd.read_csv(os.path.join(cfg["subgroup_labels_dir"], "subgroups.csv"))
df_subanalysis = df_subanalysis[["Image", "lumped_subgroup"]]
df_subanalysis.columns = ["filename", "category"]
categories = list(set(df_subanalysis["category"]))
im_to_category = {}
for i, r in df_subanalysis.iterrows():
    im_to_category[r["filename"]] = r["category"]

dat = pd.read_csv(os.path.join(cfg['data_dir'], "mura", "test_100_pos.txt"), header=None)
dat.columns = ["Image"]
dat = pd.merge(dat, labels, how="left", on="Image")
dat["Region"] = dat["Image"].apply(lambda x: x[x.find("_")+1:x.find("_p")])
dat["Study"] = dat["Image"].apply(lambda x: x[:x.find("_image")])
dat["Patient"] = dat["Image"].apply(lambda x: x[x.find("_p")+1:x.find("_s")])
dat["Subgroup"] = ""
dat["Inconclusive"] = ""

results_dir = '/export/gaon1/data/kvenka10/explain-mura/results/'
for i, r in dat.iterrows():
    im_base = r["Image"]
    # Find where CONSENSUS is ambiguous
    gt = np.load(os.path.join(results_dir, "RAD_CONSENSUS/InceptionV3", im_base + ".npy"))
    if np.all(gt==0):
        dat.loc[i, "Inconclusive"] = True
        dat.loc[i, "Subgroup"] = "n/a"
    else:
        dat.loc[i, "Inconclusive"] = False
        dat.loc[i, "Subgroup"] = im_to_category[im_base]

# Filter out the inconclusives
dat2 = dat[dat["Inconclusive"] == False]
# Ensure all subgroups are labeled
assert np.any(dat2["Subgroup"].str.contains("n/a")) == False
table = TableOne(dat2, columns=["Region", "Subgroup"])
print(table.tabulate(tablefmt="fancy_grid"))

with open(os.path.join(cfg['data_dir'], "mura", "saliency_test_set.txt"), "w") as f:
    for i, r in dat2.iterrows():
        # Write the image name, its index in test_100_pos.txt, and its subgroup
        f.write("{}, {}, {}".format(r["Image"], i, r["Subgroup"]))
        f.write("\n")