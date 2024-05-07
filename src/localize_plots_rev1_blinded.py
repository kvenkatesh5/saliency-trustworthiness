# Imports
import argparse
import os
import json

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--cfg-dir", default='cfg.json', type=str, help='')
parser.add_argument("--use-gpus", default='all', type=str, help='')
args = parser.parse_args()

# Get cfg
with open(args.cfg_dir.replace('~', os.path.expanduser('~')), 'r') as f:
    cfg = json.load(f)
cfg['data_dir'] = cfg['data_dir'].replace('~', os.path.expanduser('~'))

# Load localization scores
df_master = pd.read_csv(
    os.path.join(cfg["trustworthiness_dir"], "heatmap_localizations.csv"),
)
assert len(df_master) == (4 * 6 * 3 * 588 + 3 * 588 + 3 * 588)

# InceptionV3 models
methods = [('gcam', 'GCAM'), ('grad', 'GRAD'), ('ig', 'IG'), ('sg', 'SG'), ('sig', 'SIG'), ('xrai', 'XRAI')]
# here, raw mode is not empty string
baselines = ["BASELINE", "BASELINE_OTSU", "BASELINE_SQUARE"]
subgroups = ["arthritis", "hardware/fracture", "other"]

# Best Performing InceptionV3 models
cnn_list = ["InceptionV3_1_100", "InceptionV3_1_50", "InceptionV3_1_10", "InceptionV3_2_1"]
# Load y and yhat for cnns
to_analyze = [{"file": cfg["final_models"][x], "name": x} for x in cnn_list]
correctly_classified = {x: [] for x in cnn_list}
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
    print(t['name'])

    # Make a list of the correctly classified radiographs
    for i in range(len(dat)):
        if ((dat['y'][i] == 1.0) and (dat['yhat'][i] > 0.5)) or ((dat['y'][i] == 0.0) and (dat['yhat'][i] < 0.5)):
            correctly_classified[t['name']].append(os.path.basename(dat['file'][i]))

"""All following plots use InceptionV3_1_100"""
# Remove from consideration images where the AI model was incorrect
df = df_master.loc[df_master["Image"].isin(correctly_classified["InceptionV3_1_100"])]

# Color palette
palette = ["b", "orange", "g", "r", "purple", "brown", "pink", "gray"]
"""
Plot of localization scores on InceptionV3_1_100 with bounding box preprocessing.
"""
s_df = df[((df["Model"]=="InceptionV3_1_100") & (df["Preprocessing Mode"]=="_bbox"))]
b_df = df[(df["Model"]==baselines[2])].copy()
# for plotting purposes
b_df["Saliency Method"] = "Edge Detector"
r_df = df[(df["Model"]=="rad1") | (df["Model"]=="rad2") | (df["Model"]=="rad3")]\
    [["Image", "AUPRC"]].groupby("Image", as_index=False).mean()
# for plotting purposes
r_df["Saliency Method"] = "Inter-Radiologist"
plot_df = pd.concat([s_df, b_df, r_df])
# assertion fails because we removed incorrectly pred. images
# assert len(plot_df) == (8 * 588)
plt.figure()
ax = sns.boxplot(data=plot_df, x="Saliency Method", y="AUPRC", palette=palette, showfliers=False)
# beautify
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.xaxis.label.set_visible(False)
plt.tight_layout()
# plt.savefig(os.path.join(cfg["tmp_dir"], "plot1_rev1.png"), dpi=1200)
plt.savefig(os.path.join(cfg["trustworthiness_dir"], "localization_plot1_rev1.png"), dpi=1200)

"""
Plot of per-subgroup localization scores on InceptionV3_1_100 with bounding box preprocessing.
"""
s_df = df[((df["Model"]=="InceptionV3_1_100") & (df["Preprocessing Mode"]=="_bbox"))]
b_df = df[(df["Model"]==baselines[2])].copy()
# for plotting purposes
b_df["Saliency Method"] = "Edge Detector"
r_df = df[(df["Model"]=="rad1") | (df["Model"]=="rad2") | (df["Model"]=="rad3")].copy()
r_df.drop(columns=["Model"], inplace=True)
r_df = r_df[["Image", "Subgroup", "AUPRC"]].groupby("Image", as_index=False).agg({
    "Image": "first",
    "Subgroup": "first",
    "AUPRC": "mean"
})
# for plotting purposes
r_df["Saliency Method"] = "Inter-Radiologist"
plot_df = pd.concat([s_df, b_df, r_df])
# assertion fails because we removed incorrectly pred. images
# assert len(plot_df) == (8 * 588)
plt.figure()
ax = sns.boxplot(data=plot_df, x="Saliency Method", y="AUPRC", hue="Subgroup", showfliers=False, palette=palette)
# beautify
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
# ax.get_legend().remove()
ax.xaxis.label.set_visible(False)
plt.tight_layout()
# plt.savefig(os.path.join(cfg["tmp_dir"], "plot2s_rev1.png"))
plt.savefig(os.path.join(cfg["trustworthiness_dir"], "localization_plot2s_rev1.png"), dpi=1200)

"""
Plot of intermode comparisons.
"""
subdf = df[((df["Model"]=="InceptionV3_1_100"))]\
    [["Saliency Method", "Preprocessing Mode", "Image", "AUPRC"]].copy()
fancy_name = {
    "_raw": "Raw",
    "_otsu": "Otsu",
    "_bbox": "Boxed"
}
subdf["Preprocessing Mode"] = subdf["Preprocessing Mode"].apply(lambda x: fancy_name[x])
# plt.figure(figsize=(9,12))
plt.figure()
ax = sns.boxplot(data=subdf, x="Saliency Method", y="AUPRC", hue="Preprocessing Mode", showfliers=False, palette=palette)
# beautify
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.get_legend().remove()
ax.xaxis.label.set_visible(False)
plt.tight_layout()
# plt.savefig(os.path.join(cfg["tmp_dir"], "plot3supp_rev1.png"))
plt.savefig(os.path.join(cfg["trustworthiness_dir"], "localization_plot3_rev1.png"), dpi=1200)
