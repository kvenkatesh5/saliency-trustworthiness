# Imports
import argparse
import os
import json

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--cfg-dir", default='cfg.json', type=str, help='')
parser.add_argument("--use-gpus", default='all', type=str, help='')
args = parser.parse_args()

# Get cfg
with open(args.cfg_dir.replace('~', os.path.expanduser('~')), 'r') as f:
    cfg = json.load(f)
cfg['data_dir'] = cfg['data_dir'].replace('~', os.path.expanduser('~'))

# Load similarity scores
df = pd.read_csv(
    os.path.join(cfg["trustworthiness_dir"], "heatmap_similarity.csv"),
)
assert len(df) == (8 * 6 * 3 * 588 + 3 * 588)

cnn_list = [
    # repeatability set
    # compare best two performing Inception models
    ("InceptionV3_1_100", "InceptionV3_2_100"),
    ("InceptionV3_1_50", "InceptionV3_3_50"),
    ("InceptionV3_1_10", "InceptionV3_2_10"),
    ("InceptionV3_2_1", "InceptionV3_1_1"),
    # reproducibility set
    # compare best Inception to the DenseNet169
    ("InceptionV3_1_100", "DenseNet169_1_100"),
    ("InceptionV3_1_50", "DenseNet169_1_50"),
    ("InceptionV3_1_10", "DenseNet169_1_10"),
    ("InceptionV3_2_1", "DenseNet169_1_1"),
]
methods = [('gcam', 'GCAM'), ('grad', 'GRAD'), ('ig', 'IG'), ('sg', 'SG'), ('sig', 'SIG'), ('xrai', 'XRAI')]
# here, raw mode is not empty string
modes = ["_raw", "_otsu", "_bbox"]
subgroups = ["arthritis", "hardware/fracture", "other"]
alpha = 0.05

# Repeatability tests
"""
These tests compare the two Inception models.
Comparisons made using one-sided t-tests.
"""
repeatability_tests = {
    "CNN Training Subset %": [],
    "Method": [],
    "Mode": [],
    "Mean SSIM": [], "Std SSIM": [],
    "Mean IoU": [], "Std IoU": [],
    "Mean Pixel Error": [], "Std Pixel Error": [],
    "Weak SSIM": [], "Weak IoU": [], "Weak Pixel Error": [], 
    "Weak Repeatability": [],
    "Strong SSIM": [], "Strong IoU": [], "Strong Pixel Error": [], 
    "Strong Repeatability": [],
}

for cnn in cnn_list[:4]:
    for mth in methods:
        for mode in modes:
            subdf = df[((df["Model-1"]==cnn[0]) & (df["Model-2"]==cnn[1]) & (df["Saliency Method"]==mth[1]) & (df["Preprocessing Mode"]==mode))]
            repeatability_tests["CNN Training Subset %"].append(
                cnn[0].replace("InceptionV3_1_", "").replace("InceptionV3_2_", "")
            )
            repeatability_tests["Method"].append(
                mth[1]
            )
            repeatability_tests["Mode"].append(
                mode
            )
            repeatability_tests["Mean SSIM"].append(
                subdf["SSIM"].mean()
            )
            repeatability_tests["Std SSIM"].append(
                subdf["SSIM"].std()
            )
            repeatability_tests["Mean IoU"].append(
                subdf["IoU"][~np.isnan(subdf["IoU"])].mean()
            )
            repeatability_tests["Std IoU"].append(
                subdf["IoU"][~np.isnan(subdf["IoU"])].std()
            )
            repeatability_tests["Mean Pixel Error"].append(
                subdf["Pixel Error"].mean()
            )
            repeatability_tests["Std Pixel Error"].append(
                subdf["Pixel Error"].std()
            )
            # Weak test: compare against mean=0.5
            # this function does a two-tailed test, so we have to manually adjust to greater than (one-sided)
            s_ssim_test = ttest_1samp(subdf["SSIM"], 0.5)
            repeatability_tests["Weak SSIM"].append(
                s_ssim_test.pvalue/2
            )
            s_ssim_pass = ((s_ssim_test.pvalue/2 < alpha) & (s_ssim_test.statistic > 0))
            s_iou_test = ttest_1samp(subdf["IoU"][~np.isnan(subdf["IoU"])], 0.5)
            repeatability_tests["Weak IoU"].append(
                s_iou_test.pvalue/2
            )
            s_iou_pass = ((s_iou_test.pvalue/2 < alpha) & (s_iou_test.statistic > 0))
            s_pxlerr_test = ttest_1samp(subdf["Pixel Error"], 0.5)
            repeatability_tests["Weak Pixel Error"].append(
                s_pxlerr_test.pvalue/2
            )
            s_pxlerr_pass = ((s_pxlerr_test.pvalue/2 < alpha) & (s_pxlerr_test.statistic < 0))
            # final result asks if all the metrics passed the one-sided tests
            weak_repeatability_test = (s_ssim_pass & s_iou_pass\
                                       & s_pxlerr_pass)
            if weak_repeatability_test:
                repeatability_tests["Weak Repeatability"].append("Pass")
            else:
                repeatability_tests["Weak Repeatability"].append("Fail")
            # Strong test: compare against threshold
            # compute threshold
            tdf = df[(df["Model-2"] == "GroundTruth")][["SSIM", "IoU", "Pixel Error", "Image"]]\
                .groupby("Image", as_index=False).mean()
            t_ssim = tdf["SSIM"].mean()
            t_iou = tdf["IoU"][~np.isnan(tdf["IoU"])].mean()
            t_pxlerr = tdf["Pixel Error"].mean()
            # print(t_ssim, t_iou, t_pxlerr, "REPEAT")
            # make comparisons
            # each metric passes if it is NOT significantly less
            s_ssim_test = ttest_1samp(subdf["SSIM"], t_ssim)
            repeatability_tests["Strong SSIM"].append(
                s_ssim_test.pvalue/2
            )
            s_ssim_pass = (not ((s_ssim_test.pvalue/2 < alpha) & (s_ssim_test.statistic < 0)))
            s_iou_test = ttest_1samp(subdf["IoU"][~np.isnan(subdf["IoU"])], t_iou)
            repeatability_tests["Strong IoU"].append(
                s_iou_test.pvalue/2
            )
            s_iou_pass = (not ((s_iou_test.pvalue/2 < alpha) & (s_iou_test.statistic < 0)))
            s_pxlerr_test = ttest_1samp(subdf["Pixel Error"], t_pxlerr)
            repeatability_tests["Strong Pixel Error"].append(
                s_pxlerr_test.pvalue/2
            )
            s_pxlerr_pass = (not ((s_pxlerr_test.pvalue/2 < alpha) & (s_pxlerr_test.statistic > 0)))
            # final result asks if all the metrics passed the one-sided tests
            strong_repeatability_test = (s_ssim_pass & s_iou_pass\
                                         & s_pxlerr_pass)
            if strong_repeatability_test:
                repeatability_tests["Strong Repeatability"].append("Pass")
            else:
                repeatability_tests["Strong Repeatability"].append("Fail")

df_repeatability = pd.DataFrame().from_dict(repeatability_tests)
assert len(df_repeatability) == (4*6*3)
df_repeatability.to_csv(
     os.path.join(cfg["trustworthiness_dir"], "heatmap_repeatability.csv"),
     index=False
)

# # Repeatability stat comparisons (asked for in rev1) 
# """
# These tests compare all the repeatability scores using pairwise Tukey means. Hard-coded for 100% subset.
# """
# cnn = ("InceptionV3_1_100", "InceptionV3_2_100")
# mode = "_bbox"
# repeatability_compdf = df[((df["Model-1"]==cnn[0]) & (df["Model-2"]==cnn[1]) & (df["Preprocessing Mode"]==mode))]
# repeatability_comp_ssim = pairwise_tukeyhsd(
#     endog=repeatability_compdf["SSIM"],
#     groups=repeatability_compdf["Saliency Method"])
# repeatability_comp_iou = pairwise_tukeyhsd(
#     endog=repeatability_compdf[~np.isnan(repeatability_compdf["IoU"])]["IoU"],
#     groups=repeatability_compdf[~np.isnan(repeatability_compdf["IoU"])]["Saliency Method"])
# repeatability_comp_pxlerr = pairwise_tukeyhsd(
#     endog=repeatability_compdf["Pixel Error"],
#     groups=repeatability_compdf["Saliency Method"])

# Repeatability tests per subgroup
"""
These tests compare the two Inception models.
Comparisons made using one-sided t-tests.
"""
subgroup_repeatability_tests = {
    "CNN Training Subset %": [],
    "Method": [],
    "Mode": [],
    "Subgroup": [],
    "Mean SSIM": [], "Std SSIM": [],
    "Mean IoU": [], "Std IoU": [],
    "Mean Pixel Error": [], "Std Pixel Error": [],
    "Weak SSIM": [], "Weak IoU": [], "Weak Pixel Error": [], 
    "Weak Repeatability": [],
    "Strong SSIM": [], "Strong IoU": [], "Strong Pixel Error": [], 
    "Strong Repeatability": [],
}

for cnn in cnn_list[:4]:
    for mth in methods:
        for mode in modes:
            for subgroup in subgroups:
                subdf = df[((df["Model-1"]==cnn[0]) & (df["Model-2"]==cnn[1]) & (df["Saliency Method"]==mth[1]) & (df["Preprocessing Mode"]==mode)\
                            & (df["Subgroup"]==subgroup))]
                subgroup_repeatability_tests["CNN Training Subset %"].append(
                    cnn[0].replace("InceptionV3_1_", "").replace("InceptionV3_2_", "")
                )
                subgroup_repeatability_tests["Method"].append(
                    mth[1]
                )
                subgroup_repeatability_tests["Mode"].append(
                    mode
                )
                subgroup_repeatability_tests["Subgroup"].append(
                    subgroup
                )
                subgroup_repeatability_tests["Mean SSIM"].append(
                    subdf["SSIM"].mean()
                )
                subgroup_repeatability_tests["Std SSIM"].append(
                    subdf["SSIM"].std()
                )
                subgroup_repeatability_tests["Mean IoU"].append(
                    subdf["IoU"][~np.isnan(subdf["IoU"])].mean()
                )
                subgroup_repeatability_tests["Std IoU"].append(
                    subdf["IoU"][~np.isnan(subdf["IoU"])].std()
                )
                subgroup_repeatability_tests["Mean Pixel Error"].append(
                    subdf["Pixel Error"].mean()
                )
                subgroup_repeatability_tests["Std Pixel Error"].append(
                    subdf["Pixel Error"].std()
                )
                # Weak test: compare against mean=0.5
                # this function does a two-tailed test, so we have to manually adjust to greater than (one-sided)
                s_ssim_test = ttest_1samp(subdf["SSIM"], 0.5)
                subgroup_repeatability_tests["Weak SSIM"].append(
                    ((s_ssim_test.pvalue/2 < alpha) & (s_ssim_test.statistic > 0))
                )
                s_iou_test = ttest_1samp(subdf["IoU"][~np.isnan(subdf["IoU"])], 0.5)
                subgroup_repeatability_tests["Weak IoU"].append(
                    ((s_iou_test.pvalue/2 < alpha) & (s_iou_test.statistic > 0))
                )
                s_pxlerr_test = ttest_1samp(subdf["Pixel Error"], 0.5)
                subgroup_repeatability_tests["Weak Pixel Error"].append(
                    ((s_pxlerr_test.pvalue/2 < alpha) & (s_pxlerr_test.statistic < 0))
                )
                # final result asks if all the metrics passed the one-sided tests
                subgroup_weak_repeatbility_test = (subgroup_repeatability_tests["Weak SSIM"][-1] & subgroup_repeatability_tests["Weak IoU"][-1]\
                                                   & subgroup_repeatability_tests["Weak Pixel Error"][-1])
                if subgroup_weak_repeatbility_test:
                    subgroup_repeatability_tests["Weak Repeatability"].append("Pass")
                else:
                    subgroup_repeatability_tests["Weak Repeatability"].append("Fail")
                # Strong test: compare against threshold
                # compute threshold
                tdf = df[(df["Model-2"] == "GroundTruth") & (df["Subgroup"]==subgroup)][["SSIM", "IoU", "Pixel Error", "Image"]]\
                    .groupby("Image", as_index=False).mean()
                t_ssim = tdf["SSIM"].mean()
                t_iou = tdf["IoU"][~np.isnan(tdf["IoU"])].mean()
                t_pxlerr = tdf["Pixel Error"].mean()
                # make comparisons
                # each metric passes if it is NOT significantly less
                s_ssim_test = ttest_1samp(subdf["SSIM"], t_ssim)
                subgroup_repeatability_tests["Strong SSIM"].append(
                    (not ((s_ssim_test.pvalue/2 < alpha) & (s_ssim_test.statistic < 0)))
                )
                s_iou_test = ttest_1samp(subdf["IoU"][~np.isnan(subdf["IoU"])], t_iou)
                subgroup_repeatability_tests["Strong IoU"].append(
                    (not ((s_iou_test.pvalue/2 < alpha) & (s_iou_test.statistic < 0)))
                )
                s_pxlerr_test = ttest_1samp(subdf["Pixel Error"], t_pxlerr)
                subgroup_repeatability_tests["Strong Pixel Error"].append(
                    (not ((s_pxlerr_test.pvalue/2 < alpha) & (s_pxlerr_test.statistic > 0)))
                )
                # final result asks if all the metrics passed the one-sided tests
                subgroup_strong_repeatability_test = (subgroup_repeatability_tests["Strong SSIM"][-1] & subgroup_repeatability_tests["Strong IoU"][-1]\
                                                      & subgroup_repeatability_tests["Strong Pixel Error"][-1])
                if subgroup_strong_repeatability_test:
                    subgroup_repeatability_tests["Strong Repeatability"].append("Pass")
                else:
                    subgroup_repeatability_tests["Strong Repeatability"].append("Fail")


df_subgroup_repeatability = pd.DataFrame().from_dict(subgroup_repeatability_tests)
assert len(df_subgroup_repeatability) == (4*6*3*3)
df_subgroup_repeatability.to_csv(
     os.path.join(cfg["trustworthiness_dir"], "heatmap_repeatability_per_subgroup.csv"),
     index=False
)


# Reproducibility tests
"""
These tests compare the Inception and DenseNet models.
Comparisons made using one-sided t-tests.
"""
reproducibility_tests = {
    "CNN Training Subset %": [],
    "Method": [],
    "Mode": [],
    "Mean SSIM": [], "Std SSIM": [],
    "Mean IoU": [], "Std IoU": [],
    "Mean Pixel Error": [], "Std Pixel Error": [],
    "Weak SSIM": [], "Weak IoU": [], "Weak Pixel Error": [], 
    "Weak Reproducibility": [],
    "Strong SSIM": [], "Strong IoU": [], "Strong Pixel Error": [], 
    "Strong Reproducibility": [],
}

for cnn in cnn_list[4:]:
    for mth in methods:
        for mode in modes:
            subdf = df[((df["Model-1"]==cnn[0]) & (df["Model-2"]==cnn[1]) & (df["Saliency Method"]==mth[1]) & (df["Preprocessing Mode"]==mode))]
            reproducibility_tests["CNN Training Subset %"].append(
                cnn[0].replace("InceptionV3_1_", "").replace("InceptionV3_2_", "")
            )
            reproducibility_tests["Method"].append(
                mth[1]
            )
            reproducibility_tests["Mode"].append(
                mode
            )
            reproducibility_tests["Mean SSIM"].append(
                subdf["SSIM"].mean()
            )
            reproducibility_tests["Std SSIM"].append(
                subdf["SSIM"].std()
            )
            reproducibility_tests["Mean IoU"].append(
                subdf["IoU"][~np.isnan(subdf["IoU"])].mean()
            )
            reproducibility_tests["Std IoU"].append(
                subdf["IoU"][~np.isnan(subdf["IoU"])].std()
            )
            reproducibility_tests["Mean Pixel Error"].append(
                subdf["Pixel Error"].mean()
            )
            reproducibility_tests["Std Pixel Error"].append(
                subdf["Pixel Error"].std()
            )
            # Weak test: compare against mean=0.5
            # this function does a two-tailed test, so we have to manually adjust to greater than (one-sided)
            s_ssim_test = ttest_1samp(subdf["SSIM"], 0.5)
            reproducibility_tests["Weak SSIM"].append(
                s_ssim_test.pvalue/2
            )
            s_ssim_pass = ((s_ssim_test.pvalue/2 < alpha) & (s_ssim_test.statistic > 0))
            s_iou_test = ttest_1samp(subdf["IoU"][~np.isnan(subdf["IoU"])], 0.5)
            reproducibility_tests["Weak IoU"].append(
                s_iou_test.pvalue/2
            )
            s_iou_pass = ((s_iou_test.pvalue/2 < alpha) & (s_iou_test.statistic > 0))
            s_pxlerr_test = ttest_1samp(subdf["Pixel Error"], 0.5)
            reproducibility_tests["Weak Pixel Error"].append(
                s_pxlerr_test.pvalue/2
            )
            s_pxlerr_pass = ((s_pxlerr_test.pvalue/2 < alpha) & (s_pxlerr_test.statistic < 0))
            # final result asks if all the metrics passed the one-sided tests
            weak_reproducibility_test = (s_ssim_pass & s_iou_pass\
                                         & s_pxlerr_pass)
            if weak_reproducibility_test:
                reproducibility_tests["Weak Reproducibility"].append("Pass")
            else:
                reproducibility_tests["Weak Reproducibility"].append("Fail")
            # Strong test: compare against threshold
            # compute threshold
            tdf = df[(df["Model-2"] == "GroundTruth")][["SSIM", "IoU", "Pixel Error", "Image"]]\
                .groupby("Image", as_index=False).mean()
            t_ssim = tdf["SSIM"].mean()
            t_iou = tdf["IoU"][~np.isnan(tdf["IoU"])].mean()
            t_pxlerr = tdf["Pixel Error"].mean()
            # print(t_ssim, t_iou, t_pxlerr, "REPRODUCE")
            # make comparisons
            # each metric passes if it is NOT significantly less
            s_ssim_test = ttest_1samp(subdf["SSIM"], t_ssim)
            reproducibility_tests["Strong SSIM"].append(
                s_ssim_test.pvalue/2
            )
            s_ssim_pass = (not ((s_ssim_test.pvalue/2 < alpha) & (s_ssim_test.statistic < 0)))
            s_iou_test = ttest_1samp(subdf["IoU"][~np.isnan(subdf["IoU"])], t_iou)
            reproducibility_tests["Strong IoU"].append(
                s_iou_test.pvalue/2
            )
            s_iou_pass = (not ((s_iou_test.pvalue/2 < alpha) & (s_iou_test.statistic < 0)))
            s_pxlerr_test = ttest_1samp(subdf["Pixel Error"], t_pxlerr)
            reproducibility_tests["Strong Pixel Error"].append(
                s_pxlerr_test.pvalue/2
            )
            s_pxlerr_pass = (not ((s_pxlerr_test.pvalue/2 < alpha) & (s_pxlerr_test.statistic > 0)))
            # final result asks if all the metrics passed the one-sided tests
            strong_reproducibility_test = (s_ssim_pass & s_iou_pass\
                                           & s_pxlerr_pass)
            if strong_reproducibility_test:
                reproducibility_tests["Strong Reproducibility"].append("Pass")
            else:
                reproducibility_tests["Strong Reproducibility"].append("Fail")             


df_reproducibility = pd.DataFrame().from_dict(reproducibility_tests)
assert len(df_reproducibility) == (4*6*3)
df_reproducibility.to_csv(
     os.path.join(cfg["trustworthiness_dir"], "heatmap_reproducibility.csv"),
     index=False
)

# # Reproducibility stat comparisons (asked for in rev1) 
# """
# These tests compare all the reproducibility scores using pairwise Tukey means. Hard-coded for 100% subset.
# """
# cnn = ("InceptionV3_1_100", "DenseNet169_1_100")
# mode = "_bbox"
# reproducibility_compdf = df[((df["Model-1"]==cnn[0]) & (df["Model-2"]==cnn[1]) & (df["Preprocessing Mode"]==mode))]
# reproducibility_comp_ssim = pairwise_tukeyhsd(
#     endog=reproducibility_compdf["SSIM"],
#     groups=reproducibility_compdf["Saliency Method"])
# reproducibility_comp_iou = pairwise_tukeyhsd(
#     endog=reproducibility_compdf[~np.isnan(reproducibility_compdf["IoU"])]["IoU"],
#     groups=reproducibility_compdf[~np.isnan(reproducibility_compdf["IoU"])]["Saliency Method"])
# reproducibility_comp_pxlerr = pairwise_tukeyhsd(
#     endog=reproducibility_compdf["Pixel Error"],
#     groups=reproducibility_compdf["Saliency Method"])
# print(reproducibility_comp_pxlerr)
# exit()

# Reproducibility tests per subgroup
"""
These tests compare the Inception and DenseNet models.
Comparisons made using one-sided t-tests.
"""
subgroup_reproducibility_tests = {
    "CNN Training Subset %": [],
    "Method": [],
    "Mode": [],
    "Subgroup": [],
    "Mean SSIM": [], "Std SSIM": [],
    "Mean IoU": [], "Std IoU": [],
    "Mean Pixel Error": [], "Std Pixel Error": [],
    "Weak SSIM": [], "Weak IoU": [], "Weak Pixel Error": [], 
    "Weak Reproducibility": [],
    "Strong SSIM": [], "Strong IoU": [], "Strong Pixel Error": [], 
    "Strong Reproducibility": [],
}

for cnn in cnn_list[4:]:
    for mth in methods:
        for mode in modes:
            for subgroup in subgroups:
                subdf = df[((df["Model-1"]==cnn[0]) & (df["Model-2"]==cnn[1]) & (df["Saliency Method"]==mth[1]) & (df["Preprocessing Mode"]==mode)\
                            & (df["Subgroup"]==subgroup))]
                subgroup_reproducibility_tests["CNN Training Subset %"].append(
                    cnn[0].replace("InceptionV3_1_", "").replace("InceptionV3_2_", "")
                )
                subgroup_reproducibility_tests["Method"].append(
                    mth[1]
                )
                subgroup_reproducibility_tests["Mode"].append(
                    mode
                )
                subgroup_reproducibility_tests["Subgroup"].append(
                    subgroup
                )
                subgroup_reproducibility_tests["Mean SSIM"].append(
                    subdf["SSIM"].mean()
                )
                subgroup_reproducibility_tests["Std SSIM"].append(
                    subdf["SSIM"].std()
                )
                subgroup_reproducibility_tests["Mean IoU"].append(
                    subdf["IoU"][~np.isnan(subdf["IoU"])].mean()
                )
                subgroup_reproducibility_tests["Std IoU"].append(
                    subdf["IoU"][~np.isnan(subdf["IoU"])].std()
                )
                subgroup_reproducibility_tests["Mean Pixel Error"].append(
                    subdf["Pixel Error"].mean()
                )
                subgroup_reproducibility_tests["Std Pixel Error"].append(
                    subdf["Pixel Error"].std()
                )
                # Weak test: compare against mean=0.5
                # this function does a two-tailed test, so we have to manually adjust to greater than (one-sided)
                s_ssim_test = ttest_1samp(subdf["SSIM"], 0.5)
                subgroup_reproducibility_tests["Weak SSIM"].append(
                    ((s_ssim_test.pvalue/2 < alpha) & (s_ssim_test.statistic > 0))
                )
                s_iou_test = ttest_1samp(subdf["IoU"][~np.isnan(subdf["IoU"])], 0.5)
                subgroup_reproducibility_tests["Weak IoU"].append(
                    ((s_iou_test.pvalue/2 < alpha) & (s_iou_test.statistic > 0))
                )
                s_pxlerr_test = ttest_1samp(subdf["Pixel Error"], 0.5)
                subgroup_reproducibility_tests["Weak Pixel Error"].append(
                    ((s_pxlerr_test.pvalue/2 < alpha) & (s_pxlerr_test.statistic < 0))
                )
                # final result asks if all the metrics passed the one-sided tests
                subgroup_weak_reproducibility_test = (subgroup_repeatability_tests["Weak SSIM"][-1] & subgroup_repeatability_tests["Weak IoU"][-1]\
                                                      & subgroup_repeatability_tests["Weak Pixel Error"][-1])
                if subgroup_weak_reproducibility_test:
                    subgroup_reproducibility_tests["Weak Reproducibility"].append("Pass")
                else:
                    subgroup_reproducibility_tests["Weak Reproducibility"].append("Fail")
                # Strong test: compare against threshold
                # compute threshold
                tdf = df[(df["Model-2"] == "GroundTruth") & (df["Subgroup"]==subgroup)][["SSIM", "IoU", "Pixel Error", "Image"]]\
                    .groupby("Image", as_index=False).mean()
                t_ssim = tdf["SSIM"].mean()
                t_iou = tdf["IoU"][~np.isnan(tdf["IoU"])].mean()
                t_pxlerr = tdf["Pixel Error"].mean()
                # make comparisons
                # each metric passes if it is NOT significantly less
                s_ssim_test = ttest_1samp(subdf["SSIM"], t_ssim)
                subgroup_reproducibility_tests["Strong SSIM"].append(
                    (not ((s_ssim_test.pvalue/2 < alpha) & (s_ssim_test.statistic < 0)))
                )
                s_iou_test = ttest_1samp(subdf["IoU"][~np.isnan(subdf["IoU"])], t_iou)
                subgroup_reproducibility_tests["Strong IoU"].append(
                    (not ((s_iou_test.pvalue/2 < alpha) & (s_iou_test.statistic < 0)))
                )
                s_pxlerr_test = ttest_1samp(subdf["Pixel Error"], t_pxlerr)
                subgroup_reproducibility_tests["Strong Pixel Error"].append(
                    (not ((s_pxlerr_test.pvalue/2 < alpha) & (s_pxlerr_test.statistic > 0)))
                )
                # final result asks if all the metrics passed the one-sided tests
                subgroup_strong_reproducibility_test = (subgroup_reproducibility_tests["Strong SSIM"][-1] & subgroup_reproducibility_tests["Strong IoU"][-1]\
                                                        & subgroup_reproducibility_tests["Strong Pixel Error"][-1])
                if subgroup_strong_reproducibility_test:
                    subgroup_reproducibility_tests["Strong Reproducibility"].append("Pass")
                else:
                    subgroup_reproducibility_tests["Strong Reproducibility"].append("Fail")

df_subgroup_reproducibility = pd.DataFrame().from_dict(subgroup_reproducibility_tests)
assert len(df_subgroup_reproducibility) == (4*6*3*3)
df_subgroup_reproducibility.to_csv(
     os.path.join(cfg["trustworthiness_dir"], "heatmap_reproducibility_per_subgroup.csv"),
     index=False
)
