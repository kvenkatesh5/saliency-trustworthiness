# Imports
import argparse
import os
import json

import pickle
import numpy as np
import pandas as pd
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

# Load localization scores
# renamed because now df refers to dataframe without incorrectly classified images
df_master = pd.read_csv(
    os.path.join(cfg["trustworthiness_dir"], "heatmap_localizations.csv"),
)
assert len(df_master) == (4 * 6 * 3 * 588 + 3 * 588 + 3 * 588)

# Best Performing InceptionV3 models
cnn_list = ["InceptionV3_1_100", "InceptionV3_1_50", "InceptionV3_1_10", "InceptionV3_2_1"]
methods = [('gcam', 'GCAM'), ('grad', 'GRAD'), ('ig', 'IG'), ('sg', 'SG'), ('sig', 'SIG'), ('xrai', 'XRAI')]
# here, raw mode is not empty string
modes = ["_raw", "_otsu", "_bbox"]
baselines = ["BASELINE", "BASELINE_OTSU", "BASELINE_SQUARE"]
subgroups = ["arthritis", "hardware/fracture", "other"]

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
    # print(t['name'])

    # Make a list of the correctly classified radiographs
    for i in range(len(dat)):
        if ((dat['y'][i] == 1.0) and (dat['yhat'][i] > 0.5)) or ((dat['y'][i] == 0.0) and (dat['yhat'][i] < 0.5)):
            correctly_classified[t['name']].append(os.path.basename(dat['file'][i]))

# Intermode tests (i.e. which preprocessing mode is best?)
"""
These tests compare the three different preprocessing modes.
Comparisons made using two-sided Tukey tests.
The order of the comp tests is hardcoded to match TukeyHSD ordering
"""
intermode_tests = {"Model": [], "Saliency Method": [], "Raw Mean AUPRC": [], "Raw Std AUPRC": [],
                   "Otsu Mean AUPRC": [], "Otsu Std AUPRC": [], "Bbox Mean AUPRC": [], "Bbox Std AUPRC": [],
                   "Bbox vs. Otsu": [], "Bbox vs. Raw": [], "Otsu vs. Raw": []}

for cnn in cnn_list:
    for mth in methods:
        # Remove from consideration images where the AI model was incorrect
        df = df_master.loc[df_master["Image"].isin(correctly_classified[cnn])]
        
        intermode_tests["Model"].append(cnn)
        intermode_tests["Saliency Method"].append(mth[1])
        subdf = df[(df["Model"]==cnn) & (df["Saliency Method"]==mth[1])]
        intermode_tests["Raw Mean AUPRC"].append(subdf[subdf["Preprocessing Mode"] == "_raw"]["AUPRC"].mean())
        intermode_tests["Raw Std AUPRC"].append(subdf[subdf["Preprocessing Mode"] == "_raw"]["AUPRC"].std())
        intermode_tests["Otsu Mean AUPRC"].append(subdf[subdf["Preprocessing Mode"] == "_otsu"]["AUPRC"].mean())
        intermode_tests["Otsu Std AUPRC"].append(subdf[subdf["Preprocessing Mode"] == "_otsu"]["AUPRC"].std())
        intermode_tests["Bbox Mean AUPRC"].append(subdf[subdf["Preprocessing Mode"] == "_bbox"]["AUPRC"].mean())
        intermode_tests["Bbox Std AUPRC"].append(subdf[subdf["Preprocessing Mode"] == "_bbox"]["AUPRC"].std())
        test = pairwise_tukeyhsd(endog=subdf["AUPRC"], groups=subdf["Preprocessing Mode"])
        # Manually access the parts of the table (print summary to see full table)
        # Assert correct order for pairwise tests
        assert str(test._results_table[1][0]) == "_bbox"
        assert str(test._results_table[2][0]) == "_bbox"
        assert str(test._results_table[3][0]) == "_otsu"
        assert str(test._results_table[1][1]) == "_otsu"
        assert str(test._results_table[2][1]) == "_raw"
        assert str(test._results_table[3][1]) == "_raw"
        # Store pvalues based on asserted order
        intermode_tests["Bbox vs. Otsu"].append(test.pvalues[0])
        intermode_tests["Bbox vs. Raw"].append(test.pvalues[1])
        intermode_tests["Otsu vs. Raw"].append(test.pvalues[2])

df_intermode = pd.DataFrame().from_dict(intermode_tests)
assert len(df_intermode) == (4*6)
df_intermode.to_csv(
     os.path.join(cfg["trustworthiness_dir"], "heatmap_localizations_intermode_tests_rev1.csv"),
     index=False
)

# Stat comparisons (asked for in rev1)
"""
Pairwise comparisons between saliency methods (including edge detector and inter-rad baseline) using two-sided Tukey tests.
PRINT ONLY (comment out if unwanted) and HARDCODED
"""
# alpha = 0.05

# cnn = "InceptionV3_1_100"
# mode = "_bbox"
# j = 2
# # Remove from consideration images where the AI model was incorrect
# df = df_master.loc[df_master["Image"].isin(correctly_classified[cnn])]
# # We sift out either 1) the appropriate heatmaps or 2) mode matched baseline or 3) interrad df
# compdf = df[
#     ((df["Model"]==cnn) & (df["Preprocessing Mode"]==mode))
# ]
# interraddf = df[
#     (df["Model"]=="rad1") | (df["Model"]=="rad2") | (df["Model"]=="rad3")
# ]
# interraddf = interraddf[["Image", "AUPRC"]]
# avgraddf = interraddf.groupby("Image", as_index=False).mean()
# avgraddf["Model"] = "AvgRadiologist"
# avgraddf["Saliency Method"] = "AvgRadiologist"
# print(avgraddf["AUPRC"].mean())
# baselinedf = df[df["Model"] == baselines[j]]
# baselinedf["Saliency Method"] = "Baseline_SQUARE"
# print(baselinedf["AUPRC"].mean())
# allcompdf = pd.concat([
#     compdf[["Model", "Image", "AUPRC", "Saliency Method"]], 
#     avgraddf, baselinedf])
# comp_test = pairwise_tukeyhsd(
#     endog=allcompdf["AUPRC"],
#     groups=allcompdf["Saliency Method"])
# print(cnn, mode)
# print(comp_test)
# exit("printed the comp test")


# Weak/strong localization tests
"""
The weak test compares the heatmaps to the respective mode's baselines.
The strong test compares the heatmaps to inter-radiologist agreement.
Comparisons made using two-sided Tukey tests.
"""

criteria_tests = {"Model": [], "Saliency Method": [], "Preprocessing Mode": [],
                  "Mean AUPRC": [], "Std AUPRC": [], "Mode-Matched Baseline Mean AUPRC": [], 
                  "Mode-Matched Baseline Std AUPRC": [], "AvgRad Mean AUPRC": [], "AvgRad Std AUPRC": [],
                  "Weak Localization Pvalue": [], "Strong Localization Pvalue": [],
                  "Weak Localization": [], "Strong Localization": []}
alpha = 0.05

for cnn in cnn_list:
    for mth in methods:
        for j, mode in enumerate(modes):
            # Remove from consideration images where the AI model was incorrect
            df = df_master.loc[df_master["Image"].isin(correctly_classified[cnn])]

            criteria_tests["Model"].append(cnn)
            criteria_tests["Saliency Method"].append(mth[1])
            criteria_tests["Preprocessing Mode"].append(mode)
            criteria_tests["Mean AUPRC"].append(
                df[((df["Model"]==cnn) & (df["Saliency Method"]==mth[1]) & (df["Preprocessing Mode"]==mode))]["AUPRC"].mean()
            )
            criteria_tests["Std AUPRC"].append(
                df[((df["Model"]==cnn) & (df["Saliency Method"]==mth[1]) & (df["Preprocessing Mode"]==mode))]["AUPRC"].std()
            )
            criteria_tests["Mode-Matched Baseline Mean AUPRC"].append(
                df[df["Model"]==baselines[j]]["AUPRC"].mean()
            )
            criteria_tests["Mode-Matched Baseline Std AUPRC"].append(
                df[df["Model"]==baselines[j]]["AUPRC"].std()
            )
            # Weak version
            # We sift out either 1) the appropriate heatmaps or 2) mode matched baseline
            # The groups are in the model header
            weakdf = df[
                ((df["Model"]==cnn) & (df["Saliency Method"]==mth[1]) & (df["Preprocessing Mode"]==mode))\
                    | (df["Model"]==baselines[j])
            ]
            weak_test = pairwise_tukeyhsd(
                endog=weakdf["AUPRC"],
                groups=weakdf["Model"])
            criteria_tests["Weak Localization Pvalue"].append(weak_test.pvalues[0])
            # Weak comparison: Mean AUPRC > Baseline AUPRC with significance
            if criteria_tests["Weak Localization Pvalue"][-1] < alpha and (criteria_tests["Mean AUPRC"][-1] - criteria_tests["Mode-Matched Baseline Mean AUPRC"][-1] > 0):
                criteria_tests["Weak Localization"] = "Pass"
            else:
                criteria_tests["Weak Localization"] = "Fail"
            # Strong version
            # First we average the other radiologists AUPRCs per image
            interraddf = df[
                (df["Model"]=="rad1") | (df["Model"]=="rad2") | (df["Model"]=="rad3")
            ]
            interraddf = interraddf[["Image", "AUPRC"]]
            avgraddf = interraddf.groupby("Image", as_index=False).mean()
            avgraddf["Model"] = "AvgRadiologist"
            criteria_tests["AvgRad Mean AUPRC"].append(avgraddf["AUPRC"].mean())
            criteria_tests["AvgRad Std AUPRC"].append(avgraddf["AUPRC"].std())
            strongdf = pd.concat([
                df[((df["Model"]==cnn) & (df["Saliency Method"]==mth[1]) & (df["Preprocessing Mode"]==mode))][["Model", "Image", "AUPRC"]], 
                avgraddf])
            strong_test = pairwise_tukeyhsd(
                endog=strongdf["AUPRC"],
                groups=strongdf["Model"])
            criteria_tests["Strong Localization Pvalue"].append(strong_test.pvalues[0])
            # Strong comparison: Fail if Mean AUPRC < AvgRad AUPRC with significance
            if criteria_tests["Strong Localization Pvalue"][-1] < alpha and (criteria_tests["AvgRad Mean AUPRC"][-1] - criteria_tests["Mean AUPRC"][-1] > 0):
                criteria_tests["Strong Localization"] = "Fail"
            else:
                criteria_tests["Strong Localization"] = "Pass"

df_criteria = pd.DataFrame().from_dict(criteria_tests)
assert len(df_criteria) == (4*6*3)
df_criteria.to_csv(
     os.path.join(cfg["trustworthiness_dir"], "heatmap_localizations_criteria_tests_rev1.csv"),
     index=False
)

# Sample efficiency tests
"""
These tests compare the four different training subset sizes.
Comparisons made using two-sided Tukey tests.
The order of the comp tests is hardcoded to match TukeyHSD ordering
"""
# Ignore for now due to removing incorrectly predicted images
"""
interdata_tests = {"Saliency Method": [], "Preprocessing Mode": [], 
                   "100 %": [], "50 %": [], "10 %": [], "1 %": []}
# hard coded order to match tukey groups order
subset_sizes = [1, 10, 100, 50]
for i in range(len(subset_sizes)):
    for j in range(i+1, len(subset_sizes)):
        s1 = subset_sizes[i]
        s2 = subset_sizes[j]
        interdata_tests["{} % vs. {} %".format(s1,s2)] = []

for mth in methods:
    for j, mode in enumerate(modes):
        interdata_tests["Saliency Method"].append(mth[1])
        interdata_tests["Preprocessing Mode"].append(mode)
        subdf = df[(df["Saliency Method"]==mth[1]) & (df["Preprocessing Mode"]==mode) & (df["Model"].str.contains("InceptionV3"))].copy()
        interdata_tests["100 %"].append(
            subdf[subdf["Model"] == "InceptionV3_1_100"]["AUPRC"].mean()
        )
        interdata_tests["50 %"].append(
            subdf[subdf["Model"] == "InceptionV3_1_50"]["AUPRC"].mean()
        )
        interdata_tests["10 %"].append(
            subdf[subdf["Model"] == "InceptionV3_1_10"]["AUPRC"].mean()
        )
        interdata_tests["1 %"].append(
            subdf[subdf["Model"] == "InceptionV3_2_1"]["AUPRC"].mean()
        )
        # Adjust key names
        subdf.loc[:,"Training Subset %"] = subdf.loc[:,"Model"].map(lambda x: x.replace("InceptionV3_1_", "").replace("InceptionV3_2_", ""))
        test = pairwise_tukeyhsd(endog=subdf["AUPRC"], groups=subdf["Training Subset %"])
        # Double for-loop on groupsunique gives order of table
        c = 0
        for i in range(len(test.groupsunique)):
            for j in range(i+1, len(test.groupsunique)):
                interdata_tests["{} % vs. {} %".format(test.groupsunique[i], test.groupsunique[j])].append(test.pvalues[c])
                c += 1

df_interdata = pd.DataFrame().from_dict(interdata_tests)
assert len(df_interdata) == (6*3)
df_interdata.to_csv(
     os.path.join(cfg["trustworthiness_dir"], "heatmap_localizations_interdata_tests_rev1.csv"),
     index=False
)
"""

"""
Per Subgroup Experiments
"""

# Hidden stratification tests
"""
These tests compare the AUPRCs between subgroups.
Comparisons made using two-sided Tukey tests.
"""
subgroup_tests = {"Model": [], "Saliency Method": [], "Preprocessing Mode": []
                   }
for i in range(len(subgroups)):
    subgroup_tests["Mean {} AUPRC".format(subgroups[i].capitalize())] = []
    subgroup_tests["Std {} AUPRC".format(subgroups[i].capitalize())] = []

for i in range(len(subgroups)):
    for j in range(i+1, len(subgroups)):
        subgroup_tests["{} vs. {}".format(subgroups[i].capitalize(), subgroups[j].capitalize())] = []
    
for cnn in cnn_list:
    for mth in methods:
        for j, mode in enumerate(modes):
            # Remove from consideration images where the AI model was incorrect
            df = df_master.loc[df_master["Image"].isin(correctly_classified[cnn])]
            
            subgroup_tests["Model"].append(cnn)
            subgroup_tests["Saliency Method"].append(mth[1])
            subgroup_tests["Preprocessing Mode"].append(mode)
            for subgroup in subgroups:
                subgroup_tests["Mean {} AUPRC".format(subgroup.capitalize())].append(
                    df[(df["Model"]==cnn) & (df["Saliency Method"]==mth[1]) & \
                       (df["Preprocessing Mode"]==mode) & (df["Subgroup"]==subgroup)]["AUPRC"].mean()
                )
                subgroup_tests["Std {} AUPRC".format(subgroup.capitalize())].append(
                    df[(df["Model"]==cnn) & (df["Saliency Method"]==mth[1]) & \
                       (df["Preprocessing Mode"]==mode) & (df["Subgroup"]==subgroup)]["AUPRC"].std()
                )
            subdf = df[(df["Model"]==cnn) & (df["Saliency Method"]==mth[1]) & \
                       (df["Preprocessing Mode"]==mode)]
            test = pairwise_tukeyhsd(
                endog=subdf["AUPRC"], groups=subdf["Subgroup"])
            # subgroups order matches TukeyHSD order (lexicographic)
            assert list(subgroups) == list(test.groupsunique)
            c = 0
            for i in range(len(subgroups)):
                for j in range(i+1, len(subgroups)):
                    subgroup_tests["{} vs. {}".format(subgroups[i].capitalize(), subgroups[j].capitalize())].append(
                        test.pvalues[c]
                    )
                    c += 1

df_subgroups = pd.DataFrame().from_dict(subgroup_tests)
assert len(df_subgroups) == (4*6*3)
df_subgroups.to_csv(
     os.path.join(cfg["trustworthiness_dir"], "heatmap_localizations_subgroup_tests_rev1.csv"),
     index=False
)

# Weak/strong localization tests per subgroup
"""
The weak test compares the heatmaps to the respective mode's baselines.
The strong test compares the heatmaps to inter-radiologist agreement.
Comparisons made using two-sided Tukey tests.
"""

subgroup_criteria_tests = {"Model": [], "Saliency Method": [], "Preprocessing Mode": [], "Subgroup": [],
                  "Mean AUPRC": [], "Std AUPRC": [],
                  "Mode-Matched Baseline Mean AUPRC": [], "Mode-Matched Baseline Std AUPRC": [], 
                  "AvgRad Mean AUPRC": [], "AvgRad Std AUPRC": [],
                  "Weak Localization Pvalue": [], "Strong Localization Pvalue": [],
                  "Weak Localization": [], "Strong Localization": []}

for cnn in cnn_list:
    for mth in methods:
        for j, mode in enumerate(modes):
            for subgroup in subgroups:
                # Remove from consideration images where the AI model was incorrect
                df = df_master.loc[df_master["Image"].isin(correctly_classified[cnn])]

                subgroup_criteria_tests["Model"].append(cnn)
                subgroup_criteria_tests["Saliency Method"].append(mth[1])
                subgroup_criteria_tests["Preprocessing Mode"].append(mode)
                subgroup_criteria_tests["Subgroup"].append(subgroup)
                subgroup_criteria_tests["Mean AUPRC"].append(
                    df[((df["Model"]==cnn) & (df["Saliency Method"]==mth[1]) & (df["Preprocessing Mode"]==mode)\
                        & (df["Subgroup"]==subgroup))]["AUPRC"].mean()
                )
                subgroup_criteria_tests["Std AUPRC"].append(
                    df[((df["Model"]==cnn) & (df["Saliency Method"]==mth[1]) & (df["Preprocessing Mode"]==mode)\
                        & (df["Subgroup"]==subgroup))]["AUPRC"].std()
                )
                subgroup_criteria_tests["Mode-Matched Baseline Mean AUPRC"].append(
                    df[(df["Model"]==baselines[j]) & (df["Subgroup"]==subgroup)]["AUPRC"].mean()
                )
                subgroup_criteria_tests["Mode-Matched Baseline Std AUPRC"].append(
                    df[(df["Model"]==baselines[j]) & (df["Subgroup"]==subgroup)]["AUPRC"].std()
                )
                # Weak version
                # We sift out either 1) the appropriate heatmaps or 2) mode matched baseline
                # The groups are in the model header
                weakdf = df[
                    ((df["Model"]==cnn) & (df["Saliency Method"]==mth[1]) & (df["Preprocessing Mode"]==mode)\
                     & (df["Subgroup"]==subgroup)) | ((df["Model"]==baselines[j]) & (df["Subgroup"]==subgroup))
                ]
                weak_test = pairwise_tukeyhsd(
                    endog=weakdf["AUPRC"],
                    groups=weakdf["Model"])
                subgroup_criteria_tests["Weak Localization Pvalue"].append(weak_test.pvalues[0])
                # Weak comparison: Mean AUPRC > Baseline AUPRC with significance
                if subgroup_criteria_tests["Weak Localization Pvalue"][-1] < alpha and (subgroup_criteria_tests["Mean AUPRC"][-1] - subgroup_criteria_tests["Mode-Matched Baseline Mean AUPRC"][-1] > 0):
                    subgroup_criteria_tests["Weak Localization"] = "Pass"
                else:
                    subgroup_criteria_tests["Weak Localization"] = "Fail"
                # Strong version
                # First we average the other radiologists AUPRCs per image
                interraddf = df[
                    ((df["Model"]=="rad1") | (df["Model"]=="rad2") | (df["Model"]=="rad3")) & \
                    (df["Subgroup"]==subgroup)
                ]
                interraddf = interraddf[["Image", "AUPRC"]]
                avgraddf = interraddf.groupby("Image", as_index=False).mean()
                avgraddf["Model"] = "AvgRadiologist"
                subgroup_criteria_tests["AvgRad Mean AUPRC"].append(avgraddf["AUPRC"].mean())
                subgroup_criteria_tests["AvgRad Std AUPRC"].append(avgraddf["AUPRC"].std())
                strongdf = pd.concat([
                    df[((df["Model"]==cnn) & (df["Saliency Method"]==mth[1]) & (df["Preprocessing Mode"]==mode)\
                        & (df["Subgroup"]==subgroup))][["Model", "Image", "AUPRC"]], 
                    avgraddf])
                strong_test = pairwise_tukeyhsd(
                    endog=strongdf["AUPRC"],
                    groups=strongdf["Model"])
                subgroup_criteria_tests["Strong Localization Pvalue"].append(strong_test.pvalues[0])
                # Strong comparison: Fail if Mean AUPRC < AvgRad AUPRC with significance
                if subgroup_criteria_tests["Strong Localization Pvalue"][-1] < alpha and (criteria_tests["AvgRad Mean AUPRC"][-1] - subgroup_criteria_tests["Mean AUPRC"][-1] > 0):
                    subgroup_criteria_tests["Strong Localization"] = "Fail"
                else:
                    subgroup_criteria_tests["Strong Localization"] = "Pass"

df_subgroup_criteria = pd.DataFrame().from_dict(subgroup_criteria_tests)
assert len(df_subgroup_criteria) == (4*6*3*3)
df_subgroup_criteria.to_csv(
     os.path.join(cfg["trustworthiness_dir"], "heatmap_localizations_subgroup_criteria_tests_rev1.csv"),
     index=False
)

# Sample efficiency tests per subgroup
"""
These tests compare the four different training subset sizes.
Comparisons made using two-sided Tukey tests.
The order of the comp tests is hardcoded to match TukeyHSD ordering
"""
# Ignore for now due to removing incorrectly predicted images
"""
interdata_tests = {"Saliency Method": [], "Preprocessing Mode": [], "Subgroup": [],
                   "100 % Mean AUPRC": [], "100 % Std AUPRC": [], 
                   "50 % Mean AUPRC": [], "50 % Std AUPRC": [],
                   "10 % Mean AUPRC": [], "10 % Std AUPRC": [], 
                   "1 % Mean AUPRC": [], "1 % Std AUPRC": [],
                   }
# hard coded order to match tukey groups order
subset_sizes = [1, 10, 100, 50]
for i in range(len(subset_sizes)):
    for j in range(i+1, len(subset_sizes)):
        s1 = subset_sizes[i]
        s2 = subset_sizes[j]
        interdata_tests["{} % vs. {} %".format(s1,s2)] = []

for mth in methods:
    for j, mode in enumerate(modes):
        for subgroup in subgroups:
            interdata_tests["Saliency Method"].append(mth[1])
            interdata_tests["Preprocessing Mode"].append(mode)
            interdata_tests["Subgroup"].append(subgroup)
            subdf = df[(df["Saliency Method"]==mth[1]) & (df["Preprocessing Mode"]==mode) & (df["Model"].str.contains("InceptionV3")) & \
                       (df["Subgroup"]==subgroup)].copy()
            interdata_tests["100 % Mean AUPRC"].append(
                subdf[subdf["Model"] == "InceptionV3_1_100"]["AUPRC"].mean()
            )
            interdata_tests["100 % Std AUPRC"].append(
                subdf[subdf["Model"] == "InceptionV3_1_100"]["AUPRC"].std()
            )
            interdata_tests["50 % Mean AUPRC"].append(
                subdf[subdf["Model"] == "InceptionV3_1_50"]["AUPRC"].mean()
            )
            interdata_tests["50 % Std AUPRC"].append(
                subdf[subdf["Model"] == "InceptionV3_1_50"]["AUPRC"].std()
            )
            interdata_tests["10 % Mean AUPRC"].append(
                subdf[subdf["Model"] == "InceptionV3_1_10"]["AUPRC"].mean()
            )
            interdata_tests["10 % Std AUPRC"].append(
                subdf[subdf["Model"] == "InceptionV3_1_10"]["AUPRC"].std()
            )
            interdata_tests["1 % Mean AUPRC"].append(
                subdf[subdf["Model"] == "InceptionV3_2_1"]["AUPRC"].mean()
            )
            interdata_tests["1 % Std AUPRC"].append(
                subdf[subdf["Model"] == "InceptionV3_2_1"]["AUPRC"].std()
            )
            # Adjust key names
            subdf.loc[:,"Training Subset %"] = subdf.loc[:,"Model"].map(lambda x: x.replace("InceptionV3_1_", "").replace("InceptionV3_2_", ""))
            test = pairwise_tukeyhsd(endog=subdf["AUPRC"], groups=subdf["Training Subset %"])
            # Double for-loop on groupsunique gives order of table
            c = 0
            for i in range(len(test.groupsunique)):
                for j in range(i+1, len(test.groupsunique)):
                    interdata_tests["{} % vs. {} %".format(test.groupsunique[i], test.groupsunique[j])].append(test.pvalues[c])
                    c += 1

df_interdata = pd.DataFrame().from_dict(interdata_tests)
assert len(df_interdata) == (6*3*3)
df_interdata.to_csv(
     os.path.join(cfg["trustworthiness_dir"], "heatmap_localizations_interdata_tests_per_subgroup_rev1.csv"),
     index=False
)
"""
