# ExplainMURA: Trustworthiness Experiments

This document outlines how to use/adapt our code for preparing and running the trustworthiness experiments presented in our manuscript. If you use this repository, please cite our publication:

**Venkatesh, K., Mutasa, S., Moore, F. et al. Gradient-Based Saliency Maps Are Not Trustworthy Visual Explanations of Automated AI Musculoskeletal Diagnoses. J Digit Imaging. Inform. med. (2024). https://doi.org/10.1007/s10278-024-01136-4**

Please feel free to contact Kesavan for questions about this repository. Questions about overall study should be directed to corresponding author Paul Yi.

### Contact Information
*Kesavan Venkatesh* (first author): kvenka10 at alumni dot jh dot edu
*Paul Yi* (corresponding author): pyi at som dot umaryland dot edu

_This repository is being actively uploaded to GitHub as of May 6, 2024. Complete code will be available soon._

## Requirements
- Python 3.6.9
```
python3 -m venv tenv
source tenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Creating Datasets
- We use the publicly available MURA data (i.e., the Stanford team made public the training and validation splits from the MURA dataset) to create an in-house train/val/test split. We also curated radiologist annotations for a small subset of the test split (the "saliency test set").
- Download the open-source MURA dataset to your local system. Update the storage root in ```cfg.json``` under the ```data_dir``` header. Under this location, at ```[data_dir]/mura/images``` save all the images using the following format: ```[train/valid]_[anatomical region, all caps]_p[patient number]_s[study number]_image[image number].png```. Copy the uploaded text files in ```datafiles/``` to ```[data_dir]/mura/```. You can also see ```datafiles/``` for sample file naming. Update the ```saliency_test_images``` and ```cascade_test_images``` entries in the configuration JSON.
- The ```train_100``` and ```val_100``` and ```test_100``` files list our in-house dataset splits. The ```train_all.txt``` file comprises of all the training and validation set images; this file is used for training final models (i.e., post hyperparameter search).
- Table 1 in the manuscript can be replicated using ```figure_scripts/dataset_tables.py```. 

## Model Training
- Hyperparameter search can be done by pipelining the ```train.py``` script. See ```pipeline/hparam_lrwd.sh``` and its runners ```pipeline/hparam_runner[X]```.
- After hyperparameter search and selection, use the ```train.py``` script for training final models (i.e., models trained on the training and validation splits). See ```pipeline/train_densenet.sh``` and ```pipeline/train_inception.sh``` and their associated runners (```pipeline/densenet_runner.sh``` and ```pipeline/inception_runner[X]```) for sample runs.

## Model Evaluation
- Generate model predictions using the ```test.py``` script. See ```pipeline/test_final_models.sh``` for sample runs.
- Use ```eval.py``` and ```eval_stats_rev1.py``` and ```eval_plot.py``` for statistically evaluating these testing set predictions. An AUROC plot will be made by ```eval_plot.py``` and saved locally. A numpy zipfile with AUC and Kappa scores will be saved to the temp directory for further statistical evaluation by ```eval_stats_rev1.py```.

## Annotations
- We curated our radiologist annotations using the VGG software (https://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html) and saved them locally. Be sure to update the ```cfg.json``` with the save location (see ```annotations_dir``` entry).
- Use ```get_gt_blinded.py``` to synthesize these annotations into numpy arrays. Notice that we have blinded our radiologist names to ```rad1/2/3```; please adjust the naming to fit your needs.

## Trustworthiness Experiments
- Code to generate saliency heatmaps is in ```heatmaps.py```. See runners in ```pipeline/heatmap_runner[X].sh```.
- ```figure_scripts/saliency_map_preprocessing.py``` visualizes the two-step preprocessing routine we applied to saliency maps. See accompanying figure in supplementals S.1.
- Localization is done by ```localize_blinded.py```. Plots and statistics are done by ```localize_plots_rev1_blinded.py``` and ```localize_stats_rev1_blinded.py```. Again, radiologist names are blinded.

## Credit
We would like to specifically thank Zach Murphy (a former colleague and collaborator of Kesavan's, see https://doi.org/10.1148/ryai.220012) and Nisanth Arun (co-first author of https://doi.org/10.1148/ryai.2021200267) for their gracious help.
