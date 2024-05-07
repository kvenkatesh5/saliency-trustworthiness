import math
from tqdm import tqdm
import random
import PIL
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.numeric import NaN
import torch
from torchvision import transforms
from sklearn.metrics import roc_auc_score, roc_curve, cohen_kappa_score
from sklearn.metrics import precision_recall_curve, auc
from skimage.metrics import structural_similarity as ssim
from skimage.filters import threshold_otsu
from scipy import ndimage
import cv2
from scipy.ndimage import binary_opening
from skimage.morphology import disk

IMAGENET_TRANSFORMER = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def preprocess_images(images):
    # images is 4D with range [0, 255]
    images = np.array(images)
    images = images / 255
    images = np.transpose(images, (0, 3, 1, 2))
    images = torch.tensor(images, dtype=torch.float32)
    images = IMAGENET_TRANSFORMER.forward(images)
    return images


def load_image(pth, sz):
    im = PIL.Image.open(pth).convert('RGB')
    im = im.resize(sz)
    im = np.asarray(im)
    im = im.astype(np.float32)
    return im


def normalize_image(image):
    image = np.array(image).astype(np.float32)
    image_min = np.min(image)
    image_max = np.max(image)
    norm = (image - image_min) / (image_max - image_min)
    return norm


def SSIM(maps1, maps2, nancheck=True):
    assert maps1.shape == maps2.shape, 'Maps do not have the same shape.'
    # single map
    if len(maps1.shape) == 2 and len(maps2.shape) == 2:
        return ssim(maps1, maps2)
    # array of maps
    scores = []
    for m1, m2 in zip(maps1, maps2):
        scores.append(ssim(m1, m2))
    # if we would like to check for nans (might chance output length)
    if nancheck:
        return [x for x in scores if np.isnan(x) == False]
    return [x for x in scores]

def resize_batch(imgs, to_sz):
    # imgs: N x S x S
    n = imgs.shape[0]
    rs_imgs = np.empty((n, to_sz, to_sz))
    for i in range(n):
        rs_imgs[i] = cv2.resize(imgs[i], (to_sz, to_sz), interpolation=cv2.INTER_AREA)
    return rs_imgs

def auprc(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)

def IoU(map1, map2):
    assert map1.shape == map2.shape, 'Maps do not have the same shape.'
    intersection = np.logical_and(map1, map2)
    union = np.logical_or(map1, map2)
    if np.sum(union) == 0:
        return np.nan
    return np.sum(intersection) / np.sum(union)

def dice(gt, seg):
    '''
    dice_coeff = 2 * TP / (2*TP+FP+FN)
    notice that:  TP+FP = sum(segmentation) & TP+FN = sum(ground truth)
    '''
    assert gt.shape == seg.shape, 'Maps do not have the same shape.'
    assert len(gt.shape) == 2, 'Maps are not 2D.'
    gt = gt > 0
    seg = seg > 0
    dice = np.sum(seg[gt == 1]) * 2.0 / (np.sum(seg) + np.sum(gt))
    return dice

def binary_error(gt, seg):
    '''
    pixel error between binary images
    '''
    assert gt.shape == seg.shape, 'Maps do not have the same shape.'
    gt = gt > 0
    seg = seg > 0
    error = (gt != seg)
    tot = (gt >= 0)
    return np.sum(error) / np.sum(tot)

def edge_canny(im_ndarr, low_threshold=100, ratio=3, kernel_size=3):
    im_blur = cv2.blur(im_ndarr, (kernel_size, kernel_size))
    im_blur = np.uint8(im_blur)
    edges = cv2.Canny(im_blur, low_threshold, low_threshold*ratio, kernel_size)
    return edges

def otsu(im):
    '''
    otsu thresholding on an image ([0,1])
    zeros out pixels with value less than threshold
    '''
    # If there are degenerate pixels
    if np.any(np.isnan(im)):
        return im
    # If map is of one class
    if np.all(im == 0) or np.all(im == 1):
        return im
    threshold = threshold_otsu(im)
    mask = im > threshold
    otsu = im * mask
    return otsu

def bbox(im):
    '''
    process image by replacing each connected component with the tightest encompassing bounding box
    binarizes image
    '''
    out = np.zeros(np.shape(im))
    # If there are degenerate pixels
    if np.any(np.isnan(im)):
        return out
    label, n_components = ndimage.label(im)
    R, C = np.shape(im)
    # [topmost row, bottommost row, leftmost col, rightmost col]
    bboxs = [[R+1, -1, C+1, -1] for _ in range(n_components)]
    for r in range(R):
        for c in range(C):
            cc = label[r,c]
            if cc == 0:
                continue
            bboxs[cc-1][0] = min(bboxs[cc-1][0], r)
            bboxs[cc-1][1] = max(bboxs[cc-1][1], r)
            bboxs[cc-1][2] = min(bboxs[cc-1][2], c)
            bboxs[cc-1][3] = max(bboxs[cc-1][3], c)
    for cc in range(1, 1+n_components):
        out[bboxs[cc-1][0]:bboxs[cc-1][1]+1,bboxs[cc-1][2]:bboxs[cc-1][3]+1] = 1
    return out

def binary_open(im, factor=32, iters=2):
    '''
    opens a binary map to remove noise
    '''
    radius = np.shape(im)[0]/factor
    structure = disk(radius)
    opened = binary_opening(im, structure=structure, iterations=iters)
    return opened

# Bootstrap fxn
def bootstrap(dat, n=1000, regions=['shoulder', 'humerus', 'elbow', 'forearm', 'wrist', 'hand', 'finger']):
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
        for li, l in enumerate(regions):
            dat_sub = dat_rs[dat_rs['region'] == l]
            if (dat_sub['y'].mean() > 0) and (dat_sub['yhat'].mean() < 1):
                aucs.append(roc_auc_score(dat_sub['y'], dat_sub['yhat']))
                weights.append(dat_sub['y'].sum() / dat_rs['y'].sum())
            else:
                aucs.append(np.NaN)
                weights.append(np.NaN)
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
        'kappa_score': kappa_score
    }

