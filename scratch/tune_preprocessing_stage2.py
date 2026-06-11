# scratch/tune_preprocessing_stage2.py
import os
import cv2 as cv
import numpy as np
import pandas as pd
import importlib
import sys
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath('__file__')))
acc = importlib.import_module('all-script-accelerated')

cache_path = 'cache/data_cache.npz'
print(f"Loading cached images from {cache_path}...")
cache = np.load(cache_path, allow_pickle=True)
data = cache['data']
labels = cache['labels']
cache.close()

print(f"Loaded {len(data)} images. Augmenting to 3000...", flush=True)

data_augmented = []
labels_augmented = []
for i in range(len(data)):
    img = data[i]
    lbl = labels[i]
    data_augmented.append(img)
    flipped = acc.to_cpu(acc.Image_Ops.flip(img, axis='horizontal'))
    data_augmented.append(flipped)
    rotated = acc.to_cpu(acc.Image_Ops.rotate(img, angle=15.0, direction='ccw'))
    rotated = cv.resize(rotated, (256, 256), interpolation=cv.INTER_LINEAR)
    data_augmented.append(rotated)
    labels_augmented.extend([lbl, lbl, lbl])

data_augmented = np.array(data_augmented)
labels_augmented = np.array(labels_augmented)

# Base Feature extraction function
def extract_and_evaluate(prep_images, clip, gam):
    # GLCM: levels=16
    glcm_levels = 16
    factor = 256 // glcm_levels
    glcm_feats_list = []
    for img in prep_images:
        quantized = (img // factor).clip(0, glcm_levels - 1)
        feats = acc.GLCM.features(quantized, distances=(1, 2), angles=(0, 45, 90, 135), levels=glcm_levels, symmetric=True)
        flat_feat = []
        for name in ["contrast", "dissimilarity", "homogeneity", "energy", "entropy", "correlation", "asm"]:
            flat_feat.extend(feats[name].ravel())
        glcm_feats_list.append(flat_feat)
    glcm_feats = np.array(glcm_feats_list)
    
    # HOG: size=96, ppc=8, ori=9
    hog_feats_list = []
    for img in prep_images:
        img_small = cv.resize(img, (96, 96), interpolation=cv.INTER_LINEAR)
        hog_feat = acc.Feature_Extraction.hog_descriptor(img_small, orientations=9, pixels_per_cell=8, cells_per_block=2)
        hog_feats_list.append(hog_feat)
    hog_feats = np.array(hog_feats_list)
    
    X_combined = np.hstack([glcm_feats, hog_feats])
    X_train, X_test, y_train, y_test = train_test_split(X_combined, labels_augmented, test_size=0.2, random_state=67)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    pca = PCA(n_components=150, random_state=67)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    svm = SVC(C=5.0, kernel='rbf', gamma='scale', random_state=67)
    svm.fit(X_train_pca, y_train)
    acc_val = accuracy_score(y_test, svm.predict(X_test_pca))
    
    print(f"CLAHE clip={clip} | Gamma={gam} | SVM Acc: {acc_val:.4f}", flush=True)
    return acc_val

print("\n--- Tuning Stage 2: clip_limit and gamma ---", flush=True)
for clip in [1.5, 2.0, 2.5, 3.0]:
    for gam in [0.7, 0.8, 0.9, 1.0, 1.1]:
        # Apply custom Stage 2: Gaussian+Median -> CLAHE(clip) -> Gamma(gam)
        data_prep = []
        for img in data_augmented:
            s1 = acc.Enhancement.blur_gaussian(img, kernel_size=3)
            s1 = acc.Enhancement.blur_median(s1, kernel_size=3)
            s2 = acc.Equalization.clahe(s1, clip_limit=clip)
            s2 = acc.Enhancement.gamma_correction(s2, gamma=gam)
            data_prep.append(acc.to_cpu(s2))
        data_prep = np.array(data_prep)
        extract_and_evaluate(data_prep, clip, gam)
