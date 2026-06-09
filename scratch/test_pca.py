import os
import cv2 as cv
import numpy as np
import pandas as pd
import importlib
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Setup parent path so we can import the local all-script-accelerated module
sys.path.insert(0, os.path.dirname(os.path.abspath('__file__')))
acc = importlib.import_module('all-script-accelerated')

print("Loading dataset...", flush=True)
dst_dataset_dir = 'dataset/'
diverse_classes = {
    'A380', 'ATR-72', 'Fokker 100', 'MD-11', 'Cessna 172',
    '747-400', '737-800', 'BAE 146-200', 'DHC-6', 'E-190'
}

data = []
labels = []
sub_folders = os.listdir(dst_dataset_dir)
for sub_folder in sub_folders:
    if sub_folder not in diverse_classes:
        continue
    sub_folder_path = os.path.join(dst_dataset_dir, sub_folder)
    if not os.path.isdir(sub_folder_path):
        continue
    for filename in os.listdir(sub_folder_path):
        if filename == '.gitkeep':
            continue
        img_path = os.path.join(sub_folder_path, filename)
        img = cv.imread(img_path)
        if img is None:
            continue
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = acc.resize(img, 256, 256)
        data.append(acc.to_cpu(img))
        labels.append(sub_folder)

data = np.array(data)
labels = np.array(labels)
print(f"Loaded {len(data)} images.", flush=True)

# Preprocessing Stage 3
print("Applying Stage 3 preprocessing...", flush=True)
data_preprocessed = []
for img in data:
    s1 = acc.Enhancement.blur_gaussian(img, kernel_size=3)
    s1 = acc.Enhancement.blur_median(s1, kernel_size=3)
    s2 = acc.Equalization.clahe(s1, clip_limit=2.0)
    s2 = acc.Enhancement.gamma_correction(s2, gamma=0.9)
    s3 = acc.Enhancement.unsharp_mask(s2, sigma=1.0, strength=1.5)
    s3 = acc.Enhancement.sharpen(s3)
    data_preprocessed.append(acc.to_cpu(s3))
data_preprocessed = np.array(data_preprocessed)

# Use the best config from brute force
size = 64
ppc = 8
ori = 9
levels = 16
dists = (1,)

print(f"Extracting features (GLCM: levels={levels}, dists={dists} | HOG: size={size}, ppc={ppc}, ori={ori})...", flush=True)

# 1. GLCM features
factor = 256 // levels
glcm_feats_list = []
for img in data_preprocessed:
    quantized = (img // factor).clip(0, levels - 1)
    feats = acc.GLCM.features(quantized, distances=dists, angles=(0, 45, 90, 135), levels=levels, symmetric=True)
    flat_feat = []
    for name in ["contrast", "dissimilarity", "homogeneity", "energy", "entropy", "correlation", "asm"]:
        flat_feat.extend(feats[name].ravel())
    glcm_feats_list.append(flat_feat)
glcm_feats = np.array(glcm_feats_list)

# 2. HOG features
hog_feats_list = []
for img in data_preprocessed:
    img_small = acc.resize(img, size, size)
    hog_feat = acc.Feature_Extraction.hog_descriptor(img_small, orientations=ori, pixels_per_cell=ppc, cells_per_block=2)
    hog_feats_list.append(hog_feat)
hog_feats = np.array(hog_feats_list)

# Split indices
y = labels
X_train_indices, X_test_indices = train_test_split(np.arange(len(y)), test_size=0.2, random_state=67)
y_train, y_test = y[X_train_indices], y[X_test_indices]

# Combine features
X_combined = np.hstack([glcm_feats, hog_feats])

print(f"Combined features shape: {X_combined.shape}", flush=True)
print(f"GLCM part shape: {glcm_feats.shape}, HOG part shape: {hog_feats.shape}", flush=True)

# Define baseline evaluation
def evaluate_models(X_tr, X_te, prefix=""):
    # SVM
    svm = SVC(C=5.0, kernel='rbf', gamma='scale', random_state=67)
    svm.fit(X_tr, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_te))
    
    # RF
    rf = RandomForestClassifier(n_estimators=150, random_state=67, n_jobs=-1)
    rf.fit(X_tr, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_te))
    
    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_tr, y_train)
    knn_acc = accuracy_score(y_test, knn.predict(X_te))
    
    print(f"[{prefix}] SVM Acc: {svm_acc:.4f} | RF Acc: {rf_acc:.4f} | KNN Acc: {knn_acc:.4f}", flush=True)
    return svm_acc, rf_acc, knn_acc

# 1. Baseline: Scaled Combined Features without PCA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_combined[X_train_indices])
X_test_scaled = scaler.transform(X_combined[X_test_indices])
print("\n--- BASELINE (StandardScaler only) ---", flush=True)
evaluate_models(X_train_scaled, X_test_scaled, "Baseline")

# 2. Experiment A: PCA on combined scaled features
print("\n--- EXPERIMENT A: PCA on Combined Scaled Features ---", flush=True)
for n_comp in [10, 20, 30, 50, 100, 200, 300]:
    pca = PCA(n_components=n_comp, random_state=67)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    evaluate_models(X_train_pca, X_test_pca, f"PCA {n_comp} comps")

# 3. Experiment B: PCA on HOG only, then combine with GLCM
print("\n--- EXPERIMENT B: PCA on HOG only, then combine with GLCM ---", flush=True)
scaler_hog = StandardScaler()
X_train_hog_sc = scaler_hog.fit_transform(hog_feats[X_train_indices])
X_test_hog_sc = scaler_hog.transform(hog_feats[X_test_indices])

scaler_glcm = StandardScaler()
X_train_glcm_sc = scaler_glcm.fit_transform(glcm_feats[X_train_indices])
X_test_glcm_sc = scaler_glcm.transform(glcm_feats[X_test_indices])

for n_comp in [10, 20, 30, 50, 100, 200]:
    pca_hog = PCA(n_components=n_comp, random_state=67)
    X_train_hog_pca = pca_hog.fit_transform(X_train_hog_sc)
    X_test_hog_pca = pca_hog.transform(X_test_hog_sc)
    
    X_train_expB = np.hstack([X_train_glcm_sc, X_train_hog_pca])
    X_test_expB = np.hstack([X_test_glcm_sc, X_test_hog_pca])
    
    # Scale combined again or just use as is
    scaler_comb = StandardScaler()
    X_train_expB_sc = scaler_comb.fit_transform(X_train_expB)
    X_test_expB_sc = scaler_comb.transform(X_test_expB)
    
    evaluate_models(X_train_expB_sc, X_test_expB_sc, f"HOG-PCA {n_comp} + GLCM")
