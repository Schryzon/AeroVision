"""Profile the OPTIMIZED AeroVision pipeline to compare against baseline."""
import os, sys, time, importlib
import numpy as np
import cv2 as cv

sys.path.insert(0, os.path.dirname(os.path.abspath('__file__')))
acc = importlib.import_module('all-script-accelerated')

timings = {}

def timed(label):
    class Timer:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self
        def __exit__(self, *a):
            elapsed = time.perf_counter() - self.t0
            timings[label] = elapsed
            print(f"  [{label}] {elapsed:.2f}s", flush=True)
    return Timer()

# ── 1. Data Loading ──────────────────────────────────────────────────────────
with timed("1. Data Loading (1000 imgs, resize 256x256)"):
    dst_dataset_dir = 'dataset/'
    diverse_classes = {
        'A380', 'ATR-72', 'Fokker 100', 'MD-11', 'Cessna 172',
        '747-400', '737-800', 'BAE 146-200', 'DHC-6', 'E-190'
    }
    data, labels = [], []
    for sub in os.listdir(dst_dataset_dir):
        if sub not in diverse_classes:
            continue
        p = os.path.join(dst_dataset_dir, sub)
        if not os.path.isdir(p):
            continue
        for f in os.listdir(p):
            if f == '.gitkeep':
                continue
            img = cv.imread(os.path.join(p, f))
            if img is None:
                continue
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = acc.resize(img, 256, 256)
            data.append(acc.to_cpu(img))
            labels.append(sub)
    data = np.array(data)
    labels = np.array(labels)

# ── 2. Augmentation ──────────────────────────────────────────────────────────
with timed("2. Augmentation (flip + rotate -> 3x)"):
    data_aug, labels_aug = [], []
    for i in range(len(data)):
        img, lbl = data[i], labels[i]
        data_aug.append(acc.to_cpu(img))
        labels_aug.append(lbl)
        flipped = acc.to_cpu(acc.Image_Ops.flip(img, axis='horizontal'))
        data_aug.append(flipped)
        labels_aug.append(lbl)
        rotated = acc.to_cpu(acc.Image_Ops.rotate(img, angle=15.0, direction='ccw'))
        rotated = acc.to_cpu(acc.resize(rotated, 256, 256))
        data_aug.append(rotated)
        labels_aug.append(lbl)
    data_aug = np.array(data_aug)

# ── 3. Preprocessing ────────────────────────────────────────────────────────
with timed("3. Preprocessing Stage 1 (blur_gaussian + blur_median)"):
    data_prep = []
    for img in data:
        s1 = acc.Enhancement.blur_gaussian(img, kernel_size=3)
        s1 = acc.Enhancement.blur_median(s1, kernel_size=3)
        data_prep.append(acc.to_cpu(s1))
    data_prep = np.array(data_prep)

# ── 4. GLCM ─────────────────────────────────────────────────────────────────
with timed("4. GLCM Extraction (levels=16, dists=(1,2), 4 angles, 1000 imgs)"):
    GLCM_LEVELS = 16
    factor = 256 // GLCM_LEVELS
    glcm_feats = []
    for img in data_prep:
        q = (img // factor).clip(0, GLCM_LEVELS - 1)
        feats = acc.GLCM.features(q, distances=(1, 2), angles=(0, 45, 90, 135), levels=GLCM_LEVELS, symmetric=True)
        flat = []
        for name in ["contrast", "dissimilarity", "homogeneity", "energy", "entropy", "correlation", "asm"]:
            flat.extend(feats[name].ravel())
        glcm_feats.append(flat)
    glcm_feats = np.array(glcm_feats)

# ── 5. HOG (OPTIMIZED - vectorized) ─────────────────────────────────────────
with timed("5. HOG Extraction [OPTIMIZED] (96x96, ppc=8, ori=9, 1000 imgs)"):
    hog_feats = []
    for img in data_prep:
        img_small = acc.resize(img, 96, 96)
        hog_feat = acc.Feature_Extraction.hog_descriptor(img_small, orientations=9, pixels_per_cell=8, cells_per_block=2)
        hog_feats.append(hog_feat)
    hog_feats = np.array(hog_feats)

# ── 6. Feature Selection (OPTIMIZED - np.corrcoef) ──────────────────────────
import pandas as pd
with timed("6. Feature Selection [OPTIMIZED] (np.corrcoef, threshold=0.95)"):
    combined = np.hstack([glcm_feats, hog_feats])
    # np.corrcoef instead of pd.DataFrame.corr()
    corr_matrix = np.abs(np.corrcoef(combined, rowvar=False))
    np.fill_diagonal(corr_matrix, 0.0)
    n = corr_matrix.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if keep[i]:
            keep[(i+1):][corr_matrix[i, (i+1):] >= 0.95] = False
    X_selected = combined[:, keep]

# ── 7. Train/Test Split + Z-score ───────────────────────────────────────────
from sklearn.model_selection import train_test_split
with timed("7. Train/Test Split + Z-score Normalization"):
    X_train, X_test, y_train, y_test = train_test_split(X_selected, labels, test_size=0.2, random_state=67)
    mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

# ── 8. Model Training ───────────────────────────────────────────────────────
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

with timed("8a. Train Random Forest"):
    rf = RandomForestClassifier(n_estimators=100, random_state=67, n_jobs=-1)
    rf.fit(X_train, y_train)

with timed("8b. Train SVM (C=5.0, rbf)"):
    svm = SVC(C=5.0, kernel='rbf', gamma='scale', random_state=67)
    svm.fit(X_train, y_train)

with timed("8c. Train KNN (k=5)"):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

# ── 9. Visualization ────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

with timed("9. Visualization"):
    unique_cls = sorted(list(diverse_classes))
    fig, axes = plt.subplots(len(unique_cls), 2, figsize=(12, 40))
    for i, cls in enumerate(unique_cls):
        idx = np.where(labels == cls)[0]
        if len(idx) > 0:
            axes[i, 0].imshow(data[idx[0]], cmap='gray')
            axes[i, 1].imshow(data_prep[idx[0]], cmap='gray')
    plt.close(fig)

# ── 10. CSV Write (OPTIMIZED - shutil.copy2) ────────────────────────────────
import shutil
with timed("10. CSV Write [OPTIMIZED] (single write + shutil.copy2)"):
    df_full = pd.DataFrame(np.hstack([glcm_feats, hog_feats]))
    df_full.to_csv('scratch/_profile_test.csv', index=False)
    shutil.copy2('scratch/_profile_test.csv', 'scratch/_profile_test_copy.csv')
    os.remove('scratch/_profile_test.csv')
    os.remove('scratch/_profile_test_copy.csv')

# ═══════════════════════════════════════════════════════════════════════════════
# baseline timings from the first profiler run
baseline = {
    "1. Data Loading": 10.26,
    "2. Augmentation": 1.04,
    "3. Preprocessing": 0.17,
    "4. GLCM Extraction": 4.94,
    "5. HOG Extraction": 20.95,
    "6. Feature Selection": 103.76,
    "7. Train/Test Split": 0.07,
    "8a. Random Forest": 0.71,
    "8b. SVM": 1.67,
    "8c. KNN": 0.35,
    "9. Visualization": 0.34,
    "10. CSV Write": 7.36,
}

print("\n" + "=" * 80)
print("  OPTIMIZED PIPELINE PROFILING RESULTS")
print("=" * 80)

total = sum(timings.values())
baseline_total = sum(baseline.values())
sorted_items = sorted(timings.items(), key=lambda x: -x[1])

for label, elapsed in sorted_items:
    pct = elapsed / total * 100
    bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
    print(f"  {bar} {pct:5.1f}%  {elapsed:6.2f}s  {label}")

print(f"\n  OPTIMIZED TOTAL: {total:.2f}s  (was {baseline_total:.2f}s)")
print(f"  SPEEDUP: {baseline_total / total:.1f}x faster!")
print("=" * 80)
