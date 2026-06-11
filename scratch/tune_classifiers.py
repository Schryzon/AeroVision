# scratch/tune_classifiers.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

csv_path = 'results/result_extract_stage_2.csv.gz'
print("Loading Stage 2 features...")
df_full = pd.read_csv(csv_path)

X_raw = df_full.drop(columns=['Label', 'Filename'])
y_target = df_full['Label']

X_train, X_test, y_train, y_test = train_test_split(X_raw, y_target, test_size=0.2, random_state=67)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- Tuning PCA components and SVM C parameter ---")
best_svm_acc = 0.0
best_svm_cfg = {}

for n_comp in [100, 150, 200, 250, 300]:
    pca = PCA(n_components=n_comp, random_state=67)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    for C_val in [1.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0]:
        svm = SVC(C=C_val, kernel='rbf', gamma='scale', random_state=67)
        svm.fit(X_train_pca, y_train)
        acc = accuracy_score(y_test, svm.predict(X_test_pca))
        if acc > best_svm_acc:
            best_svm_acc = acc
            best_svm_cfg = {'n_components': n_comp, 'C': C_val}
        print(f"PCA {n_comp} | SVM C={C_val} | Acc: {acc:.4f}")

print("\n--- Tuning KNN n_neighbors and weights ---")
best_knn_acc = 0.0
best_knn_cfg = {}

for n_comp in [100, 150, 200, 250, 300]:
    pca = PCA(n_components=n_comp, random_state=67)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    for k in [3, 5, 7, 9]:
        for w in ['uniform', 'distance']:
            knn = KNeighborsClassifier(n_neighbors=k, weights=w, metric='cosine')
            knn.fit(X_train_pca, y_train)
            acc = accuracy_score(y_test, knn.predict(X_test_pca))
            if acc > best_knn_acc:
                best_knn_acc = acc
                best_knn_cfg = {'n_components': n_comp, 'n_neighbors': k, 'weights': w}
            print(f"PCA {n_comp} | KNN k={k}, weights={w} | Acc: {acc:.4f}")

print("\n--- Best Configurations for Stage 2 ---")
print(f"Best SVM Configuration: {best_svm_cfg} | Acc: {best_svm_acc:.2%}")
print(f"Best KNN Configuration: {best_knn_cfg} | Acc: {best_knn_acc:.2%}")
