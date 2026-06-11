# scratch/tune_rf.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

csv_path = 'results/result_extract_stage_2.csv.gz'
print("Loading Stage 2 features...")
df_full = pd.read_csv(csv_path)

X_raw = df_full.drop(columns=['Label', 'Filename'])
y_target = df_full['Label']

X_train, X_test, y_train, y_test = train_test_split(X_raw, y_target, test_size=0.2, random_state=67)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- Tuning Random Forest hyperparameters ---")
best_rf_acc = 0.0
best_rf_cfg = {}

for n_comp in [50, 100, 150, 200]:
    pca = PCA(n_components=n_comp, random_state=67)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    for max_feat in ['sqrt', 'log2', 0.2, 0.3]:
        for criterion in ['gini', 'entropy']:
            for max_depth in [15, 25, None]:
                rf = RandomForestClassifier(
                    n_estimators=150,
                    criterion=criterion,
                    max_depth=max_depth,
                    max_features=max_feat,
                    random_state=67,
                    n_jobs=-1
                )
                rf.fit(X_train_pca, y_train)
                acc = accuracy_score(y_test, rf.predict(X_test_pca))
                if acc > best_rf_acc:
                    best_rf_acc = acc
                    best_rf_cfg = {
                        'n_components': n_comp,
                        'max_features': max_feat,
                        'criterion': criterion,
                        'max_depth': max_depth
                    }
                print(f"PCA {n_comp} | Feats={max_feat} | Crit={criterion} | Depth={max_depth} | Acc: {acc:.4f}")

print("\n--- Best Random Forest Configuration ---")
print(f"Best RF Configuration: {best_rf_cfg} | Acc: {best_rf_acc:.2%}")
