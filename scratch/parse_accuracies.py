# scratch/parse_accuracies.py
import json
import re
import os

accuracies = {}

for stage in range(8):
    filename = f"Stage{stage}_AeroVision.ipynb"
    if not os.path.exists(filename):
        print(f"Skipping {filename} - not found.")
        continue
        
    with open(filename, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    full_text = ""
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            for out in cell.get("outputs", []):
                if out.get("output_type") == "stream" and out.get("name") == "stdout":
                    full_text += "".join(out.get("text", [])) + "\n"
                    
    rf_acc = None
    svm_acc = None
    knn_acc = None
    cnn_acc = None
    
    # Parse RF
    rf_section = re.search(r"=== TRAINING RANDOM FOREST ===(.*?)=== TRAINING SVM ===", full_text, re.DOTALL)
    if rf_section:
        match = re.search(r"Accuracy:\s+([0-9.]+)", rf_section.group(1))
        if match:
            rf_acc = float(match.group(1))
            
    # Parse SVM
    svm_section = re.search(r"=== TRAINING SVM ===(.*?)=== TRAINING KNN ===", full_text, re.DOTALL)
    if svm_section:
        match = re.search(r"Accuracy:\s+([0-9.]+)", svm_section.group(1))
        if match:
            svm_acc = float(match.group(1))
            
    # Parse KNN
    knn_section = re.search(r"=== TRAINING KNN ===(.*?)(?:CNN Preprocessing|CNN Research Test Accuracy|$)", full_text, re.DOTALL)
    if knn_section:
        match = re.search(r"Accuracy:\s+([0-9.]+)", knn_section.group(1))
        if match:
            knn_acc = float(match.group(1))
            
    # Parse CNN
    match = re.search(r"CNN Research Test Accuracy:\s+([0-9.]+)", full_text)
    if match:
        cnn_acc = float(match.group(1))
                            
    accuracies[stage] = {
        "RF": rf_acc,
        "SVM": svm_acc,
        "KNN": knn_acc,
        "CNN": cnn_acc
    }

print("| Stage | Random Forest | SVM (RBF) | KNN | CNN (Research) |")
print("|-------|---------------|-----------|-----|----------------|")
for stage in range(8):
    data = accuracies.get(stage, {})
    rf = f"{data.get('RF', 0)*100:.2f}%" if data.get('RF') is not None else "TBD"
    svm = f"{data.get('SVM', 0)*100:.2f}%" if data.get('SVM') is not None else "TBD"
    knn = f"{data.get('KNN', 0)*100:.2f}%" if data.get('KNN') is not None else "TBD"
    cnn = f"{data.get('CNN', 0)*100:.2f}%" if data.get('CNN') is not None else "TBD"
    print(f"| Stage {stage} | {rf} | {svm} | {knn} | {cnn} |")
