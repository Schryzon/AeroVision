# scratch/parse_stage2_results.py
import json

with open("Stage2_AeroVision.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print("=== PARSING STAGE 2 NOTEBOOK RESULTS ===")
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        # Look for cells printing classifier metrics
        source = "".join(cell.get("source", []))
        outputs = cell.get("outputs", [])
        if "=== TRAINING" in source or "Accuracy:" in source or "CNN Test Accuracy" in source or "CNN" in source:
            for out in outputs:
                if out.get("output_type") == "stream" and out.get("name") == "stdout":
                    text = "".join(out.get("text", []))
                    if "Accuracy:" in text or "Training Time" in text or "Test Accuracy" in text:
                        print(text.strip())
