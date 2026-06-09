import json
import sys
import os

notebook_path = "AeroVision.ipynb"
print(f"===========================================")
print(f" DRY RUN TESTING FOR MASTER: {notebook_path}")
print(f"===========================================")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Extract code cells
code_cells = [cell["source"] for cell in nb["cells"] if cell["cell_type"] == "code"]

# Compile and run code cells in order
global_scope = {}

# Inject Agg backend configuration at the start
import matplotlib
matplotlib.use('Agg')

for i, source in enumerate(code_cells):
    code_text = "".join(source)
    # Skip Colab git clone / Drive mount shell commands as they will fail in python eval
    cleaned_lines = []
    for line in code_text.splitlines():
        if line.strip().startswith("get_ipython()"):
            continue
        cleaned_lines.append(line)
    
    cleaned_code = "\n".join(cleaned_lines)
    print(f"\n--- Master Executing Cell {i} ---")
    try:
        exec(cleaned_code, global_scope)
    except Exception as e:
        print(f"Error in Master Cell {i}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\nMaster notebook dry run completed successfully!")
if 'df_compare' in global_scope:
    print("\n--- COMPARISON RESULTS ---")
    print(global_scope['df_compare'])
