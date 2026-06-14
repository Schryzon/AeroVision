# update_notebook_explanations.py
#
# wtf: script to update markdown/explanation cells in AeroVision notebooks
# using the generator script, without clearing the execution outputs.
#
# How it works:
# 1. Backs up existing executed notebooks (.ipynb -> .ipynb.bak)
# 2. Runs create_aerovision_notebook.py to generate fresh templates
# 3. Merges the execution outputs/counts from backup into the new templates
# 4. Cleans up backup files

import os
import json
import subprocess
import shutil

notebooks = [
    "AeroVision.ipynb",
    "Stage0_AeroVision.ipynb",
    "Stage1_AeroVision.ipynb",
    "Stage2_AeroVision.ipynb",
    "Stage3_AeroVision.ipynb",
    "Stage4_AeroVision.ipynb",
    "Stage5_AeroVision.ipynb",
    "Stage6_AeroVision.ipynb",
    "Stage7_AeroVision.ipynb"
]

def merge_notebooks(template_path, backup_path, output_path):
    print(f"[*] Merging outputs from {backup_path} into {template_path}...")
    with open(template_path, 'r', encoding='utf-8') as f:
        template = json.load(f)
    with open(backup_path, 'r', encoding='utf-8') as f:
        backup = json.load(f)
        
    template_cells = template.get('cells', [])
    backup_cells = backup.get('cells', [])
    
    t_code = [(i, c) for i, c in enumerate(template_cells) if c.get('cell_type') == 'code']
    b_code = [(i, c) for i, c in enumerate(backup_cells) if c.get('cell_type') == 'code']
    
    matched_backup_indices = set()
    
    # Pass 1: Match by exact source content
    for t_idx, t_cell in t_code:
        t_source = "".join(t_cell.get('source', []))
        for b_idx, b_cell in b_code:
            if b_idx in matched_backup_indices:
                continue
            b_source = "".join(b_cell.get('source', []))
            if t_source == b_source:
                t_cell['outputs'] = b_cell.get('outputs', [])
                t_cell['execution_count'] = b_cell.get('execution_count', None)
                matched_backup_indices.add(b_idx)
                break
                
    # Pass 2: Match remaining by relative position
    t_unmatched = [
        (i, c) for i, c in t_code 
        if 'outputs' not in c or (not c.get('outputs') and c.get('execution_count') is None)
    ]
    b_unmatched = [(i, c) for i, c in b_code if i not in matched_backup_indices]
    
    for (t_idx, t_cell), (b_idx, b_cell) in zip(t_unmatched, b_unmatched):
        t_cell['outputs'] = b_cell.get('outputs', [])
        t_cell['execution_count'] = b_cell.get('execution_count', None)
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=1)

def main():
    print("==================================================")
    # backup existing executed notebooks
    backup_created = []
    for nb in notebooks:
        if os.path.exists(nb):
            bak_path = nb + ".bak"
            print(f"[*] Backing up: {nb} -> {bak_path}")
            shutil.copy2(nb, bak_path)
            backup_created.append((nb, bak_path))
            
    if not backup_created:
        print("[!] No existing notebooks found to preserve outputs.")
        
    # Run the generator script to create new templates
    print("\n[*] Running create_aerovision_notebook.py to generate new templates...")
    result = subprocess.run(["python312", "create_aerovision_notebook.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print("[!] Failed to run create_aerovision_notebook.py:")
        print(result.stderr)
        # Restore backups and exit
        for nb, bak in backup_created:
            shutil.move(bak, nb)
        return

    print("[+] Templates generated successfully.\n")
    
    # Merge outputs back
    for nb, bak in backup_created:
        if os.path.exists(nb) and os.path.exists(bak):
            merge_notebooks(nb, bak, nb)
            # Remove backup
            os.remove(bak)
            print(f"[+] Restored and merged outputs for {nb}")
            
    print("\n==================================================")
    print("[+] All notebooks updated successfully with preserved outputs!")
    print("==================================================")

if __name__ == "__main__":
    main()
