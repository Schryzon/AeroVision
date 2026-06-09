#!/bin/bash
# run_notebooks.sh
# wtf: execute all AeroVision notebooks inside WSL with GPU acceleration.

set -e # exit immediately if any command fails

export PATH="$HOME/.local/bin:$PATH"

echo "========================================="
echo " AeroVision WSL Notebook Runner (Bash)"
echo "========================================="

# 1. install nbconvert and ipykernel if missing
if ! python3 -c "import nbconvert" 2>/dev/null; then
    echo "[*] Installing nbconvert and ipykernel in WSL..."
    python3 -m pip install nbconvert ipykernel --break-system-packages
else
    echo "[*] nbconvert is already installed."
fi

# 2. register ipykernel
python3 -m ipykernel install --user >/dev/null 2>&1

# 3. resolve CUDA library paths dynamically from pip packages
echo "[*] Resolving GPU CUDA library paths..."
CUDA_PATHS=$(python3 -c "import os, nvidia; print(':'.join([os.path.join(os.path.dirname(nvidia.__file__), d, 'lib') for d in os.listdir(os.path.dirname(nvidia.__file__)) if os.path.isdir(os.path.join(os.path.dirname(nvidia.__file__), d, 'lib'))]))")
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$CUDA_PATHS"
export TF_FORCE_GPU_ALLOW_GROWTH=true
echo "[*] LD_LIBRARY_PATH and TF_FORCE_GPU_ALLOW_GROWTH are configured."

# 4. run notebooks
targets=("$@")

# default to all if no arguments are passed
if [ ${#targets[@]} -eq 0 ]; then
    targets=("all")
fi

notebooks=(
    "AeroVision.ipynb"
    "Stage0_AeroVision.ipynb"
    "Stage1_AeroVision.ipynb"
    "Stage2_AeroVision.ipynb"
    "Stage3_AeroVision.ipynb"
    "Stage4_AeroVision.ipynb"
    "Stage5_AeroVision.ipynb"
    "Stage6_AeroVision.ipynb"
    "Stage7_AeroVision.ipynb"
)

cd /mnt/c/Users/nyoma/Downloads/AeroVision

# Helper function to check if an array contains an element
contains_element() {
    local e match="$1"
    shift
    for e; do [[ "$e" == "$match" ]] && return 0; done
    return 1
}

for nb in "${notebooks[@]}"; do
    # Skip if not targeting all and this is not in target list
    if ! contains_element "all" "${targets[@]}" && ! contains_element "$nb" "${targets[@]}"; then
        continue
    fi

    if [ -f "$nb" ]; then
        echo "----------------------------------------"
        echo "[*] Executing: $nb"
        start_time=$(date +%s)
        
        # run notebook execution in-place
        python3 -m jupyter nbconvert --to notebook --execute --inplace "$nb"
        
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        
        # calculate minutes and seconds
        mins=$((duration / 60))
        secs=$((duration % 60))
        
        echo "[+] Success: $nb completed in ${mins}m ${secs}s"
    else
        echo "[!] Skipping $nb - File not found."
    fi
done

echo "========================================="
echo " All notebooks processed successfully!"
echo "========================================="
