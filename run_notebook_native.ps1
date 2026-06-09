# run_notebook_native.ps1
#
# wtf: native Windows PowerShell script to execute AeroVision notebooks.
# runs notebooks in-place using jupyter nbconvert on the Windows python312 installation.

param (
    [string[]]$notebooks = @("all")
)

$all_notebooks = @(
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

Write-Host "=========================================" -ForegroundColor Green
Write-Host " AeroVision Native Windows Notebook Runner" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green

# 1. Resolve targets
$targets = @()
if ($notebooks -contains "all" -or $notebooks.Length -eq 0) {
    $targets = $all_notebooks
} else {
    foreach ($nb in $notebooks) {
        # Check if the notebook ends with .ipynb, if not, add it
        if (-not $nb.EndsWith(".ipynb")) {
            $targets += ($nb + ".ipynb")
        } else {
            $targets += $nb
        }
    }
}

# 2. Check if jupyter / nbconvert is installed
try {
    python312 -c "import nbconvert, ipykernel" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[*] Installing nbconvert and ipykernel on Windows..." -ForegroundColor Yellow
        python312 -m pip install nbconvert ipykernel
    } else {
        Write-Host "[*] nbconvert and ipykernel are already installed on Windows." -ForegroundColor Gray
    }
} catch {
    Write-Host "[!] Error checking/installing jupyter. Make sure python312 is available in path." -ForegroundColor Red
    exit 1
}

# 3. Execute target notebooks
foreach ($nb in $targets) {
    if (Test-Path $nb) {
        Write-Host "----------------------------------------" -ForegroundColor White
        Write-Host "[*] Executing: $nb" -ForegroundColor Yellow
        $start_time = [DateTime]::Now
        
        # Run nbconvert execution in-place
        python312 -m jupyter nbconvert --to notebook --execute --inplace $nb
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[!] Execution failed for $nb" -ForegroundColor Red
            exit $LASTEXITCODE
        }
        
        $end_time = [DateTime]::Now
        $duration = $end_time - $start_time
        $mins = [Math]::Floor($duration.TotalMinutes)
        $secs = [Math]::Floor($duration.TotalSeconds % 60)
        
        Write-Host "[+] Success: $nb completed in $($mins)m $($secs)s" -ForegroundColor Green
    } else {
        Write-Host "[!] Skipping $nb - File not found." -ForegroundColor DarkYellow
    }
}

Write-Host "=========================================" -ForegroundColor Green
Write-Host " All target notebooks processed successfully!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
