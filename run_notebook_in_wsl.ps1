# run_notebook_in_wsl.ps1
#
# wtf: simple script to run the WSL notebook execution pipeline.
# delegates all complexity to a native bash script to avoid PowerShell quoting bugs.

param (
    [string[]]$notebooks = @("all")
)

$script_path = "c:\Users\nyoma\Downloads\AeroVision\scratch\run_notebooks.sh"

if (-not (Test-Path $script_path)) {
    Write-Host "[!] Error: Bash script not found at $script_path" -ForegroundColor Red
    exit 1
}

# sanitize line endings to prevent CRLF execution issues in bash
Write-Host "[*] Sanitizing script line endings..." -ForegroundColor Gray
wsl -d Ubuntu sed -i 's/\r$//' /mnt/c/Users/nyoma/Downloads/AeroVision/scratch/run_notebooks.sh

# run the bash runner inside WSL Ubuntu, forwarding the notebooks array as arguments
Write-Host "[*] Starting WSL execution..." -ForegroundColor Yellow
wsl -d Ubuntu bash /mnt/c/Users/nyoma/Downloads/AeroVision/scratch/run_notebooks.sh $notebooks

if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] WSL execution failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "[+] Finished executing notebook(s)!" -ForegroundColor Green
