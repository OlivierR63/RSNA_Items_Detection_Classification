<#
.SYNOPSIS
    Downloads and restores the extracted outcome directory structure directly from Kaggle.

.DESCRIPTION
    This script retrieves the uncompressed files from Kaggle and restores them.
    1. It sets up the Kaggle API execution environment.
    2. It pulls the files directly into the target destination directory.
#>

# -------------------- CONFIGURATION --------------------
$RootDir       = "C:\Users\Olivier\Desktop\Projet_Kaggle\RSNA_Items_Detection_Classification"
$TargetDestDir = $RootDir + "\data\lumbar_spine\outcome_restored" 
$KaggleUserId  = "olivierrochat"
$DatasetName   = "rsna-lumbar-spine-tfrecords" 

# Path to the explicit Kaggle executable inside your Anaconda environment
$KaggleExe     = "C:\Users\Olivier\anaconda3\envs\airflow_env\Scripts\kaggle.exe"
# -------------------------------------------------------

# 0. Force Environment Variables for the current PowerShell session process
$env:KAGGLE_USERNAME = $null
$env:KAGGLE_KEY      = $null

# 1. Prepare target destination (clean it if it exists to ensure accurate mirror)
if (Test-Path -Path $TargetDestDir) {
    Write-Host "[INFO] Cleaning up existing restoration directory..." -ForegroundColor Yellow
    Remove-Item -Path (Join-Path $TargetDestDir "*") -Recurse -Force -ErrorAction SilentlyContinue | Out-Null
} else {
    New-Item -ItemType Directory -Force -Path $TargetDestDir | Out-Null
}

# 2. Download the entire dataset files structure
Write-Host "[INFO] Pulling extracted tree from 'olivierrochat/rsna-lumbar-spine-tfrecords'..." -ForegroundColor Yellow

# Using the standard download command directly on the target folder
& $KaggleExe datasets download -d "$KaggleUserId/$DatasetName" -p $TargetDestDir --unzip

if ($LastExitCode -eq 0) {
    Write-Host "[SUCCESS] Entire directory tree successfully restored to: $TargetDestDir" -ForegroundColor Green
} else {
    Write-Error "Kaggle CLI download operation failed."
}
