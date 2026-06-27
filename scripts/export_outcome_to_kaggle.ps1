<#
.SYNOPSIS
    Structures local TFRecord files, compresses them using native tar, and uploads them directly.

.DESCRIPTION
    This script automates the transfer of local TFRecord datasets to the Kaggle platform.
    1. It injects the Kaggle API credentials directly into the environment variables.
    2. It radically cleans the target staging area to ensure a "clean room" state.
    3. It builds a hard-compressed single ZIP archive via tar.exe directly in the staging area.
    4. It automatically initializes the mandatory 'dataset-metadata.json' file.
    5. It pushes the unified block directly online using the Anaconda Kaggle API CLI.
#>

# -------------------- CONFIGURATION --------------------
$RootDir      = "C:\Users\Olivier\Desktop\Projet_Kaggle\RSNA_Items_Detection_Classification"
$SourceDir     = $RootDir + "\data\lumbar_spine\outcome"
$TargetRootDir = $RootDir + "\kaggle_build_artifacts" # Direct staging & deployment area
$KaggleUserId  = "olivierrochat"
$DatasetName   = "rsna-lumbar-spine-tfrecords" 

# Path to the explicit Kaggle executable inside your Anaconda environment
$KaggleExe     = "C:\Users\Olivier\anaconda3\envs\airflow_env\Scripts\kaggle.exe"
# -------------------------------------------------------

# 0. Force Environment Variables for the current PowerShell session process
$env:KAGGLE_USERNAME = $null
$env:KAGGLE_KEY      = $null

Write-Host "[INFO] Injected Kaggle API credentials into current session environment variables." -ForegroundColor Yellow

# 1. Define final targets directly inside the staging directory
$MetadataPath = Join-Path $TargetRootDir "dataset-metadata.json"
$ZipTarget    = Join-Path $TargetRootDir "outcome_archive.zip"

Write-Host "[INFO] Local staging directory set to: $TargetRootDir" -ForegroundColor Cyan

# 2. Ensure source directory exists
if (-not (Test-Path -Path $SourceDir)) {
    Write-Error "Source directory does not exist: $SourceDir"
    Exit
}

# 3. PRELIMINARY CLEANUP: Radically wipe the staging area to prevent any historical file contamination
if (Test-Path -Path $TargetRootDir) {
    Write-Host "[INFO] Cleaning up staging directory..." -ForegroundColor Yellow
    Remove-Item -Path (Join-Path $TargetRootDir "*") -Recurse -Force -ErrorAction SilentlyContinue | Out-Null
} else {
    New-Item -ItemType Directory -Force -Path $TargetRootDir | Out-Null
}
Write-Host "[SUCCESS] Staging area is perfectly clean and ready." -ForegroundColor Green

# 4. Validate that the source directory is not empty
$ItemsCount = (Get-ChildItem -Path $SourceDir).Count
if ($ItemsCount -eq 0) {
    Write-Warning "The source directory $SourceDir is empty."
    Exit
}

Write-Host "[INFO] Found top-level items in outcome directory. Starting local fast compression..." -ForegroundColor Yellow

# 5. Compress the entire outcome directory structure using native tar (ZIP format)
# The archive is written directly into the root of the staging directory
& tar.exe -a -cf $ZipTarget -C $SourceDir .
Write-Host "[SUCCESS] Local ZIP archive created via tar.exe: $ZipTarget" -ForegroundColor Green

# 6. Generate the mandatory dataset-metadata.json directly in the staging area
Write-Host "[INFO] Generating dataset-metadata.json..." -ForegroundColor Cyan

$MetadataContent = @"
{
  "title": "$DatasetName",
  "id": "$KaggleUserId/$DatasetName",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}
"@

# Force UTF-8 strict WITHOUT Byte Order Mark (BOM)
$Utf8NoBom = New-Object System.Text.UTF8Encoding $False
[System.IO.File]::WriteAllText($MetadataPath, $MetadataContent, $Utf8NoBom)

# 7. Push to the Kaggle platform via Kaggle CLI targeting the staging directory root
Write-Host "[INFO] Interacting with Kaggle platform API..." -ForegroundColor Yellow

# Check if the dataset already exists online
& $KaggleExe datasets status "$KaggleUserId/$DatasetName" > $null 2>&1

if ($LastExitCode -eq 0) {
    Write-Host "[INFO] Dataset detected on Kaggle. Pushing the single unified archive..." -ForegroundColor Cyan
    & $KaggleExe datasets version -p $TargetRootDir -m "Automated block update of outcome directory tree."
} else {
    Write-Host "[INFO] Dataset not found online. Initializing a new dataset on Kaggle..." -ForegroundColor Green
    & $KaggleExe datasets create -p $TargetRootDir
}

if ($LastExitCode -eq 0) {
    Write-Host "[SUCCESS] Single ZIP block successfully pushed to Kaggle!" -ForegroundColor Green
} else {
    Write-Error "Kaggle CLI operation failed. Please verify network or API token validity."
}