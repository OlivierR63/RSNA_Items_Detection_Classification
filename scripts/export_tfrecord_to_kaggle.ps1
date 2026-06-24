<#
.SYNOPSIS
    Structures local TFRecord files and uploads them as a dataset directly to Kaggle.

.DESCRIPTION
    This script automates the transfer of local TFRecord datasets to the Kaggle platform.
    1. It injects the Kaggle API credentials directly into the environment variables.
    2. It structures the files locally under a simulated layout.
    3. It automatically initializes the mandatory 'dataset-metadata.json' file.
    4. It pushes the dataset directly online using the Anaconda Kaggle API CLI.

.PARAMETER SourceDir
    The absolute or relative path to the local directory containing the generated TFRecord files.

.PARAMETER TargetRootDir
    The base directory where the Kaggle tree will be staged locally before upload.

.PARAMETER KaggleUserId
    Your official Kaggle Username (slug/ID) used for the platform path and metadata ownership.

.PARAMETER DatasetName
    The URL slug/name of the target Kaggle dataset (lowercase, alphanumeric, and dashes only).
#>

# -------------------- CONFIGURATION --------------------
$SourceDir     = "C:\Users\Olivier\Desktop\Projet_Kaggle\RSNA_Items_Detection_Classification\data\lumbar_spine\tfrecords"
$TargetRootDir = "C:\Users\Olivier\Desktop\Projet_Kaggle\RSNA_Items_Detection_Classification\kaggle_simulated"
$KaggleUserId  = "olivierrochat"
$DatasetName   = "dummy-rsna-lumbar-spine-tfrecords" # The dataset name must be lowercase,
                                                     # alphanumeric, and can include dashes only.

# Hardcoded Authentication Credentials (Failsafe for local environment profile)
$KaggleApiToken = 'KGAT_a525e5127e52f44d4d8d1cfabe46a69e'

# Path to the explicit Kaggle executable inside your Anaconda environment
$KaggleExe     = "C:\Users\Olivier\anaconda3\envs\airflow_env\Scripts\kaggle.exe"
# -------------------------------------------------------

# 0. Force Environment Variables for the current PowerShell session process
# Ancient versions of the Kaggle CLI are sanitized in order to avoid issues. 
$env:KAGGLE_USERNAME = $null
$env:KAGGLE_KEY      = $null

# Now inject the new official Kaggle API credentials into the current session environment variables
$env:KAGGLE_API_TOKEN = $KaggleApiToken

Write-Host "[INFO] Injected Kaggle API credentials into current session environment variables." -ForegroundColor Yellow

# 1. Build the full Kaggle compliant path: /kaggle/input/<KaggleUserId>/<DatasetName>
$KaggleSubPath = Join-Path "kaggle\input" $KaggleUserId
$KaggleSubPath = Join-Path $KaggleSubPath $DatasetName
$FullTargetDir = Join-Path $TargetRootDir $KaggleSubPath

Write-Host "[INFO] Local staging directory set to: $FullTargetDir" -ForegroundColor Cyan

# 2. Ensure source directory exists
if (-not (Test-Path -Path $SourceDir)) {
    Write-Error "Source directory does not exist: $SourceDir"
    Exit
}

# 3. Create target directory structure if missing
if (-not (Test-Path -Path $FullTargetDir)) {
    New-Item -ItemType Directory -Force -Path $FullTargetDir | Out-Null
    Write-Host "[INFO] Created missing target directories." -ForegroundColor Green
}

# 4. Count files to provide feedback
$Files = Get-ChildItem -Path $SourceDir -Filter "*.tfrecord"
if ($Files.Count -eq 0) {
    Write-Warning "No .tfrecord files found in $SourceDir"
    Exit
}

Write-Host "[INFO] Found $($Files.Count) TFRecord files to stage." -ForegroundColor Yellow

# 5. Copy files to the local staging area
Write-Host "[INFO] Copying files to local staging path..." -ForegroundColor Cyan
foreach ($File in $Files) {
    $DestinationPath = Join-Path $FullTargetDir $File.Name
    Copy-Item -Path $File.FullName -Destination $DestinationPath -Force
}

# 6. Generate the mandatory dataset-metadata.json for Kaggle CLI
Write-Host "[INFO] Generating dataset-metadata.json..." -ForegroundColor Cyan
$MetadataPath = Join-Path $FullTargetDir "dataset-metadata.json"

# We write the JSON as a raw string to avoid any PowerShell object serialization bugs
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

# 7. Push to the Kaggle platform via Kaggle CLI
Write-Host "[INFO] Interacting with Kaggle platform API..." -ForegroundColor Yellow

# Check if the dataset already exists online (output redirected to $null, we check the exit code)
& $KaggleExe datasets status "$KaggleUserId/$DatasetName" > $null 2>&1

if ($LastExitCode -eq 0) {
    Write-Host "[INFO] Dataset detected on Kaggle. Creating a new version..." -ForegroundColor Cyan
    & $KaggleExe datasets version -p $FullTargetDir -m "Automated update of TFRecord files from local PC."
} else {
    Write-Host "[INFO] Dataset not found online. Initializing a new dataset on Kaggle..." -ForegroundColor Green
    # Creates the dataset as private by default
    & $KaggleExe datasets create -p $FullTargetDir --dir-mode zip
}

if ($LastExitCode -eq 0) {
    Write-Host "[SUCCESS] Dataset successfully structured, processed, and pushed to Kaggle!" -ForegroundColor Green
} else {
    Write-Error "Kaggle CLI operation failed. Please verify that your API token is still valid."
}