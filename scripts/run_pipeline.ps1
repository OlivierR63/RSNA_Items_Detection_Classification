# 1. Set the paths
$PY  = "C:\Users\Olivier\anaconda3\envs\airflow_env\python.exe"
$SRC = "src/RSNA_2024_Lumbar_Spine_Degenerative_Classification.py"
$LOG = "data/lumbar_spine/output/logs/full_session_output.log"

# 2. Extract the directory path from the log file path
$LOG_DIR = Split-Path -Parent $LOG

# 3. Ensure the directory exists (create it if it doesn't)
if (-not (Test-Path $LOG_DIR)) {
    New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
    Write-Host "Created missing directory: $LOG_DIR" -ForegroundColor Cyan
}

# 4. Remove the formerly existing log file if it exists
if (Test-Path $LOG) { 
    Remove-Item $LOG -Force 
}

# 5. Launch the script, forcing UTF8 and redirecting all streams
$OutputEncoding = [System.Text.Encoding]::UTF8

# We use -u for unbuffered python output so Tee-Object shows logs in real-time
& $PY -u $SRC 2>&1 | Tee-Object -FilePath $LOG