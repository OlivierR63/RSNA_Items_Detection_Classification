# 1. Set the path toward the python executable
$PY = "C:\Users\Olivier\anaconda3\envs\airflow_env\python.exe"
$SRC  = "src/RSNA_2024_Lumbar_Spine_Degenerative_Classification.py"
$LOG  = "data/lumbar_spine/output/logs/full_session_output.log"

# 2. Remove the formerly existing log file
if (Test-Path $LOG) { Remove-Item $LOG -Force }

# 2. Launch the script , forcing the redirection of all streams toward the file
$OutputEncoding = [System.Text.Encoding]::UTF8
& $PY -u $SRC 2>&1 | Tee-Object -FilePath $LOG