

# We use -u for unbuffered python output so Tee-Object shows logs in real-time
& $PY -u $SRC 2>&1 | Tee-Object -FilePath $LOG