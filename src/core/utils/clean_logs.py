# coding: utf-8

from pathlib import Path
from datetime import datetime, timedelta


def clean_old_logs(log_dir: str = "config/logs", days: int = 30):
    """Deletes log files older than `days` days.

    Args:
        log_dir: Directory containing log files (default: "config/logs").
        days: Maximum age of log files to keep (default: 30).
    """
    log_dir = Path(log_dir)
    now = datetime.now()

    for log_file in log_dir.glob("*.log"):
        file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
        if now - file_time > timedelta(days=days):
            log_file.unlink()
            print(f"Deleting {log_file.name}")


if __name__ == "__main__":
    clean_old_logs()
