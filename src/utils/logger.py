# src/utils/logger.py
from datetime import datetime

def log(line: str, path: str = "logs.txt"):
    ts = datetime.now().isoformat(timespec="seconds")
    with open(path, "a") as f:
        f.write(f"[{ts}] {line}\n")