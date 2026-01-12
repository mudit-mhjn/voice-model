import json
from pathlib import Path

def load_json_if_exists(path: Path, default):
    if path.exists():
        return json.loads(path.read_text())
    return default
