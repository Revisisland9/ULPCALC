# src/storage.py
import json, os, time
from pathlib import Path
from typing import Optional

DATA_DIR = Path("data")
VERSIONS_DIR = DATA_DIR / "versions"
UPLOADS_DIR = DATA_DIR / "uploads"
VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

def save_upload(tmp_file, suffix: str) -> str:
    ts = int(time.time())
    path = UPLOADS_DIR / f"upload_{ts}{suffix}"
    with open(path, "wb") as f:
        f.write(tmp_file.read())
    return str(path)

def list_versions():
    files = sorted(VERSIONS_DIR.glob("*.json"), reverse=True)
    items = []
    for p in files:
        with open(p, "r") as f:
            doc = json.load(f)
        items.append({"path": str(p), "version_id": doc.get("version_id"), "created_at": doc.get("created_at"), "metrics": doc.get("metrics")})
    return items

def load_version(version_path: Optional[str] = None):
    if not version_path:
        files = sorted(VERSIONS_DIR.glob("*.json"), reverse=True)
        if not files:
            return None
        version_path = str(files[0])
    with open(version_path, "r") as f:
        return json.load(f)

def publish_version(payload: dict) -> str:
    ts = int(time.time())
    payload["version_id"] = f"v{ts}"
    payload["created_at"] = ts
    out = VERSIONS_DIR / f"{payload['version_id']}.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    return str(out)
