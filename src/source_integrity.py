"""Match frozen experiment source records to the renamed release modules."""
import hashlib
import json
from pathlib import Path


def training_hashes(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.glob("*.py")}


def archive_sources():
    return json.loads(Path(__file__).with_name("source_manifest.json").read_text(encoding="utf-8"))


def matches_source(name, digest, current, archived):
    if name in archived:
        record = archived[name]
        return (digest == record["archive_sha256"]
                and current.get(record["module"]) == record["release_sha256"])
    return current.get(name) == digest
