"""Check release paths and their correspondence to the experiment archive."""
from pathlib import Path
import sys
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from cli import SCRIPTS
from source_integrity import archive_sources, matches_source, training_hashes


class SourceLayoutTest(unittest.TestCase):
    def test_cli_targets(self):
        for action, filename in SCRIPTS.items():
            with self.subTest(action=action):
                self.assertTrue((SRC / "evaluation" / filename).is_file())

    def test_release_and_archive_digests(self):
        manifest = archive_sources()
        hashes = training_hashes(SRC / "training")
        self.assertEqual(set(hashes), {r["module"] for r in manifest.values()})
        for name, record in manifest.items():
            with self.subTest(module=record["module"]):
                digest = record["archive_sha256"]
                self.assertTrue(matches_source(name, digest, hashes, manifest))
                self.assertFalse(matches_source(name, "changed", hashes, manifest))
                changed = dict(hashes, **{record["module"]: "changed"})
                self.assertFalse(matches_source(name, digest, changed, manifest))
