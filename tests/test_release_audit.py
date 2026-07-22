from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_release", ROOT / "scripts" / "check_release.py"
)
CHECK_RELEASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CHECK_RELEASE)


class ReleaseAuditTests(unittest.TestCase):
    def test_repository_passes(self) -> None:
        self.assertEqual(CHECK_RELEASE.audit(ROOT), [])

    def test_weight_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.pth").write_bytes(b"not a real checkpoint")
            errors = CHECK_RELEASE.audit(root)
            self.assertTrue(any("forbidden release artifact" in item for item in errors))

    def test_dataset_readme_is_allowed_but_image_is_not(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "Datasets"
            dataset.mkdir()
            (dataset / "README.md").write_text("layout only", encoding="utf-8")
            self.assertEqual(CHECK_RELEASE.audit(root), [])
            (dataset / "sample.png").write_bytes(b"not an image")
            errors = CHECK_RELEASE.audit(root)
            self.assertTrue(
                any("generated/private directory content" in item for item in errors)
            )


if __name__ == "__main__":
    unittest.main()

