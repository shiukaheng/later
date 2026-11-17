import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lars import store
from lars.factored import FactoredBandit


class ProfileTests(unittest.TestCase):
    def test_profiles_use_isolated_files_and_default_preserves_original_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            default = Path(tmp) / "data" / "session.json"
            with patch.object(store, "PATH", default):
                self.assertEqual(store.profile_path(), default)
                work = store.profile_path("work")
                personal = store.profile_path("personal")
                bandit = FactoredBandit({"topic": ["sports"]})
                store.save(bandit, [{"reward": 1}], {"topic": []}, work)
                self.assertIsNone(store.load(default))
                self.assertIsNone(store.load(personal))
                self.assertEqual(store.load(work)[1], [{"reward": 1}])

    def test_invalid_names_cannot_escape_profile_directory(self):
        for name in ("", "../other", ".", "some/name", "two words"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                store.profile_path(name)


if __name__ == "__main__":
    unittest.main()
