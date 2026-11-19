import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lars.feed import profiles as store
from lars.generation.posts import Tweet
from lars.learning.bandit import FactoredBandit
from lars.session import Session


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

    def test_current_events_profile_can_resume_without_news(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            bandit = FactoredBandit({"style": ["plain"], "topic": ["sports"], "current_events": ["use news"]})
            levels = {"style": "plain", "topic": "sports", "current_events": "use news"}
            bandit.update(levels, 1)
            history = [{"levels": levels, "tweet": "An older post", "reward": 1}]
            store.save(bandit, history, {dim: [] for dim in bandit.dims}, path)

            with patch("lars.feed.profiles.profile_path", return_value=path), \
                 patch("lars.generation.posts.ask", return_value=Tweet(text="A new post")):
                session = Session("demo")
                try:
                    recommender = session.recommender
                    self.assertEqual(recommender.history, history)
                    self.assertEqual(recommender.bandit.dims["topic"].alpha[0], 2)
                    self.assertNotIn("current_events", recommender.bandit.dims)
                    self.assertNotIn("current_events", recommender.retired)
                    post = session.next()
                    self.assertEqual(post, {"id": 0, "text": "A new post"})
                    session.rate(post["id"], 1)
                    saved_bandit, saved_log, _ = store.load(path)
                    self.assertGreater(saved_bandit.dims["topic"].alpha[0], 2)
                    self.assertEqual(saved_log[0], history[0])
                    self.assertNotIn("current_events", saved_log[-1]["levels"])
                finally:
                    session.close()


if __name__ == "__main__":
    unittest.main()
