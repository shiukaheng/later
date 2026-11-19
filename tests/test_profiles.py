import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

from lars.feed import profiles as store
from lars.generation.posts import Tweet
from lars.learning.bandit import FactoredBandit
from lars.learning.recommender import Recommendation
from lars.session import Session


class ProfileTests(unittest.TestCase):
    def test_evidence_override_persists_without_rewriting_rewards(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            bandit = FactoredBandit({"style": ["plain", "deadpan"]})
            history = [{"reward": 0, "tweet": "A post", "levels": {"style": "plain"}}]
            store.save(bandit, history, {"style": []}, path)
            with patch("lars.feed.profiles.profile_path", return_value=path), patch("lars.session.Prefetcher") as prefetch:
                session = Session("demo")
                session.change_option("evidence", "style", option="plain", field="likes", value=12.5)
                for invalid in (-1, float("nan"), float("inf"), True, None):
                    with self.assertRaises(ValueError):
                        session.change_option("evidence", "style", option="plain", field="likes", value=invalid)
                prefetch.return_value.refill.assert_called_once()
                self.assertEqual(session.queue_revision, 1)
                reloaded = Session("demo")
                snapshot = reloaded.policy_snapshot()
                option = snapshot["dimensions"]["style"]["active"][0]
                self.assertEqual((option["likes"], option["skips"], option["effective_n"]), (12.5, 0, 12.5))
                self.assertAlmostEqual(option["estimate"], 13.5 / 14.5)
                trace = snapshot["posterior_trends"]["style"]["plain"]
                self.assertEqual([point["rating"] for point in trace], [1, 1])
                self.assertLess(trace[0]["event"], trace[1]["event"])
                self.assertEqual(trace[0]["credible_interval"], [0.025, 0.975])
                low, high = trace[1]["credible_interval"]
                self.assertGreater(low, 0.7)
                self.assertLess(high - low, 0.3)
                self.assertEqual(snapshot["ratings"], [{"reward": 0, "levels": {"style": "plain"}}])

    def test_edit_preserves_old_feedback_without_crediting_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            bandit = FactoredBandit({"style": ["plain"], "topic": ["sports", "food"]})
            retired = {dim: [] for dim in bandit.dims}
            store.save(bandit, [], retired, path)
            with patch("lars.feed.profiles.profile_path", return_value=path), \
                 patch("lars.session.Prefetcher") as prefetch:
                session = Session("demo")
                ready = Future()
                ready.set_result(Recommendation("A sports post", {"topic": "sports", "style": "plain"}))
                prefetch.return_value.take.return_value = ready
                post = session.next()
                session.rate(post["id"], 1)

                pending = Future()
                pending.set_result(Recommendation("Another sports post", {"topic": "sports", "style": "plain"}))
                prefetch.return_value.take.return_value = pending
                second = session.next()
                session.change_option("edit", "topic", option="sports", text="Baseball stories")
                saved, _, archived, _ = store.load(path)
                self.assertEqual(saved.dims["topic"].arms, ["Baseball stories", "food"])
                self.assertEqual(archived["topic"], ["sports"])
                prefetch.return_value.refill.assert_called_once()

                session.rate(second["id"], 1)
                reloaded = Session("demo")
                snapshot = reloaded.policy_snapshot()
                self.assertEqual(snapshot["dimensions"]["topic"]["active"][0],
                                 {"text": "Baseball stories", "estimate": 0.5, "effective_n": 0,
                                  "likes": 0, "skips": 0})
                trace = snapshot["posterior_trends"]["topic"]
                self.assertEqual([point["rating"] for point in trace["sports"]], [0, 1])
                self.assertEqual([point["rating"] for point in trace["Baseball stories"]], [1, 2])
                self.assertEqual(trace["Baseball stories"][0]["credible_interval"], [0.025, 0.975])
                self.assertGreater(snapshot["dimensions"]["style"]["active"][0]["effective_n"], 1)
                self.assertEqual([rating["levels"]["topic"] for rating in snapshot["ratings"]], ["sports", "sports"])
                events = reloaded.policy_timeline()
                self.assertEqual([entry["kind"] for entry in events], ["checkpoint", "served", "rated", "served", "edited", "rated"])
                self.assertEqual(events[3]["details"]["post_id"], events[5]["details"]["post_id"])
                self.assertEqual(events[5]["details"]["levels"]["topic"], "sports")
                self.assertEqual(reloaded.next_id, 2)

    def test_in_flight_post_is_replaced_after_policy_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            bandit = FactoredBandit({"style": ["plain"]})
            store.save(bandit, [], {"style": []}, path)
            with patch("lars.feed.profiles.profile_path", return_value=path), \
                 patch("lars.session.Prefetcher") as prefetch:
                session = Session("demo")
                fresh = Future()
                fresh.set_result(Recommendation("New post", {"style": "deadpan"}))
                stale = Future()

                def finish_after_edit():
                    session.change_option("edit", "style", option="plain", text="deadpan")
                    return Recommendation("Old post", {"style": "plain"})

                with patch.object(stale, "result", side_effect=finish_after_edit):
                    prefetch.return_value.take.side_effect = [stale, fresh]
                    self.assertEqual(session.next(), {"id": 0, "text": "New post"})

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
                    saved_bandit, saved_log, _, _ = store.load(path)
                    self.assertGreater(saved_bandit.dims["topic"].alpha[0], 2)
                    self.assertEqual(saved_log[0], history[0])
                    self.assertNotIn("current_events", saved_log[-1]["levels"])
                finally:
                    session.close()


if __name__ == "__main__":
    unittest.main()
