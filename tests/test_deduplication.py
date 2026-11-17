import unittest
from concurrent.futures import Future
from unittest.mock import patch

from lars.bandit import ThompsonBandit
from lars.evolve import Level, propose_level
from lars.factored import FactoredBandit
from lars.session import MAX_RETRIES, Session


class DeduplicationTests(unittest.TestCase):
    def test_merge_preserves_counts_and_history(self):
        session = Session.__new__(Session)
        session.bandit = FactoredBandit({"style": ["style"], "topic": ["Tech and AI", "tech and ai.", "sports"]})
        b = session.bandit.dims["topic"]
        b.update(0, 1)
        b.update(1, 0)
        session.log = [{"levels": {"topic": "tech and ai."}}]
        session.retired = {"topic": []}
        session._migrate()
        self.assertEqual(b.arms, ["Tech and AI", "sports"])
        self.assertEqual((b.alpha[0], b.beta[0]), (2, 2))
        self.assertEqual(session.log[0]["levels"]["topic"], "Tech and AI")
        self.assertFalse(b.add("tech and AI!"))

    def test_proposal_retries_duplicates_and_rejects_retired(self):
        b = ThompsonBandit(["tech and AI", "sports"])
        responses = [Level(level="Tech and AI."), Level(level="food"), Level(level="science")]
        with patch("lars.evolve.ask", side_effect=responses) as ask:
            result = propose_level(b, "topic", [], ["food"], "feed")
        self.assertEqual(result, "science")
        self.assertEqual(ask.call_count, 3)
        self.assertIn("previous proposal", ask.call_args.args[0])

    def test_long_tweets_are_never_served_on_last_retry(self):
        session = Session.__new__(Session)
        import threading
        session.lock = threading.Lock()
        future = Future()
        future.set_result(("x" * 281, None))
        session.prefetch = unittest.mock.Mock()
        session.prefetch.take.return_value = ({}, future)
        session.seen = set()
        with self.assertRaises(RuntimeError):
            session.next()
        self.assertEqual(session.prefetch.take.call_count, MAX_RETRIES + 1)


if __name__ == "__main__":
    unittest.main()
