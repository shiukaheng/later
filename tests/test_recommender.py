import unittest
from unittest.mock import Mock, patch

from later.learning.bandit import FactoredBandit
from later.learning.posteriors import ledger_events, posterior_trends
from later.learning.recommender import Recommender


class RecommenderTests(unittest.TestCase):
    def test_manual_retirement_reserves_old_text_and_keeps_a_valid_dimension(self):
        bandit = FactoredBandit({"topic": ["cats", "sports"]})
        recommender = Recommender(bandit, "test feed")
        recommender.retire_option("topic", "sports")
        self.assertEqual(recommender.choose(), {"topic": "cats"})
        with self.assertRaises(ValueError):
            recommender.retire_option("topic", "cats")
        for text in ("", " Cats! ", "sports"):
            with self.assertRaises(ValueError):
                recommender.add_option("topic", text)
        recommender.add_option("topic", "science")
        self.assertEqual(bandit.dims["topic"].arms, ["cats", "science"])

    def test_headless_feedback_credits_the_strategy_used_to_generate(self):
        bandit = FactoredBandit({"topic": ["cats"], "style": ["deadpan"]})
        generate = Mock(return_value="The cat has declined to comment.")
        recommender = Recommender(bandit, "a test feed", generate=generate)

        item = recommender.recommend()
        generate.assert_called_once_with({"topic": "cats", "style": "deadpan"}, "a test feed")
        recommender.learn(item, 1)

        self.assertEqual(item.text, recommender.history[0]["tweet"])
        self.assertEqual(bandit.dims["topic"].alpha.tolist(), [2])
        self.assertEqual(bandit.dims["style"].alpha.tolist(), [2])
        self.assertEqual(set(recommender.history[0]), {"levels", "tweet", "reward", "chars", "emojis"})

    def test_headless_recommender_replaces_an_underperforming_option(self):
        bandit = FactoredBandit({"topic": ["cats", "sports"]})
        topic = bandit.dims["topic"]
        topic.alpha[:] = [101, 1]
        topic.beta[:] = [1, 101]
        propose = Mock(return_value="science")
        recommender = Recommender(bandit, "a test feed", generate=lambda *_: "A post", propose=propose)
        item = recommender.recommend({"topic": "sports"})

        with patch("later.learning.evolve.MIN_N", 10), patch("later.learning.evolve.THRESHOLD", 0.99), \
             patch("later.learning.recommender.PROPOSE_EVERY", 15), patch("later.learning.evolve.PROPOSE_EVERY", 15), \
             patch("later.learning.bandit.DISCOUNT", 1):
            self.assertTrue(recommender.learn(item, 0))

        self.assertEqual(topic.arms, ["cats", "science"])
        self.assertEqual((topic.alpha[-1], topic.beta[-1]), (1, 1))
        self.assertEqual(recommender.retired["topic"], ["sports"])
        self.assertEqual(propose.call_args.args[-1], 1)
        trends = posterior_trends(recommender.checkpoints)["topic"]
        self.assertEqual([point["rating"] for point in trends["sports"]], [0, 1])
        self.assertAlmostEqual(trends["sports"][-1]["estimate"], 1 / 103)
        self.assertEqual(trends["science"][0]["rating"], 1)
        self.assertEqual(trends["science"][0]["credible_interval"], [0.025, 0.975])
        event = ledger_events(recommender.checkpoints)[-1]
        self.assertEqual(event["kind"], "retired")
        self.assertEqual(event["removed"], [("topic", "sports")])
        self.assertEqual(event["added"], [("topic", "science")])
        self.assertEqual(event["details"]["comparison"], "cats")
        self.assertGreater(event["details"]["probability"], event["details"]["threshold"])
