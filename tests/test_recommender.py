import unittest
from unittest.mock import Mock, patch

from lars.learning.bandit import FactoredBandit
from lars.learning.recommender import Recommender


class RecommenderTests(unittest.TestCase):
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

        with patch("lars.learning.evolve.MIN_N", 10), patch("lars.learning.evolve.THRESHOLD", 0.99), \
             patch("lars.learning.recommender.PROPOSE_EVERY", 15), patch("lars.learning.evolve.PROPOSE_EVERY", 15), \
             patch("lars.learning.bandit.DISCOUNT", 1):
            self.assertTrue(recommender.learn(item, 0))

        self.assertEqual(topic.arms, ["cats", "science"])
        self.assertEqual((topic.alpha[-1], topic.beta[-1]), (1, 1))
        self.assertEqual(recommender.retired["topic"], ["sports"])
        self.assertEqual(propose.call_args.args[-1], 1)
