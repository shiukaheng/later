import unittest
from unittest.mock import patch

from later.learning.bandit import FactoredBandit


class DiscountTests(unittest.TestCase):
    def test_all_evidence_ages_once_before_new_rating(self):
        bandit = FactoredBandit({"topic": ["tech", "sports"], "mood": ["calm"]})
        topic = bandit.dims["topic"]
        mood = bandit.dims["mood"]
        topic.alpha[:] = [5, 3]
        topic.beta[:] = [3, 5]
        mood.alpha[:] = [3]
        mood.beta[:] = [3]

        with patch("later.learning.bandit.DISCOUNT", 0.5):
            bandit.update({"topic": "tech", "mood": "retired"}, 1)

        self.assertEqual(topic.alpha.tolist(), [4, 2])
        self.assertEqual(topic.beta.tolist(), [2, 3])
        self.assertEqual(mood.alpha.tolist(), [2])
        self.assertEqual(mood.beta.tolist(), [2])

    def test_unit_discount_preserves_old_evidence(self):
        bandit = FactoredBandit({"topic": ["tech"]})
        with patch("later.learning.bandit.DISCOUNT", 1):
            bandit.update({"topic": "tech"}, 1)
            bandit.update({"topic": "tech"}, 0)
        self.assertEqual(bandit.dims["topic"].alpha[0], 2)
        self.assertEqual(bandit.dims["topic"].beta[0], 2)


if __name__ == "__main__":
    unittest.main()
