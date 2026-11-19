"""Headless recommendation loop: sample a strategy, produce content, learn from feedback."""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from lars.generation.posts import SEEDED, generate_tweet, propose_strategies
from lars.generation.text import measure
from lars.learning.bandit import FactoredBandit, ThompsonBandit
from lars.learning.evolve import PROPOSE_EVERY, propose_level, should_propose, should_retire

log = logging.getLogger(__name__)

type Generate = Callable[[dict[str, str], str], str]
type Propose = Callable[[ThompsonBandit, str, list[dict], list[str], str, int | None], str | None]


@dataclass(frozen=True)
class Recommendation:
    text: str
    levels: dict[str, str]


class Recommender:
    """Own the learning state; callers handle scheduling, persistence, and presentation."""

    def __init__(
        self, bandit: FactoredBandit, feed: str, *, history: list[dict] | None = None,
        retired: dict[str, list[str]] | None = None, generate: Generate = generate_tweet,
        propose: Propose = propose_level,
    ):
        self.bandit = bandit
        self.feed = feed
        self.history = history if history is not None else []
        self.retired = retired if retired is not None else {dim: [] for dim in bandit.dims}
        self._generate = generate
        self._propose = propose

    @classmethod
    def fresh(cls, feed: str) -> Recommender:
        """Start from the demo's built-in options and model-proposed styles."""
        return cls(FactoredBandit({"style": propose_strategies(feed), **SEEDED}), feed)

    def choose(self) -> dict[str, str]:
        return self.bandit.levels(self.bandit.select())

    def recommend(self, levels: dict[str, str] | None = None) -> Recommendation:
        """Generate content and keep the strategy that should receive credit."""
        chosen = self.choose() if levels is None else dict(levels)
        return Recommendation(self._generate(chosen, self.feed), chosen)

    def learn(self, item: Recommendation, reward: int) -> bool:
        """Learn from a like (1) or skip (0). Return whether an option was retired."""
        if reward not in (0, 1):
            raise ValueError("reward must be 0 or 1")
        self.bandit.update(item.levels, reward)
        self.history.append({
            "levels": dict(item.levels), "tweet": item.text, "reward": reward, **measure(item.text),
        })
        return self._evolve_options()

    def _evolve_options(self) -> bool:
        n_ratings = len(self.history)
        dimensions = list(self.bandit.dims)
        turn = dimensions[(n_ratings // PROPOSE_EVERY) % len(dimensions)]
        retired_any = False
        for dim, bandit in self.bandit.dims.items():
            retired = self.retired[dim]
            if (loser := should_retire(bandit)) is not None:
                replacement = self._propose(bandit, dim, self.history, retired, self.feed, loser)
                option = bandit.arms[loser]
                retired.append(option)
                bandit.remove(loser)
                if replacement:
                    bandit.add(replacement)
                retired_any = True
                log.info("retired %s option %r; replacement: %r", dim, option, replacement)
            if dim == turn and should_propose(bandit, n_ratings):
                option = self._propose(bandit, dim, self.history, retired, self.feed, None)
                if option and bandit.add(option):
                    log.info("added %s option %r", dim, option)
        return retired_any
