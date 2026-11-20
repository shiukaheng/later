"""Headless recommendation loop: sample a strategy, produce content, learn from feedback."""

import logging
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from later.generation.posts import SEEDED, generate_tweet, propose_strategies
from later.generation.text import measure
from later.learning.bandit import FactoredBandit, ThompsonBandit, same
from later.learning.evolve import PROPOSE_EVERY, propose_level, retirement_decision, should_propose
from later.learning.posteriors import Checkpoint, capture

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
        propose: Propose = propose_level, checkpoints: list[Checkpoint] | None = None,
    ):
        self.bandit = bandit
        self.feed = feed
        self.history = history if history is not None else []
        self.retired = retired if retired is not None else {dim: [] for dim in bandit.dims}
        self._generate = generate
        self._propose = propose
        self.checkpoints = checkpoints if checkpoints is not None else []
        if checkpoints is None:
            self.record_posterior()

    def record_posterior(self, kind: str = "checkpoint", **details) -> bool:
        """Record the current state if it differs from the last checkpoint."""
        checkpoint = capture(self.bandit, len(self.history))
        previous = next((entry for entry in reversed(self.checkpoints) if "counts" in entry), None)
        if kind != "checkpoint" or previous is None or any(checkpoint[key] != previous[key] for key in ("rating", "counts")):
            checkpoint.update(kind=kind, at=datetime.now(UTC).isoformat(), details=details)
            self.checkpoints.append(checkpoint)
            return True
        return False

    @classmethod
    def fresh(cls, feed: str) -> Recommender:
        """Start from the demo's built-in options and model-proposed styles."""
        recommender = cls(FactoredBandit({"style": propose_strategies(feed), **SEEDED}), feed, checkpoints=[])
        recommender.record_posterior("initialized", origins={"style": "model", **dict.fromkeys(SEEDED, "seeded")})
        return recommender

    def choose(self) -> dict[str, str]:
        return self.bandit.levels(self.bandit.select())

    def recommend(self, levels: dict[str, str] | None = None) -> Recommendation:
        """Generate content and keep the strategy that should receive credit."""
        chosen = self.choose() if levels is None else dict(levels)
        return Recommendation(self._generate(chosen, self.feed), chosen)

    def record_served(self, post_id: int, item: Recommendation) -> None:
        self.checkpoints.append({
            "kind": "served", "rating": len(self.history), "at": datetime.now(UTC).isoformat(),
            "details": {"post_id": post_id, "levels": dict(item.levels), "text": item.text},
        })

    def learn(self, item: Recommendation, reward: int, *, post_id: int | None = None) -> bool:
        """Learn from a like (1) or skip (0). Return whether an option was retired."""
        if reward not in (0, 1):
            raise ValueError("reward must be 0 or 1")
        self.bandit.update(item.levels, reward)
        self.history.append({
            "levels": dict(item.levels), "tweet": item.text, "reward": reward, **measure(item.text),
        })
        self.record_posterior("rated", post_id=post_id, reward=reward, levels=dict(item.levels))
        return self._evolve_options()

    def add_option(self, dimension: str, text: str) -> None:
        bandit = self._dimension(dimension)
        text = self._new_text(dimension, text)
        bandit.add(text)
        self.record_posterior("added", dimension=dimension, option=text, origin="user")

    def edit_option(self, dimension: str, option: str, text: str) -> bool:
        """Archive the old wording and replace it in place with a fresh prior."""
        bandit = self._dimension(dimension)
        index = self._option_index(bandit, option)
        text = text.strip()
        if text == option:
            return False
        text = self._new_text(dimension, text, exclude=option)
        self.retired[dimension].append(option)
        bandit.arms[index] = text
        bandit.alpha[index] = bandit.beta[index] = 1
        self.record_posterior("edited", dimension=dimension, option=option, replacement=text, origin="user")
        return True

    def retire_option(self, dimension: str, option: str) -> None:
        bandit = self._dimension(dimension)
        index = self._option_index(bandit, option)
        if len(bandit.arms) == 1:
            raise ValueError("Keep at least one active option in each dimension.")
        self.retired[dimension].append(option)
        bandit.remove(index)
        self.record_posterior("retired", dimension=dimension, option=option, reason="manual")

    def override_evidence(self, dimension: str, option: str, field: str, value: float | None) -> None:
        """Set effective evidence above the prior without changing recorded feedback."""
        bandit = self._dimension(dimension)
        index = self._option_index(bandit, option)
        if field not in ("likes", "skips"):
            raise ValueError("Choose effective likes or skips.")
        if type(value) not in (int, float) or not 0 <= value <= 1e12 or not math.isfinite(value):
            raise ValueError("Evidence must be a finite number from 0 to 1 trillion.")
        counts = bandit.alpha if field == "likes" else bandit.beta
        before = float(counts[index] - 1)
        counts[index] = 1 + value
        self.record_posterior("override", dimension=dimension, option=option, field=field, before=before, after=value)

    def _dimension(self, dimension: str) -> ThompsonBandit:
        if dimension not in self.bandit.dims:
            raise ValueError("Unknown policy dimension.")
        return self.bandit.dims[dimension]

    @staticmethod
    def _option_index(bandit: ThompsonBandit, option: str) -> int:
        if option not in bandit.arms:
            raise ValueError("This option is no longer active. Refresh and try again.")
        return bandit.arms.index(option)

    def _new_text(self, dimension: str, text: str, *, exclude: str | None = None) -> str:
        text = text.strip()
        if not any(char.isalnum() for char in text):
            raise ValueError("Enter a non-empty instruction.")
        active = self.bandit.dims[dimension].arms
        if any(same(text, arm) for arm in active if arm != exclude) or text in self.retired[dimension]:
            raise ValueError("This instruction is already active or has been retired.")
        return text

    def _evolve_options(self) -> bool:
        n_ratings = len(self.history)
        dimensions = list(self.bandit.dims)
        turn = dimensions[(n_ratings // PROPOSE_EVERY) % len(dimensions)]
        retired_any = False
        for dim, bandit in self.bandit.dims.items():
            retired = self.retired[dim]
            if decision := retirement_decision(bandit):
                loser = decision.option
                replacement = self._propose(bandit, dim, self.history, retired, self.feed, loser)
                option = bandit.arms[loser]
                retired.append(option)
                bandit.remove(loser)
                added = replacement if replacement and replacement not in retired and bandit.add(replacement) else None
                evidence = asdict(decision)
                evidence.pop("option")
                self.record_posterior("retired", dimension=dim, option=option, reason="automatic",
                                      replacement=added, **evidence)
                retired_any = True
                log.info("retired %s option %r; replacement: %r", dim, option, replacement)
            if dim == turn and should_propose(bandit, n_ratings):
                option = self._propose(bandit, dim, self.history, retired, self.feed, None)
                if option and option not in retired and bandit.add(option):
                    self.record_posterior("added", dimension=dim, option=option, origin="model")
                    log.info("added %s option %r", dim, option)
        return retired_any
