import logging
import os
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel

from lars.generation.llm import ask
from lars.learning.bandit import ThompsonBandit, same

logger = logging.getLogger("lars")

# Retire a level if P(worse than the best tested level) > THRESHOLD and it has >= MIN_N
# ratings. Simulated trade-off, per dimension of 6 levels over 300 ratings at a 30% like rate:
#   0.95/5:  ~3.9 false retirements of equally good levels; a clearly bad level caught fast
#   0.99/10: ~0.3 false retirements; a bad level caught in ~half of runs, after ~300 ratings
THRESHOLD = float(os.environ.get("LARS_RETIRE_THRESHOLD", 0.99))
MIN_N = int(os.environ.get("LARS_RETIRE_MIN_N", 10))
EXAMPLES = 3  # tweets of each kind shown to the LLM
PROPOSE_EVERY = int(os.environ.get("LARS_PROPOSE_EVERY", 15))  # add a new level every N ratings
MAX_LEVELS = int(os.environ.get("LARS_MAX_LEVELS", 8))  # ...unless the dimension already has this many


def tested_best(bandit: ThompsonBandit) -> int | None:
    """Best level among those with at least MIN_N ratings. Untested levels sit at the
    prior mean 0.5, which would beat everything when the user rarely likes."""
    n = bandit.alpha + bandit.beta - 2
    tested = [i for i in range(len(n)) if n[i] >= MIN_N]
    return max(tested, key=lambda i: bandit.means()[i]) if tested else None


@dataclass(frozen=True)
class Retirement:
    option: int
    comparison: str
    probability: float
    threshold: float
    effective_n: float
    comparison_n: float


def retirement_decision(bandit: ThompsonBandit, samples: int = 4000) -> Retirement | None:
    """Return the tested level most confidently worse than the best tested level, if any."""
    best = tested_best(bandit)
    if best is None:
        return None
    draws = bandit.rng.beta(bandit.alpha, bandit.beta, size=(samples, len(bandit.arms)))
    p_worse = (draws < draws[:, [best]]).mean(axis=0)
    n = bandit.alpha + bandit.beta - 2
    ok = [i for i in range(len(p_worse)) if n[i] >= MIN_N and p_worse[i] > THRESHOLD]
    if not ok:
        return None
    loser = max(ok, key=lambda i: p_worse[i])
    return Retirement(int(loser), bandit.arms[best], float(p_worse[loser]), THRESHOLD,
                      float(n[loser]), float(n[best]))


def should_propose(bandit: ThompsonBandit, n_ratings: int) -> bool:
    """Periodically grow the dimension with a fresh level, up to a cap."""
    return n_ratings % PROPOSE_EVERY == 0 and len(bandit.arms) < MAX_LEVELS


class Level(BaseModel):
    level: str


def propose_level(
    bandit: ThompsonBandit,
    dim: str,
    log: list[dict],
    retired: list[str],
    feed: str,
    dead: int | None = None,
    attempts: int = 3,
) -> str | None:
    """Ask the LLM for one new level of `dim`, conditioned on the existing levels,
    their like rates, example tweets, and everything retired. If `dead` is given,
    the new level replaces that one. Returns None if it only ever repeats known levels."""
    b_i = tested_best(bandit)
    best = bandit.arms[b_i if b_i is not None else int(np.argmax(bandit.means()))]

    def tweets(level: str, reward: int) -> str:
        hits = [e["tweet"] for e in log if e["levels"].get(dim) == level and e["reward"] == reward]
        return "\n".join(f"- {t}" for t in hits[-EXAMPLES:]) or "- (none)"

    stats = "\n".join(
        f"- {s} (like rate {m:.2f}, effective n={a + b - 2:.1f})"
        for s, m, a, b in zip(bandit.arms, bandit.means(), bandit.alpha, bandit.beta)
    )
    excluded = retired + ([bandit.arms[dead]] if dead is not None else [])
    parts = [
        f"We write tweets for {feed}. A user likes or skips each tweet.",
        f"Tweets vary along several dimensions; here we tune the '{dim}' dimension.",
        f"Current {dim} options:\n{stats}",
        f"Best {dim}: {best}\nTweets with it the user liked:\n{tweets(best, 1)}",
    ]
    if dead is not None:
        d = bandit.arms[dead]
        parts.append(
            f"Retiring this underperforming {dim}: {d}\n"
            f"Tweets with it the user skipped:\n{tweets(d, 0)}"
        )
    if excluded:
        parts.append("Previously retired (do not repeat):\n" + "\n".join(f"- {s}" for s in excluded))
    parts.append(
        f"Propose ONE new {dim}, clearly distinct from all of the above (not a copy or rewording "
        f"of any listed option), that the user is likely to like given what has worked. It must "
        f"be achievable in a plain-text tweet of at most 280 characters: no markdown, bold or "
        f"other rich formatting. One short sentence. Only describe the {dim}."
    )
    known = bandit.arms + excluded
    for _ in range(attempts):
        level = ask("\n\n".join(parts), Level).level.strip()
        if level and any(c.isalnum() for c in level) and not any(same(level, k) for k in known):
            return level
        logger.info("LLM proposed an empty or existing %s level, retrying", dim)
        parts.append(f"The previous proposal {level!r} was invalid or already known. Propose a different option.")
    return None
