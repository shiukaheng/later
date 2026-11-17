import os

import numpy as np
from pydantic import BaseModel

from lars.bandit import ThompsonBandit
from lars.llm import ask

THRESHOLD = 0.95  # retire if P(level worse than best) exceeds this
MIN_N = 5  # ...and the level has at least this many ratings
EXAMPLES = 3  # tweets of each kind shown to the LLM
PROPOSE_EVERY = int(os.environ.get("LARS_PROPOSE_EVERY", 15))  # add a new level every N ratings
MAX_LEVELS = int(os.environ.get("LARS_MAX_LEVELS", 8))  # ...unless the dimension already has this many


def should_retire(bandit: ThompsonBandit, samples: int = 4000) -> int | None:
    """Return the level most confidently worse than the best level, if any."""
    draws = bandit.rng.beta(bandit.alpha, bandit.beta, size=(samples, len(bandit.arms)))
    best = int(np.argmax(bandit.means()))
    p_worse = (draws < draws[:, [best]]).mean(axis=0)
    n = bandit.alpha + bandit.beta - 2
    ok = [i for i in range(len(p_worse)) if n[i] >= MIN_N and p_worse[i] > THRESHOLD]
    return max(ok, key=lambda i: p_worse[i]) if ok else None


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
) -> str:
    """Ask the LLM for one new level of `dim`, conditioned on the existing levels,
    their like rates, example tweets, and everything retired. If `dead` is given,
    the new level replaces that one."""
    best = bandit.arms[int(np.argmax(bandit.means()))]

    def tweets(level: str, reward: int) -> str:
        hits = [e["tweet"] for e in log if e["levels"].get(dim) == level and e["reward"] == reward]
        return "\n".join(f"- {t}" for t in hits[-EXAMPLES:]) or "- (none)"

    stats = "\n".join(
        f"- {s} (like rate {m:.2f}, n={int(a + b - 2)})"
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
        f"Propose ONE new {dim}, clearly distinct from all of the above, that the user is "
        f"likely to like given what has worked. One short sentence. Only describe the {dim}."
    )
    return ask("\n\n".join(parts), Level).level
