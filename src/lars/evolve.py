import numpy as np
from pydantic import BaseModel

from lars.bandit import ThompsonBandit
from lars.llm import ask

THRESHOLD = 0.95  # retire if P(level worse than best) exceeds this
MIN_N = 5  # ...and the level has at least this many ratings
EXAMPLES = 3  # tweets of each kind shown to the LLM


def should_retire(bandit: ThompsonBandit, samples: int = 4000) -> int | None:
    """Return the level most confidently worse than the best level, if any."""
    draws = bandit.rng.beta(bandit.alpha, bandit.beta, size=(samples, len(bandit.arms)))
    best = int(np.argmax(bandit.means()))
    p_worse = (draws < draws[:, [best]]).mean(axis=0)
    n = bandit.alpha + bandit.beta - 2
    ok = [i for i in range(len(p_worse)) if n[i] >= MIN_N and p_worse[i] > THRESHOLD]
    return max(ok, key=lambda i: p_worse[i]) if ok else None


class Level(BaseModel):
    level: str


def propose_replacement(
    bandit: ThompsonBandit,
    dim: str,
    arm: int,
    log: list[dict],
    retired: list[str],
    topic: str,
    audience: str,
) -> str:
    """Ask the LLM for one new level of dimension `dim`, given what worked and what didn't."""
    best = bandit.arms[int(np.argmax(bandit.means()))]
    dead = bandit.arms[arm]

    def tweets(level: str, reward: int) -> str:
        hits = [e["tweet"] for e in log if e["levels"][dim] == level and e["reward"] == reward]
        return "\n".join(f"- {t}" for t in hits[-EXAMPLES:]) or "- (none)"

    stats = "\n".join(
        f"- {s} (like rate {m:.2f}, n={int(a + b - 2)})"
        for s, m, a, b in zip(bandit.arms, bandit.means(), bandit.alpha, bandit.beta)
    )
    prompt = (
        f"We write tweets about {topic} for {audience}. A user likes or skips each tweet.\n"
        f"Tweets vary along several dimensions; here we tune the '{dim}' dimension.\n\n"
        f"Current {dim} options:\n{stats}\n\n"
        f"Retiring this underperforming {dim}: {dead}\n"
        f"Tweets with it the user skipped:\n{tweets(dead, 0)}\n"
        f"Tweets with it the user liked:\n{tweets(dead, 1)}\n\n"
        f"Best {dim}: {best}\nTweets with it the user liked:\n{tweets(best, 1)}\n\n"
        f"Previously retired (do not repeat):\n"
        + "\n".join(f"- {s}" for s in retired + [dead])
        + f"\n\nPropose ONE new {dim}, distinct from all the above, likely to be liked. "
        f"One short sentence. Only describe the {dim}, not other dimensions."
    )
    return ask(prompt, Level).level
