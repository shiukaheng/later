import os
import threading

from lars import store
from lars.evolve import propose_replacement, should_retire
from lars.factored import FactoredBandit
from lars.measure import measure
from lars.prefetch import Prefetcher
from lars.strategies import propose_strategies

TOPIC = "learning Python"
AUDIENCE = "beginner programmers"
MAX_CONCURRENT = int(os.environ.get("LARS_MAX_CONCURRENT", 3))

# Fixed dimensions. "style" levels are proposed by the LLM and evolve (EVOLVING).
LENGTH = ["tiny: under 50 characters", "short: 50-100 characters", "medium: 100-160 characters"]
EMOJI = ["no emojis at all", "one or two emojis", "lots of emojis (5 or more)"]
MOOD = [
    "happy and upbeat",
    "ragebait: a provocative take designed to make people argue",
    "rant: frustrated and venting",
    "calm and matter-of-fact",
]
EVOLVING = ["style"]


class Session:
    """The shared backend: hands out tweets, takes ratings, learns, persists.
    Thread-safe so the web server can call it from multiple requests."""

    def __init__(self):
        loaded = store.load()
        if loaded:
            self.bandit, self.log, self.retired = loaded
        else:
            dims = {"style": propose_strategies(TOPIC, AUDIENCE), "length": LENGTH, "emoji": EMOJI, "mood": MOOD}
            self.bandit, self.log, self.retired = FactoredBandit(dims), [], {d: [] for d in dims}
        self.prefetch = Prefetcher(self.bandit, TOPIC, AUDIENCE, MAX_CONCURRENT)
        self.pending: dict[int, tuple[dict[str, str], str]] = {}
        self.next_id = 0
        self.lock = threading.Lock()

    def next(self) -> tuple[int, str]:
        """Return (id, tweet). Blocks only if the queued tweet is still generating."""
        with self.lock:
            levels, future = self.prefetch.take()
            tid, self.next_id = self.next_id, self.next_id + 1
        tweet = future.result()
        with self.lock:
            self.pending[tid] = (levels, tweet)
        return tid, tweet

    def rate(self, tid: int, reward: int) -> None:
        with self.lock:
            if tid not in self.pending:
                return
            levels, tweet = self.pending.pop(tid)
            self.bandit.update(levels, reward)
            self.log.append({"levels": levels, "tweet": tweet, "reward": reward, **measure(tweet)})

            for dim in EVOLVING:
                b = self.bandit.dims[dim]
                dead = should_retire(b)
                if dead is not None:
                    new = propose_replacement(b, dim, dead, self.log, self.retired[dim], TOPIC, AUDIENCE)
                    self.retired[dim].append(b.arms[dead])
                    b.replace(dead, new)
                    self.prefetch.refill()  # queued tweets may use the retired level
            store.save(self.bandit, self.log, self.retired)

    def info(self) -> str:
        """Unblinds the experiment. Only shown via the hidden 'i' command."""
        out = [f"{len(self.log)} ratings. Like rate per level (n, measured avg chars/emojis):"]
        for dim, b in self.bandit.dims.items():
            out.append(f"{dim}:")
            for level, m, a, bb in zip(b.arms, b.means(), b.alpha, b.beta):
                hits = [e for e in self.log if e["levels"][dim] == level]
                avg = lambda k: sum(e[k] for e in hits) / len(hits) if hits else 0
                out.append(f"  {m:.2f}  n={int(a + bb - 2):2d}  {avg('chars'):3.0f}ch {avg('emojis'):3.1f}em  {level}")
            out += [f"  retired: {level}" for level in self.retired[dim]]
        if self.log:
            out.append(f"previous tweet: {self.log[-1]['levels']}")
        return "\n".join(out)

    def close(self) -> None:
        self.prefetch.close()
