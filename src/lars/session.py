import logging
import os
import threading
import time

from lars import store
from lars.evolve import propose_level, should_propose, should_retire
from lars.factored import FactoredBandit
from lars.measure import measure
from lars.prefetch import Prefetcher
from lars.strategies import propose_strategies

TOPIC = "learning Python"
AUDIENCE = "beginner programmers"
MAX_CONCURRENT = int(os.environ.get("LARS_MAX_CONCURRENT", 3))

# Dimensions. "style" starts LLM-proposed; MOOD starts from these seeds. Dimensions in
# EVOLVING get levels retired/replaced and new ones proposed; length and emoji stay fixed.
LENGTH = ["tiny: under 50 characters", "short: 50-100 characters", "medium: 100-160 characters"]
EMOJI = ["no emojis at all", "one or two emojis", "lots of emojis (5 or more)"]
MOOD = [
    "happy and upbeat",
    "ragebait: a provocative take designed to make people argue",
    "rant: frustrated and venting",
    "calm and matter-of-fact",
]
EVOLVING = ["style", "mood"]
MAX_RETRIES = 5  # regenerate a duplicate tweet at most this many times, then show it anyway
log = logging.getLogger("lars")


class Session:
    """The shared backend: hands out tweets, takes ratings, learns, persists.
    Thread-safe so the web server can call it from multiple requests."""

    def __init__(self):
        loaded = store.load()
        if loaded:
            self.bandit, self.log, self.retired = loaded
        else:
            log.info("new session: asking LLM for initial styles...")
            dims = {"style": propose_strategies(TOPIC, AUDIENCE), "length": LENGTH, "emoji": EMOJI, "mood": MOOD}
            self.bandit, self.log, self.retired = FactoredBandit(dims), [], {d: [] for d in dims}
        log.info("session ready: %d ratings so far, %d-way prefetch", len(self.log), MAX_CONCURRENT)
        self.prefetch = Prefetcher(self.bandit, TOPIC, AUDIENCE, MAX_CONCURRENT)
        self.pending: dict[int, tuple[dict[str, str], str]] = {}
        self.seen = {e["tweet"] for e in self.log}  # every tweet ever shown
        self.next_id = 0
        self.lock = threading.Lock()

    def next(self) -> tuple[int, str]:
        """Return (id, tweet), skipping exact duplicates of any tweet already shown.
        Blocks only if the queued tweet is still generating."""
        start = time.monotonic()
        for attempt in range(MAX_RETRIES + 1):
            with self.lock:
                levels, future = self.prefetch.take()
            try:
                tweet = future.result()
            except Exception as e:  # LLM error, or cancelled by a queue refill
                log.warning("generation failed (%s: %s), trying next queued tweet", type(e).__name__, e)
                continue
            with self.lock:
                if tweet in self.seen and attempt < MAX_RETRIES:
                    log.info("skipped exact duplicate tweet")
                    continue
                self.seen.add(tweet)
                tid, self.next_id = self.next_id, self.next_id + 1
                self.pending[tid] = (levels, tweet)
            log.info("served #%d after %.2fs (%d on screen unrated)", tid, time.monotonic() - start, len(self.pending))
            return tid, tweet
        raise RuntimeError(f"no tweet after {MAX_RETRIES + 1} attempts")

    def rate(self, tid: int, reward: int) -> None:
        with self.lock:
            if tid not in self.pending:
                log.warning("rating for unknown tweet #%d ignored", tid)
                return
            levels, tweet = self.pending.pop(tid)
            self.bandit.update(levels, reward)
            self.log.append({"levels": levels, "tweet": tweet, "reward": reward, **measure(tweet)})
            likes = sum(e["reward"] for e in self.log)
            log.info("rated #%d %s (%d ratings, %d likes)", tid, "LIKE" if reward else "skip", len(self.log), likes)

            for dim in EVOLVING:
                b = self.bandit.dims[dim]
                dead = should_retire(b)
                if dead is not None:
                    log.info("retiring a %s level, asking LLM for a replacement...", dim)
                    new = propose_level(b, dim, self.log, self.retired[dim], TOPIC, AUDIENCE, dead)
                    self.retired[dim].append(b.arms[dead])
                    b.replace(dead, new)
                    self.prefetch.refill()  # queued tweets may use the retired level
                    log.info("replaced; tweet queue refilled")
                if should_propose(b, len(self.log)):
                    log.info("proposing a new %s level...", dim)
                    b.add(propose_level(b, dim, self.log, self.retired[dim], TOPIC, AUDIENCE))
                    log.info("now %d %s levels", len(b.arms), dim)
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
