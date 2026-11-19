"""Shared backend: serve a post, record a rating, and save the profile."""

import logging
import os
import threading
import time

from lars.feed import profiles
from lars.feed.prefetch import Prefetcher
from lars.generation.posts import SEEDED
from lars.learning.bandit import ThompsonBandit, same
from lars.learning.recommender import Recommendation, Recommender

FEED = os.environ.get("LARS_FEED", "a general Twitter/X timeline")
MAX_CONCURRENT = int(os.environ.get("LARS_MAX_CONCURRENT", 3))

MAX_CHARS = 280  # Twitter's limit; longer generations are discarded
MAX_RETRIES = 5  # regenerate a duplicate tweet at most this many times, then show it anyway
log = logging.getLogger("lars")


class Session:
    """The shared backend: hands out tweets, takes ratings, learns, persists.
    Thread-safe so the web server can call it from multiple requests."""

    def __init__(self, profile: str = "default"):
        self.path = profiles.profile_path(profile)
        loaded = profiles.load(self.path)
        if loaded:
            bandit, history, retired = loaded
            self.recommender = Recommender(bandit, FEED, history=history, retired=retired)
            self._migrate()
        else:
            log.info("new session: asking LLM for initial styles...")
            self.recommender = Recommender.fresh(FEED)
        log.info("session ready: %d ratings so far, %d-way prefetch", len(self.recommender.history), MAX_CONCURRENT)
        self.prefetch = Prefetcher(self.recommender, MAX_CONCURRENT)
        self.pending: dict[int, Recommendation] = {}
        self.seen = {entry["tweet"] for entry in self.recommender.history}
        self.next_id = 0
        self.lock = threading.Lock()

    def next(self) -> dict:
        """Serve generated text and assign an ID for later feedback."""
        start = time.monotonic()
        for attempt in range(MAX_RETRIES + 1):
            with self.lock:
                future = self.prefetch.take()
            try:
                item = future.result()
            except Exception as e:  # LLM error, or cancelled by a queue refill
                log.warning("generation failed (%s: %s), trying next queued tweet", type(e).__name__, e)
                continue
            tweet = item.text
            with self.lock:
                if tweet in self.seen and attempt < MAX_RETRIES:
                    log.info("skipped exact duplicate tweet")
                    continue
                if len(tweet) > MAX_CHARS:
                    log.info("skipped over-length tweet (%d chars)", len(tweet))
                    continue
                self.seen.add(tweet)
                tid, self.next_id = self.next_id, self.next_id + 1
                self.pending[tid] = item
            log.info("served #%d after %.2fs (%d on screen unrated)", tid, time.monotonic() - start, len(self.pending))
            return {"id": tid, "text": tweet}
        raise RuntimeError(f"no tweet after {MAX_RETRIES + 1} attempts")

    def rate(self, tid: int, reward: int) -> None:
        with self.lock:
            if tid not in self.pending:
                log.warning("rating for unknown tweet #%d ignored", tid)
                return
            recommender = self.recommender
            if reward not in (0, 1):
                raise ValueError("reward must be 0 or 1")
            if recommender.learn(self.pending.pop(tid), reward):
                self.prefetch.refill()
            profiles.save(recommender.bandit, recommender.history, recommender.retired, self.path)
            log.info("rated #%d %s (%d ratings)", tid, "LIKE" if reward else "skip", len(recommender.history))

    @property
    def rating_count(self) -> int:
        return len(self.recommender.history)

    def info(self) -> str:
        """Unblinds the experiment. Only shown via the hidden 'i' command."""
        with self.lock:
            return self._format_info()

    def _format_info(self) -> str:
        recommender = self.recommender
        history = recommender.history
        out = [f"{len(history)} ratings. Like rate per level (effective n, measured avg chars/emojis):"]
        for dim, b in recommender.bandit.dims.items():
            out.append(f"{dim}:")
            for level, m, a, bb in zip(b.arms, b.means(), b.alpha, b.beta):
                hits = [e for e in history if e["levels"].get(dim) == level]
                avg = lambda k: sum(e[k] for e in hits) / len(hits) if hits else 0
                out.append(f"  {m:.2f}  n={a + bb - 2:4.1f}  {avg('chars'):3.0f}ch {avg('emojis'):3.1f}em  {level}")
            out += [f"  retired: {level}" for level in recommender.retired[dim]]
        if history:
            out.append(f"previous tweet: {history[-1]['levels']}")
        return "\n".join(out)

    def close(self) -> None:
        self.prefetch.close()

    def _migrate(self) -> None:
        """Bring a saved profile in line with the current dimensions and options."""
        recommender = self.recommender
        dimensions = recommender.bandit.dims
        wanted = {"style", *SEEDED}
        for name in set(dimensions) - wanted:
            del dimensions[name]
            recommender.retired.pop(name, None)
            log.info("dropped dimension '%s'", name)
        for name, levels in SEEDED.items():
            if name not in dimensions:
                dimensions[name] = ThompsonBandit(levels)
                recommender.retired[name] = []
                log.info("added dimension '%s'", name)
        for name, bandit in dimensions.items():
            if merged := bandit.dedupe():
                for entry in recommender.history:
                    level = entry["levels"].get(name)
                    if level is not None:
                        canonical = next((arm for arm in bandit.arms if same(arm, level)), None)
                        if canonical is not None:
                            entry["levels"][name] = canonical
                log.info("merged %d duplicate %s level(s)", merged, name)
