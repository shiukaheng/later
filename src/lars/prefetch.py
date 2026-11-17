from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

from lars.factored import FactoredBandit
from lars.generator import generate_tweet


class Prefetcher:
    """Keeps a queue of tweets generating in the background. Each queued tweet's
    levels are chosen by Thompson sampling at submit time (a few ratings stale)."""

    def __init__(self, bandit: FactoredBandit, feed: str, max_workers: int = 3, news=None):
        self.bandit, self.feed = bandit, feed
        self.news = news
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.size = max_workers
        self.queue = deque()
        self.refill()

    def _submit(self) -> tuple[dict[str, str], Future]:
        levels = self.bandit.levels(self.bandit.select())
        stories = self.news.stories() if self.news else []
        if not stories:
            levels.pop("current_events", None)
        return levels, self.pool.submit(generate_tweet, levels, self.feed, stories)

    def take(self) -> tuple[dict[str, str], Future]:
        """Pop the next (levels, future-tweet) and queue a replacement. Non-blocking."""
        item = self.queue.popleft()
        self.queue.append(self._submit())
        return item

    def refill(self) -> None:
        """Discard queued tweets (e.g. after a level was replaced) and start over."""
        for _, future in self.queue:
            future.cancel()
        self.queue = deque(self._submit() for _ in range(self.size))

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
