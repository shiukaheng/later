from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

from later.learning.recommender import Recommendation, Recommender


class Prefetcher:
    """Keeps a queue of tweets generating in the background. Each queued tweet's
    levels are chosen by Thompson sampling at submit time (a few ratings stale)."""

    def __init__(self, recommender: Recommender, max_workers: int = 3):
        self.recommender = recommender
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.size = max_workers
        self.queue = deque()
        self.refill()

    def _submit(self) -> Future[Recommendation]:
        levels = self.recommender.choose()
        return self.pool.submit(self.recommender.recommend, levels)

    def take(self) -> Future[Recommendation]:
        """Pop the next (levels, future-tweet) and queue a replacement. Non-blocking."""
        item = self.queue.popleft()
        self.queue.append(self._submit())
        return item

    def refill(self) -> None:
        """Discard queued tweets (e.g. after a level was replaced) and start over."""
        for future in self.queue:
            future.cancel()
        self.queue = deque(self._submit() for _ in range(self.size))

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
