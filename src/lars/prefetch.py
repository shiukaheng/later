from collections import deque
from concurrent.futures import ThreadPoolExecutor

from lars.factored import FactoredBandit
from lars.generator import generate_tweet


class Prefetcher:
    """Keeps a queue of tweets generating in the background. Each queued tweet's
    levels are chosen by Thompson sampling at submit time (a few ratings stale)."""

    def __init__(self, bandit: FactoredBandit, topic: str, audience: str, max_workers: int = 3):
        self.bandit, self.topic, self.audience = bandit, topic, audience
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.size = max_workers
        self.queue = deque()
        self.refill()

    def _submit(self):
        choice = self.bandit.select()
        levels = self.bandit.levels(choice)
        return choice, levels, self.pool.submit(generate_tweet, levels, self.topic, self.audience)

    def take(self) -> tuple[dict[str, int], dict[str, str], str]:
        choice, levels, future = self.queue.popleft()
        self.queue.append(self._submit())
        return choice, levels, future.result()

    def refill(self) -> None:
        """Discard queued tweets (e.g. after a level was replaced) and start over."""
        for *_, future in self.queue:
            future.cancel()
        self.queue = deque(self._submit() for _ in range(self.size))

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
