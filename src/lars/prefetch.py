from concurrent.futures import ThreadPoolExecutor

from lars.generator import generate_tweet


class Prefetcher:
    """Keeps one ready tweet per arm, generated in the background."""

    def __init__(self, arms: list[str], topic: str, audience: str, max_workers: int = 3):
        self.arms, self.topic, self.audience = list(arms), topic, audience
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.ready = [self._submit(i) for i in range(len(arms))]

    def _submit(self, arm: int):
        return self.pool.submit(generate_tweet, self.arms[arm], self.topic, self.audience)

    def take(self, arm: int) -> str:
        tweet = self.ready[arm].result()
        self.ready[arm] = self._submit(arm)
        return tweet

    def replace(self, arm: int, strategy: str) -> None:
        self.ready[arm].cancel()
        self.arms[arm] = strategy
        self.ready[arm] = self._submit(arm)

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
