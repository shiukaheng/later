import numpy as np


class ThompsonBandit:
    """Beta-Bernoulli Thompson sampling over a list of arms."""

    def __init__(self, arms: list[str], seed: int | None = None):
        self.arms = list(arms)
        self.alpha = np.ones(len(arms))
        self.beta = np.ones(len(arms))
        self.rng = np.random.default_rng(seed)

    def select(self) -> int:
        return int(np.argmax(self.rng.beta(self.alpha, self.beta)))

    def update(self, arm: int, reward: int) -> None:
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward

    def replace(self, arm: int, strategy: str) -> None:
        self.arms[arm] = strategy
        self.alpha[arm] = self.beta[arm] = 1

    def means(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)
