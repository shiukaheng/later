import os

import numpy as np

# Exploration: >1 inflates posteriors (acts as if we had 1/EXPLORE as much data),
# so weaker arms get tried more often. 1 = standard Thompson sampling.
EXPLORE = float(os.environ.get("LARS_EXPLORE", 1.0))


class ThompsonBandit:
    """Beta-Bernoulli Thompson sampling over a list of arms."""

    def __init__(self, arms: list[str], seed: int | None = None):
        self.arms = list(arms)
        self.alpha = np.ones(len(arms))
        self.beta = np.ones(len(arms))
        self.rng = np.random.default_rng(seed)

    def select(self) -> int:
        a = 1 + (self.alpha - 1) / EXPLORE
        b = 1 + (self.beta - 1) / EXPLORE
        return int(np.argmax(self.rng.beta(a, b)))

    def update(self, arm: int, reward: int) -> None:
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward

    def add(self, arm: str) -> None:
        self.arms.append(arm)
        self.alpha = np.append(self.alpha, 1.0)
        self.beta = np.append(self.beta, 1.0)

    def replace(self, arm: int, strategy: str) -> None:
        self.arms[arm] = strategy
        self.alpha[arm] = self.beta[arm] = 1

    def means(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)
