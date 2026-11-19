import os

import numpy as np

# Exploration: >1 inflates posteriors (acts as if we had 1/EXPLORE as much data),
# so weaker arms get tried more often. 1 = standard Thompson sampling.
EXPLORE = float(os.environ.get("LARS_EXPLORE", 1.0))
# Evidence halves after roughly 138 ratings at the default rate.
DISCOUNT = float(os.environ.get("LARS_DISCOUNT", 0.995))
if not 0 < DISCOUNT <= 1:
    raise ValueError("LARS_DISCOUNT must be greater than 0 and at most 1")


class FactoredBandit:
    """Select one arm per dimension and update each selected arm with the same reward."""

    def __init__(self, dims: dict[str, list[str]]):
        self.dims = {name: ThompsonBandit(levels) for name, levels in dims.items()}

    def select(self) -> dict[str, int]:
        return {name: bandit.select() for name, bandit in self.dims.items()}

    def levels(self, choice: dict[str, int]) -> dict[str, str]:
        return {name: self.dims[name].arms[index] for name, index in choice.items()}

    def update(self, levels: dict[str, str], reward: int) -> None:
        """Age all evidence, then credit options still active when the post is rated."""
        for bandit in self.dims.values():
            bandit.discount()
        for name, level in levels.items():
            bandit = self.dims[name]
            if level in bandit.arms:
                bandit.update(bandit.arms.index(level), reward)


def same(a: str, b: str) -> bool:
    """Levels that differ only in case, spacing or punctuation are the same level."""
    norm = lambda s: " ".join("".join(c for c in s.lower() if c.isalnum() or c.isspace()).split())
    return norm(a) == norm(b)


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

    def discount(self) -> None:
        """Age all evidence, preserving each arm's Beta(1, 1) prior."""
        self.alpha = 1 + DISCOUNT * (self.alpha - 1)
        self.beta = 1 + DISCOUNT * (self.beta - 1)

    def add(self, arm: str) -> bool:
        """Add a new arm with a fresh prior. Refuses duplicates: a copy of an existing arm
        would never be credited (ratings match by text) yet keep getting picked."""
        if any(same(arm, a) for a in self.arms):
            return False
        self.arms.append(arm)
        self.alpha = np.append(self.alpha, 1.0)
        self.beta = np.append(self.beta, 1.0)
        return True

    def remove(self, arm: int) -> None:
        del self.arms[arm]
        self.alpha = np.delete(self.alpha, arm)
        self.beta = np.delete(self.beta, arm)

    def dedupe(self) -> int:
        """Merge duplicate arms into their first copy (summing ratings). Returns # merged."""
        merged = 0
        for i in range(len(self.arms) - 1, 0, -1):
            first = next(j for j in range(i + 1) if same(self.arms[j], self.arms[i]))
            if first != i:
                self.alpha[first] += self.alpha[i] - 1
                self.beta[first] += self.beta[i] - 1
                self.remove(i)
                merged += 1
        return merged

    def means(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)
