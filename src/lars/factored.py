from lars.bandit import ThompsonBandit


class FactoredBandit:
    """One independent Thompson bandit per tweet dimension (style, length, ...).
    Every reward updates the chosen level of every dimension."""

    def __init__(self, dims: dict[str, list[str]]):
        self.dims = {name: ThompsonBandit(levels) for name, levels in dims.items()}

    def select(self) -> dict[str, int]:
        return {name: b.select() for name, b in self.dims.items()}

    def levels(self, choice: dict[str, int]) -> dict[str, str]:
        return {name: self.dims[name].arms[i] for name, i in choice.items()}

    def update(self, choice: dict[str, int], reward: int) -> None:
        for name, i in choice.items():
            self.dims[name].update(i, reward)
