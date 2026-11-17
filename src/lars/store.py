import json
from pathlib import Path

from lars.bandit import ThompsonBandit

PATH = Path("data/session.json")


def save(bandit: ThompsonBandit, log: list[dict], retired: list[str]) -> None:
    PATH.parent.mkdir(exist_ok=True)
    state = {
        "arms": bandit.arms,
        "alpha": bandit.alpha.tolist(),
        "beta": bandit.beta.tolist(),
        "retired": retired,
        "log": log,
    }
    PATH.write_text(json.dumps(state, indent=2))


def load() -> tuple[ThompsonBandit, list[dict], list[str]] | None:
    if not PATH.exists():
        return None
    state = json.loads(PATH.read_text())
    bandit = ThompsonBandit(state["arms"])
    bandit.alpha[:] = state["alpha"]
    bandit.beta[:] = state["beta"]
    return bandit, state["log"], state["retired"]
