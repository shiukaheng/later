import json
from pathlib import Path

from lars.factored import FactoredBandit

PATH = Path("data/session.json")


def save(bandit: FactoredBandit, log: list[dict], retired: dict[str, list[str]]) -> None:
    PATH.parent.mkdir(exist_ok=True)
    dims = {
        name: {"levels": b.arms, "alpha": b.alpha.tolist(), "beta": b.beta.tolist()}
        for name, b in bandit.dims.items()
    }
    PATH.write_text(json.dumps({"dims": dims, "retired": retired, "log": log}, indent=2))


def load() -> tuple[FactoredBandit, list[dict], dict[str, list[str]]] | None:
    if not PATH.exists():
        return None
    state = json.loads(PATH.read_text())
    bandit = FactoredBandit({name: d["levels"] for name, d in state["dims"].items()})
    for name, d in state["dims"].items():
        bandit.dims[name].alpha[:] = d["alpha"]
        bandit.dims[name].beta[:] = d["beta"]
    return bandit, state["log"], state["retired"]
