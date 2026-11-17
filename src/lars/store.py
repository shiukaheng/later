import json
import re
from pathlib import Path

from lars.factored import FactoredBandit

PATH = Path("data/session.json")


def profile_path(profile: str = "default") -> Path:
    """Keep the original session as the default; named profiles have separate files."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", profile):
        raise ValueError("profile must start with a letter or number and contain only letters, numbers, underscores, or hyphens")
    return PATH if profile == "default" else PATH.parent / "profiles" / f"{profile}.json"


def save(bandit: FactoredBandit, log: list[dict], retired: dict[str, list[str]], path: Path = PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dims = {
        name: {"levels": b.arms, "alpha": b.alpha.tolist(), "beta": b.beta.tolist()}
        for name, b in bandit.dims.items()
    }
    path.write_text(json.dumps({"dims": dims, "retired": retired, "log": log}, indent=2))


def load(path: Path = PATH) -> tuple[FactoredBandit, list[dict], dict[str, list[str]]] | None:
    if not path.exists():
        return None
    state = json.loads(path.read_text())
    bandit = FactoredBandit({name: d["levels"] for name, d in state["dims"].items()})
    for name, d in state["dims"].items():
        bandit.dims[name].alpha[:] = d["alpha"]
        bandit.dims[name].beta[:] = d["beta"]
    return bandit, state["log"], state["retired"]
