import json
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

from lars.learning.bandit import FactoredBandit
from lars.learning.posteriors import Checkpoint

PATH = Path("data/session.json")


def profile_path(profile: str = "default") -> Path:
    """Keep the original session as the default; named profiles have separate files."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", profile):
        raise ValueError("profile must start with a letter or number and contain only letters, numbers, underscores, or hyphens")
    return PATH if profile == "default" else PATH.parent / "profiles" / f"{profile}.json"


def save(bandit: FactoredBandit, log: list[dict], retired: dict[str, list[str]], path: Path = PATH,
         *, checkpoints: list[Checkpoint] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dims = {
        name: {"levels": b.arms, "alpha": b.alpha.tolist(), "beta": b.beta.tolist()}
        for name, b in bandit.dims.items()
    }
    state = {"dims": dims, "retired": retired, "log": log,
             "ledger_version": 1, "ledger": checkpoints or []}
    with NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as output:
        temp_path = Path(output.name)
        try:
            json.dump(state, output)
            output.close()
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)


def load(path: Path = PATH) -> tuple[FactoredBandit, list[dict], dict[str, list[str]], list[Checkpoint]] | None:
    if not path.exists():
        return None
    state = json.loads(path.read_text())
    bandit = FactoredBandit({name: d["levels"] for name, d in state["dims"].items()})
    for name, d in state["dims"].items():
        bandit.dims[name].alpha[:] = d["alpha"]
        bandit.dims[name].beta[:] = d["beta"]
    return bandit, state["log"], state["retired"], state.get("ledger", state.get("checkpoints", []))
