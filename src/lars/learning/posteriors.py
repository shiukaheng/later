"""Recorded bandit state and estimates derived from it."""

from typing import NotRequired, TypedDict

import numpy as np
from scipy.special import betaincinv

from lars.learning.bandit import FactoredBandit

type Counts = dict[str, dict[str, list[float]]]


class Checkpoint(TypedDict):
    rating: int
    counts: NotRequired[Counts]
    kind: NotRequired[str]
    at: NotRequired[str]
    details: NotRequired[dict]


def capture(bandit: FactoredBandit, rating: int) -> Checkpoint:
    return {"rating": rating, "counts": {
        dimension: {arm: [float(a), float(b)] for arm, a, b in zip(state.arms, state.alpha, state.beta)}
        for dimension, state in bandit.dims.items()
    }}


def posterior_trends(checkpoints: list[Checkpoint]) -> dict:
    """Group past states by option and report central 95% Beta credible intervals."""
    series: dict[str, dict[str, list[tuple[int, int, list[float]]]]] = {}
    for event, checkpoint in enumerate(checkpoints):
        for dimension, options in checkpoint.get("counts", {}).items():
            trends = series.setdefault(dimension, {})
            for text, counts in options.items():
                points = trends.setdefault(text, [])
                rating = checkpoint["rating"]
                if not points or (rating, counts) != (points[-1][0], points[-1][2]):
                    points.append((rating, event, counts))

    def summarize(points: list[tuple[int, int, list[float]]]) -> list[dict]:
        alpha, beta = np.asarray([counts for _, _, counts in points]).T
        bounds = betaincinv(alpha[:, None], beta[:, None], [0.025, 0.975]).tolist()
        return [
            {"rating": rating, "event": event, "estimate": mean, "credible_interval": interval, "effective_n": n}
            for (rating, event, _), mean, interval, n in zip(
                points, (alpha / (alpha + beta)).tolist(), bounds, (alpha + beta - 2).tolist(),
            )
        ]

    return {dimension: {text: summarize(points) for text, points in options.items()}
            for dimension, options in series.items()}


def ledger_events(checkpoints: list[Checkpoint]) -> list[dict]:
    """Expose event order and lifecycle changes, including those visible in old checkpoints."""
    active: set[tuple[str, str]] = set()
    events = []
    for event, entry in enumerate(checkpoints):
        added, removed = [], []
        if "counts" in entry:
            current = {(dim, text) for dim, options in entry["counts"].items() for text in options}
            added = sorted(current - active)
            removed = sorted(active - current)
            active = current
        events.append({
            "event": event, "rating": entry["rating"], "kind": entry.get("kind", "checkpoint"),
            "at": entry.get("at"), "details": entry.get("details", {}), "added": added, "removed": removed,
        })
    return events
