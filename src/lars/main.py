import os

from lars import store
from lars.bandit import ThompsonBandit
from lars.evolve import propose_replacement, should_retire
from lars.prefetch import Prefetcher
from lars.rater import rate
from lars.strategies import propose_strategies

TOPIC = "learning Python"
AUDIENCE = "beginner programmers"
MAX_CONCURRENT = int(os.environ.get("LARS_MAX_CONCURRENT", 3))


def info(bandit: ThompsonBandit, log: list[dict], retired: list[str]) -> None:
    """Unblinds the experiment. Only shown via the hidden 'i' command."""
    print(f"\n  {len(log)} ratings. Posterior mean like rate per strategy:")
    for i, (s, m, a, b) in enumerate(zip(bandit.arms, bandit.means(), bandit.alpha, bandit.beta)):
        print(f"  [{i}] {m:.2f}  (n={int(a + b - 2):2d})  {s}")
    for s in retired:
        print(f"  retired: {s}")
    if log:
        print(f"  previous tweet was from: {log[-1]['strategy']}")


def main() -> None:
    bandit, log, retired = store.load() or (ThompsonBandit(propose_strategies(TOPIC, AUDIENCE)), [], [])
    prefetch = Prefetcher(bandit.arms, TOPIC, AUDIENCE, MAX_CONCURRENT)

    try:
        while True:
            arm = bandit.select()
            strategy = bandit.arms[arm]
            tweet = prefetch.take(arm)
            reward = rate(tweet, lambda: info(bandit, log, retired))
            bandit.update(arm, reward)
            log.append({"strategy": strategy, "tweet": tweet, "reward": reward})

            dead = should_retire(bandit)
            if dead is not None:
                new = propose_replacement(bandit, dead, log, retired, TOPIC, AUDIENCE)
                retired.append(bandit.arms[dead])
                bandit.replace(dead, new)
                prefetch.replace(dead, new)
            store.save(bandit, log, retired)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        prefetch.close()
        print(f"\nsaved {len(log)} ratings to {store.PATH}")


if __name__ == "__main__":
    main()
