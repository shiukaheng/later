import os

from lars import store
from lars.evolve import propose_replacement, should_retire
from lars.factored import FactoredBandit
from lars.measure import measure
from lars.prefetch import Prefetcher
from lars.rater import rate
from lars.strategies import propose_strategies

TOPIC = "learning Python"
AUDIENCE = "beginner programmers"
MAX_CONCURRENT = int(os.environ.get("LARS_MAX_CONCURRENT", 3))

# Fixed dimensions. "style" levels are proposed by the LLM and evolve (EVOLVING).
LENGTH = ["short: under 100 characters", "medium: 100-200 characters", "long: 200-280 characters"]
EMOJI = ["no emojis at all", "one or two emojis", "lots of emojis (5 or more)"]
EVOLVING = ["style"]


def info(bandit: FactoredBandit, log: list[dict], retired: dict[str, list[str]]) -> None:
    """Unblinds the experiment. Only shown via the hidden 'i' command."""
    print(f"\n  {len(log)} ratings. Like rate per level (n, measured avg chars/emojis):")
    for dim, b in bandit.dims.items():
        print(f"  {dim}:")
        for level, m, a, bb in zip(b.arms, b.means(), b.alpha, b.beta):
            hits = [e for e in log if e["levels"][dim] == level]
            avg = lambda k: sum(e[k] for e in hits) / len(hits) if hits else 0
            print(f"    {m:.2f}  n={int(a + bb - 2):2d}  {avg('chars'):3.0f}ch {avg('emojis'):3.1f}em  {level}")
        for level in retired[dim]:
            print(f"    retired: {level}")
    if log:
        print(f"  previous tweet: {log[-1]['levels']}")


def main() -> None:
    loaded = store.load()
    if loaded:
        bandit, log, retired = loaded
    else:
        dims = {"style": propose_strategies(TOPIC, AUDIENCE), "length": LENGTH, "emoji": EMOJI}
        bandit, log, retired = FactoredBandit(dims), [], {d: [] for d in dims}
    prefetch = Prefetcher(bandit, TOPIC, AUDIENCE, MAX_CONCURRENT)

    try:
        while True:
            choice, levels, tweet = prefetch.take()
            reward = rate(tweet, lambda: info(bandit, log, retired))
            bandit.update(choice, reward)
            log.append({"levels": levels, "tweet": tweet, "reward": reward, **measure(tweet)})

            for dim in EVOLVING:
                b = bandit.dims[dim]
                dead = should_retire(b)
                if dead is not None:
                    new = propose_replacement(b, dim, dead, log, retired[dim], TOPIC, AUDIENCE)
                    retired[dim].append(b.arms[dead])
                    b.replace(dead, new)
                    prefetch.refill()  # queued tweets may use the retired level
            store.save(bandit, log, retired)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        prefetch.close()
        print(f"\nsaved {len(log)} ratings to {store.PATH}")


if __name__ == "__main__":
    main()
