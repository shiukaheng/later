import math
import random


def engagement() -> dict[str, int]:
    """Plausible, heavy-tailed counts for a tweet, independent of its content (so they add
    noise to the reward, not bias). Logged with each rating for later analysis."""
    views = max(12, int(math.exp(random.gauss(7.0, 1.6))))
    likes = int(views * random.uniform(0.002, 0.04))
    return {
        "views": views,
        "likes": likes,
        "reposts": int(likes * random.uniform(0.02, 0.25)),
        "replies": int(likes * random.uniform(0.03, 0.35)),
        "minutes_ago": int(math.exp(random.uniform(0, math.log(60 * 24 * 3)))),  # 1 min .. 3 days
    }
