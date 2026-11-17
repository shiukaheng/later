def is_emoji(c: str) -> bool:
    o = ord(c)
    return 0x1F300 <= o <= 0x1FAFF or 0x2600 <= o <= 0x27BF


def measure(tweet: str) -> dict[str, int]:
    """What the tweet actually is, to check the LLM followed the requested levels."""
    return {"chars": len(tweet), "emojis": sum(map(is_emoji, tweet))}
