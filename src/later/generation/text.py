import re


def strip_markdown(text: str) -> str:
    """Remove common Markdown formatting from generated post text."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # **bold** (not __x__: breaks __init__)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", text)  # *italic*
    text = re.sub(r"```\w*\n?|`", "", text)  # code fences / inline code
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.M)  # headings (hashtags have no space)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1 \2", text)
    return text.strip()


def is_emoji(c: str) -> bool:
    o = ord(c)
    return 0x1F300 <= o <= 0x1FAFF or 0x2600 <= o <= 0x27BF


def measure(tweet: str) -> dict[str, int]:
    """What the tweet actually is, to check the LLM followed the requested levels."""
    return {"chars": len(tweet), "emojis": sum(map(is_emoji, tweet))}
