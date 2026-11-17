import re

from pydantic import BaseModel

from lars.llm import ask


class Tweet(BaseModel):
    text: str


def strip_markdown(text: str) -> str:
    """Twitter shows raw text, so markdown would appear as literal symbols."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # **bold** (not __x__: breaks __init__)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", text)  # *italic*
    text = re.sub(r"```\w*\n?|`", "", text)  # code fences / inline code
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.M)  # headings (hashtags have no space)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1 \2", text)  # [text](url)
    return text.strip()


def generate_tweet(levels: dict[str, str], feed: str) -> str:
    """LLM writes one tweet following the chosen level of each dimension."""
    spec = "\n".join(f"- {dim}: {level}" for dim, level in levels.items())
    prompt = (
        f"Write one short, realistic tweet as it might appear on {feed}.\n"
        f"Follow every requirement strictly, but never mention or label them "
        f"(no hashtags or words naming the mood, style, etc.):\n{spec}\n"
        "Plain tweet text only. No markdown: no **bold**, *italics*, `code`, headings or "
        "[links](...). Twitter does not render markdown. Emojis and line breaks are fine."
    )
    return strip_markdown(ask(prompt, Tweet).text)
