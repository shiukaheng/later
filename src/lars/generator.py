import re

from pydantic import BaseModel

from lars.llm import NewsTool, ask


class Tweet(BaseModel):
    text: str
    source_index: int | None = None


def strip_markdown(text: str) -> str:
    """Twitter shows raw text, so markdown would appear as literal symbols."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # **bold** (not __x__: breaks __init__)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", text)  # *italic*
    text = re.sub(r"```\w*\n?|`", "", text)  # code fences / inline code
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.M)  # headings (hashtags have no space)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1 \2", text)  # [text](url)
    return text.strip()


def generate_tweet(levels: dict[str, str], feed: str, stories: list[dict] | None = None) -> tuple[str, dict | None]:
    """LLM writes one tweet following the chosen level of each dimension."""
    spec = "\n".join(f"- {dim}: {level}" for dim, level in levels.items())
    prompt = (
        f"Write one short, realistic tweet (at most 280 characters) as it might appear on {feed}.\n"
        f"Follow every requirement strictly, but never mention or label them "
        f"(do not name the mood, style, etc., in words or hashtags):\n{spec}\n"
        "Plain tweet text only. No markdown: no **bold**, *italics*, `code`, headings or "
        "[links](...). Twitter does not render markdown. Emojis and line breaks are fine."
    )
    stories = stories or []
    news = NewsTool(stories) if stories else None
    if stories:
        prompt += (
            "\nYou may call read_news_cache if it is useful for the selected requirements. "
            "If using a story, use only facts supported by its description, don't invent details, "
            "and return its source_index from the tool. Otherwise return null for source_index. "
            "Do not invent a news event."
        )
    result = ask(prompt, Tweet, news=news) if news else ask(prompt, Tweet)
    index = result.source_index
    source = stories[index] if news and news.used and index is not None and 0 <= index < len(stories) else None
    return strip_markdown(result.text), source
