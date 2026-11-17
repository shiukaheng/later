"""Shared, periodically refreshed context from The News API."""

import fcntl
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

log = logging.getLogger("lars.news")
PATH = Path("data/news.json")
INTERVAL = 1800  # 48 requests per day when running continuously
MAX_AGE = 48 * 3600
MAX_STORIES = 30
PAGE_SIZE = 3
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"


def fresh(story: dict, now: float) -> bool:
    try:
        published = datetime.fromisoformat(story["published_at"].replace("Z", "+00:00"))
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        return 0 <= now - published.timestamp() <= MAX_AGE
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def clean(story: dict, now: float) -> dict | None:
    url = story.get("url")
    title = story.get("title")
    if not isinstance(url, str) or urlsplit(url).scheme not in ("http", "https") or not isinstance(title, str) or not title.strip() or not fresh(story, now):
        return None
    return {key: str(story.get(key) or "") for key in ("title", "description", "source", "published_at", "url")}


def load_cache(path: Path) -> dict:
    try:
        state = json.loads(path.read_text())
        if isinstance(state, dict) and isinstance(state.get("stories"), list):
            return state
    except (OSError, ValueError):
        pass
    return {"stories": [], "last_attempt": 0}


class News:
    def __init__(self, token: str = "", path: Path = PATH, interval: int = INTERVAL):
        self.token, self.path, self.interval = token, path, interval
        self.stop = threading.Event()
        self.thread = None
        if token:
            self.thread = threading.Thread(target=self._run, name="news-refresh", daemon=True)
            self.thread.start()
        else:
            log.info("news disabled: set THENEWSAPI_TOKEN to enable current events")

    def stories(self) -> list[dict]:
        if not self.token:
            return []
        now = time.time()
        return [item for s in load_cache(self.path)["stories"] if isinstance(s, dict) and (item := clean(s, now))]

    def _run(self) -> None:
        while not self.stop.is_set():
            self.refresh()
            self.stop.wait(self.interval)

    def refresh(self) -> None:
        """Serialize refreshes across profiles and persist attempts, including failures."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.with_suffix(".lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                state = load_cache(self.path)
                now = time.time()
                if now - state.get("last_attempt", 0) < self.interval:
                    return
                state["last_attempt"] = now
                unique = {}
                for s in state["stories"]:
                    if isinstance(s, dict) and (item := clean(s, now)):
                        unique.setdefault(item["url"], item)
                prefill = not state.get("prefilled", False) and len(unique) < MAX_STORIES
                try:
                    pages = (MAX_STORIES + PAGE_SIZE - 1) // PAGE_SIZE if prefill else 1
                    for page in range(1, pages + 1):
                        query = urlencode({"api_token": self.token, "language": "en", "limit": PAGE_SIZE, "page": page})
                        request = Request(f"https://api.thenewsapi.com/v1/news/top?{query}", headers={"User-Agent": USER_AGENT})
                        with urlopen(request, timeout=10) as response:
                            data = json.load(response)["data"]
                        if not isinstance(data, list):
                            raise ValueError("unexpected news response")
                        for s in data:
                            if not isinstance(s, dict) or not (item := clean(s, now)):
                                continue
                            unique.setdefault(item["url"], item)
                        if len(data) < PAGE_SIZE or (prefill and len(unique) >= MAX_STORIES):
                            break
                    state["prefilled"] = True
                except Exception as exc:
                    # HTTP errors can include the request URL and API token; log only the type.
                    log.warning("news refresh failed (%s)", type(exc).__name__)
                state["stories"] = sorted(unique.values(), key=lambda s: s["published_at"], reverse=True)[:MAX_STORIES]
                log.info("news cache: %d recent articles", len(state["stories"]))
                temp = self.path.with_suffix(f".{os.getpid()}.tmp")
                temp.write_text(json.dumps(state))
                os.replace(temp, self.path)
        except (OSError, TypeError, ValueError) as exc:
            log.warning("news cache unavailable (%s)", type(exc).__name__)

    def close(self) -> None:
        self.stop.set()
