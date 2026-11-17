import json
import logging
import os
import random
from collections import deque
from pathlib import Path

from lars.author import make_author

PATH = Path("data/authors.json")
SIZE = int(os.environ.get("LARS_AUTHORS", 500))  # accounts generated on first start
RECENT = 20  # an account never appears twice within this many posts
log = logging.getLogger("lars.authors")


class AuthorPool:
    """A fixed pool of synthetic accounts, generated once (instantly, no LLM) and saved.
    Each post gets a random account: who posts is unrelated to what is posted."""

    def __init__(self):
        if PATH.exists():
            self.authors = json.loads(PATH.read_text())
        else:
            self.authors = [make_author() for _ in range(SIZE)]
            PATH.parent.mkdir(exist_ok=True)
            PATH.write_text(json.dumps(self.authors, ensure_ascii=False, indent=1))
            log.info("generated %d accounts -> %s", len(self.authors), PATH)
        self.recent: deque[str] = deque(maxlen=min(RECENT, len(self.authors) - 1))

    def take(self) -> dict:
        author = random.choice([a for a in self.authors if a["id"] not in self.recent])
        self.recent.append(author["id"])
        return author
