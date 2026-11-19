"""Synthetic authors and engagement counts for displaying generated posts."""

import hashlib
import json
import logging
import math
import os
import random
import re
import uuid
from collections import deque
from datetime import date
from functools import cache
from pathlib import Path

from faker import Faker
from unidecode import unidecode

AUTHOR_PATH = Path("data/authors.json")
AUTHOR_COUNT = int(os.environ.get("LARS_AUTHORS", 500))
RECENT_AUTHORS = 20
log = logging.getLogger("lars.authors")


class AuthorPool:
    """Reuse local synthetic accounts without showing the same one in quick succession."""

    def __init__(self):
        if AUTHOR_PATH.exists():
            self.authors = json.loads(AUTHOR_PATH.read_text())
        else:
            self.authors = [make_author() for _ in range(AUTHOR_COUNT)]
            AUTHOR_PATH.parent.mkdir(exist_ok=True)
            AUTHOR_PATH.write_text(json.dumps(self.authors, ensure_ascii=False, indent=1))
            log.info("generated %d accounts -> %s", len(self.authors), AUTHOR_PATH)
        self.recent: deque[str] = deque(maxlen=min(RECENT_AUTHORS, len(self.authors) - 1))

    def take(self) -> dict:
        author = random.choice([a for a in self.authors if a["id"] not in self.recent])
        self.recent.append(author["id"])
        return author


def engagement() -> dict[str, int]:
    """Generate plausible counts independently of post content."""
    views = max(12, int(math.exp(random.gauss(7.0, 1.6))))
    likes = int(views * random.uniform(0.002, 0.04))
    return {
        "views": views,
        "likes": likes,
        "reposts": int(likes * random.uniform(0.02, 0.25)),
        "replies": int(likes * random.uniform(0.03, 0.35)),
        "minutes_ago": int(math.exp(random.uniform(0, math.log(60 * 24 * 3)))),
    }


# Name origin -> Faker locale. Each account draws an origin at random, so names are diverse.
ORIGINS = {
    "American": "en_US", "Canadian": "en_CA", "Australian": "en_AU", "British": "en_GB",
    "Irish": "en_IE", "New Zealander": "en_NZ", "Indian": "en_IN", "Nigerian": "en_NG",
    "Yoruba": "yo_NG", "Igbo": "ig_NG", "Hausa": "ha_NG", "Ghanaian": "tw_GH", "Zulu": "zu_ZA",
    "Swahili": "sw", "Spanish": "es_ES", "Mexican": "es_MX", "Argentine": "es_AR",
    "Colombian": "es_CO", "Brazilian": "pt_BR", "Portuguese": "pt_PT", "French": "fr_FR",
    "German": "de_DE", "Italian": "it_IT", "Dutch": "nl_NL", "Polish": "pl_PL", "Swedish": "sv_SE",
    "Norwegian": "no_NO", "Danish": "da_DK", "Finnish": "fi_FI", "Turkish": "tr_TR",
    "Czech": "cs_CZ", "Romanian": "ro_RO", "Hungarian": "hu_HU", "Greek": "el_GR",
    "Russian": "ru_RU", "Ukrainian": "uk_UA", "Azerbaijani": "az_AZ", "Georgian": "ka_GE",
    "Armenian": "hy_AM", "Arab": "ar_SA", "Iranian": "fa_IR", "Israeli": "he_IL",
    "Bangladeshi": "bn_BD", "Nepali": "ne_NP", "Indonesian": "id_ID", "Vietnamese": "vi_VN",
    "Thai": "th_TH", "Japanese": "ja_JP", "Chinese": "zh_CN", "Taiwanese": "zh_TW", "Korean": "ko_KR",
}
# Non-Latin scripts that transliterate well enough to build a handle from the name.
TRANSLITERABLE = {"el_GR", "ru_RU", "uk_UA", "ka_GE", "hy_AM", "ko_KR"}
# Non-Latin scripts that don't: shown in native script with a handle not derived from the name.
NATIVE_ONLY = {"ar_SA", "fa_IR", "he_IL", "bn_BD", "ne_NP", "th_TH"}

WORDS = ["coffee", "moon", "daily", "vibes", "writes", "eats", "art", "fit", "travels", "music",
         "games", "irl", "xo", "films", "reads", "codes", "cooks", "runs", "tv", "photo", "studio"]
EMOJI = ["🌙", "✨", "🌻", "☕", "🎧", "🌊", "🍉", "🏳️‍🌈", "🖤", "🌿", "⚽", "📚", "🎮", "🌸", "🔥", "🦋"]
# Gravatar's generated defaults (these random emails have no real Gravatar). "mp" is the
# grey silhouette, like an account that never set a photo.
AVATAR_STYLES = {"mp": 3, "identicon": 2, "retro": 2, "wavatar": 2, "monsterid": 1, "robohash": 1}


@cache
def faker(locale: str) -> Faker:
    return Faker(locale)


def ascii_word(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", unidecode(s.replace("Ə", "E").replace("ə", "e")).lower())


def person_name(locale: str, gender: str) -> tuple[str, str, str]:
    """(first, last, native display name or "") for a locale and gender."""
    f, g = faker(locale), "male" if gender == "man" else "female"
    pick = lambda kind: getattr(f, f"{kind}_{g}", getattr(f, kind))()
    if locale in ("zh_CN", "zh_TW"):  # pinyin from the actual characters
        last, first = pick("last_name"), pick("first_name")
        native = last + first if random.random() < 0.4 else ""
        return unidecode(first).strip().title().replace(" ", ""), unidecode(last).strip().title(), native
    if locale == "ja_JP":  # kanji don't transliterate; native names get a non-name handle
        if random.random() < 0.3:
            return "", "", pick("last_name") + pick("first_name")
        return getattr(f, f"first_romanized_name_{g}")(), f.last_romanized_name(), ""
    first, last = pick("first_name"), pick("last_name")
    if locale in NATIVE_ONLY:
        return "", "", f"{first} {last}"
    if locale in TRANSLITERABLE:
        native = f"{first} {last}" if random.random() < 0.5 else ""
        return unidecode(first).replace("`", "").replace("'", ""), unidecode(last).replace("`", ""), native
    return first, last, ""


def make_handle(first: str, last: str, age: int) -> str:
    year = date.today().year - age + random.choice([0, 0, -1])
    if not first:  # name not usable for a handle
        w1, w2 = random.sample(WORDS, 2)
        return random.choice([f"{w1}{random.randint(1, 999)}", f"{w1}_{w2}", f"{w1}{w2}{random.randint(1, 99)}"])[:15]
    fi, la = ascii_word(first), ascii_word(last.split()[-1]) if last else ""
    yy, w = str(year)[2:], random.choice(WORDS)
    options = [
        f"{fi}{la}", f"{fi}_{la}", f"{fi}{la[:1]}", f"{fi[:1]}{la}", f"{fi}{yy}", f"{fi}_{yy}",
        f"{fi}{la}{random.randint(1, 99)}", f"the{fi}{la}", f"{fi}{w}", f"{fi}_{w}", f"its{fi}",
        f"{fi}{year}", f"{fi}x{la}", f"{fi}{fi[-1]}{fi[-1]}", f"{la}{fi[:1]}",
        f"{first[:1].upper()}{la.capitalize()}{random.randint(10, 99999)}",  # X's default style
    ]
    h = random.choice([o for o in options if 3 <= len(o) <= 15] or [f"{fi[:9]}{yy}"])
    if len(h) < 3:
        h += str(random.randint(10, 999))
    if random.random() < 0.15 and h.islower():
        h = "".join(p.capitalize() for p in re.split(r"(_)", h))
    return h


def display_name(first: str, last: str, native: str, locale: str, gender: str) -> str:
    if native and (not first or random.random() < 0.6):
        return native
    full = f"{first} {last}"
    flag = "".join(chr(0x1F1E6 + ord(c) - 65) for c in locale[-2:]) if "_" in locale else ""
    r = random.random()
    if r < 0.45:
        return full
    if r < 0.60:
        return first
    if r < 0.70:
        return full.lower()
    if r < 0.82:
        return f"{random.choice([first, full])} {random.choice(EMOJI)}"
    if r < 0.88 and flag:
        return f"{full} {flag}"
    if r < 0.93:
        return f"{first} {last[:1]}."
    if r < 0.96:
        return first.upper()
    pronouns = "he/him" if gender == "man" else "she/her"
    return f"{first} ({pronouns if random.random() < 0.8 else 'they/them'})"


def gravatar(seed: str) -> str:
    style = random.choices(list(AVATAR_STYLES), list(AVATAR_STYLES.values()))[0]
    digest = hashlib.sha256(f"{seed}@lars.invalid".encode()).hexdigest()
    return f"https://gravatar.com/avatar/{digest}?d={style}&s=96"


def make_author() -> dict:
    """One synthetic account. No LLM, no network: just Faker and string rules."""
    gender, age = random.choice(["man", "woman"]), random.randint(18, 65)
    locale = "en_US" if random.random() < 0.25 else random.choice(list(ORIGINS.values()))
    first, last, native = person_name(locale, gender)
    aid = uuid.uuid4().hex[:12]
    return {
        "id": aid,
        "name": display_name(first, last, native, locale, gender),
        "handle": make_handle(first, last, age),
        "avatar": gravatar(aid),
        "locale": locale,
        "gender": gender,
        "age": age,
    }
