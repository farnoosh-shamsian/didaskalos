"""Lightweight JSON-file-based localization for the Didaskalos app.

UI strings live in ``locales/<lang>.json``. Add a new language by dropping in a
new JSON file with the same keys and registering it in ``AVAILABLE_LANGS`` /
``LANG_NAMES`` below.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

LOCALES_DIR = Path(__file__).resolve().parent / "locales"

DEFAULT_LANG = "en"
# Order here is the order shown in the language selector; first entry is default.
AVAILABLE_LANGS = ["en", "fa"]
LANG_NAMES = {"en": "English", "fa": "فارسی"}
# Languages that render right-to-left.
RTL_LANGS = {"fa"}


@functools.lru_cache(maxsize=None)
def _load(lang: str) -> dict:
    path = LOCALES_DIR / f"{lang}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def t(key: str, lang: str = DEFAULT_LANG, **kwargs) -> str:
    """Return the translated string for ``key`` in ``lang``.

    Falls back to the default language, then to the key itself, so a missing
    translation degrades gracefully instead of crashing. ``kwargs`` are applied
    via ``str.format`` when present.
    """
    value = _load(lang).get(key)
    if value is None and lang != DEFAULT_LANG:
        value = _load(DEFAULT_LANG).get(key)
    if value is None:
        value = key
    if kwargs:
        try:
            value = value.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            pass
    return value


def is_rtl(lang: str) -> bool:
    return lang in RTL_LANGS


def rtl_css() -> str:
    """CSS injected when the active language is right-to-left.

    Flips the app/sidebar direction and switches to a Persian-friendly font,
    while keeping tabular data, code, and preview blocks left-to-right so Greek
    and Latin tokens stay aligned.
    """
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], .stApp, button, input, textarea, select,
    section[data-testid="stSidebar"] {
        font-family: 'Noto Naskh Arabic', 'B Lotus', 'Segoe UI', Tahoma, sans-serif;
    }
    .stApp, section[data-testid="stSidebar"] { direction: rtl; }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stApp p, .stApp li, .stApp label, .stApp .stMarkdown { text-align: right; }
    /* Keep tabular/greek/latin data and code left-to-right. */
    [data-testid="stDataFrame"], [data-testid="stTable"],
    .stCode, pre, code { direction: ltr; text-align: left; }
    </style>
    """
