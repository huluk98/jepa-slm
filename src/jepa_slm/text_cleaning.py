"""Text cleaning helpers shared by corpus prep and training."""

from __future__ import annotations

import html
import unicodedata


def clean_text(
    value: object,
    *,
    normalize: bool = True,
    min_chars: int = 0,
    max_chars: int | None = None,
) -> str | None:
    """Apply cheap corpus hygiene before tokenization."""

    if value is None:
        return None
    text = str(value)
    if normalize:
        text = unicodedata.normalize("NFKC", html.unescape(text))
    text = "".join(" " if unicodedata.category(char)[0] == "C" else char for char in text)
    text = " ".join(text.split())
    if len(text) < min_chars:
        return None
    if max_chars is not None and len(text) > max_chars:
        cut = text[:max_chars]
        if not text[max_chars].isspace():
            cut = cut.rsplit(" ", 1)[0] or cut
        text = cut
    return text
