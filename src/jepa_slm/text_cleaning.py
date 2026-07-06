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
    chars = []
    for char in text:
        category = unicodedata.category(char)
        if category == "Cf":
            # Format characters (soft hyphen, zero-width space/joiner, BOM, ...)
            # sit INSIDE words; replacing them with a space would split the word
            # into garbage fragments ("docu­ment" -> "docu ment").
            continue
        chars.append(" " if category[0] == "C" else char)
    text = " ".join("".join(chars).split())
    if max_chars is not None and len(text) > max_chars:
        cut = text[:max_chars]
        if not text[max_chars].isspace():
            cut = cut.rsplit(" ", 1)[0] or cut
        text = cut
    # Length floor applies to the FINAL text: a max_chars cut on a document with
    # sparse spaces can otherwise emit a degenerate short sample that would have
    # been rejected had it arrived at that length.
    if len(text) < min_chars:
        return None
    return text
