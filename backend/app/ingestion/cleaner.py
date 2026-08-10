"""Text cleaning stage.

Conservative, deterministic cleanup for extracted PDF text: normalize
whitespace and drop excess blank lines while keeping meaningful paragraph
boundaries and page information. This is a light touch — it never rewrites or
summarizes the document, and callers clean one page at a time so page numbers
survive untouched.
"""

import re


def clean_text(text: str) -> str:
    """Return a cleaned copy of ``text``.

    - unifies line endings and strips null bytes / non-breaking spaces
    - trims trailing whitespace per line
    - collapses 3+ blank lines into a single blank line (paragraphs survive)
    - collapses runs of 3+ spaces (keeps two-space indentation)
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = text.replace("\t", " ").replace("\u00a0", " ")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]{3,}", " ", text)
    return text.strip()
