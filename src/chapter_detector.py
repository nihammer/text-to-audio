"""Detect chapter boundaries in Vietnamese novel text."""
import re
from dataclasses import dataclass, field


CHAPTER_PATTERNS = [
    # Vietnamese: Chương 1 / CHƯƠNG I / Hồi 5
    r"^(Chương|CHƯƠNG|Hồi|HỒI)\s+[\dIVXivx]+[^\n]*",
    # English fallback
    r"^(Chapter|CHAPTER)\s+\d+[^\n]*",
    # Numeric-only headings: "1." or "I." at start of line
    r"^\d+\.[^\n]{0,60}$",
]

_COMPILED = [re.compile(p, re.MULTILINE) for p in CHAPTER_PATTERNS]


@dataclass
class Chapter:
    number: int
    title: str
    text: str


def detect_chapters(full_text: str) -> list[Chapter]:
    """
    Split *full_text* into chapters using heading patterns.

    Returns a list of Chapter objects. If no chapter headings are found,
    the entire text is returned as a single chapter titled "Toàn bộ".
    """
    # Find all heading match positions
    matches: list[re.Match] = []
    for pattern in _COMPILED:
        for m in pattern.finditer(full_text):
            # Avoid duplicates from overlapping patterns
            if not any(abs(m.start() - existing.start()) < 5 for existing in matches):
                matches.append(m)

    if not matches:
        return [Chapter(number=1, title="", text=full_text.strip())]

    # Sort by position in file
    matches.sort(key=lambda m: m.start())

    chapters: list[Chapter] = []

    # Text before first heading becomes a prologue if non-trivial
    preface = full_text[: matches[0].start()].strip()
    if len(preface) > 100:
        chapters.append(Chapter(number=0, title="Lời mở đầu", text=preface))

    for i, match in enumerate(matches):
        title = match.group(0).strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body = full_text[body_start:body_end].strip()
        chapters.append(Chapter(number=i + 1, title=title, text=body))

    return chapters
