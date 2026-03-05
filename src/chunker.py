"""Split chapter text into TTS-friendly chunks."""
import re
from dataclasses import dataclass
from typing import Literal


ChunkType = Literal["sentence", "paragraph_break", "chapter_boundary"]

# Sentence-ending punctuation followed by whitespace + next token
_SENTENCE_END = re.compile(
    r"(?<=[.!?…])\s+(?=[A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ])",
    re.UNICODE,
)


@dataclass
class Chunk:
    text: str
    chunk_type: ChunkType
    index: int  # position within parent chapter


def split_into_chunks(
    chapter_text: str,
    max_chars: int = 200,
    min_chars: int = 50,
) -> list[Chunk]:
    """
    Split chapter text into chunks of at most *max_chars* characters.

    Strategy:
    1. Split into paragraphs (double newline).
    2. Within each paragraph, split at sentence boundaries.
    3. Group sentences into chunks ≤ max_chars without breaking mid-sentence.
    4. Paragraph boundaries become explicit 'paragraph_break' sentinel chunks.
    5. Post-process: merge any text chunk < min_chars into its nearest
       preceding sentence chunk (prevents wasted F5-TTS calls on fragments
       like "Hiệu quả:", "……", "A").
    """
    paragraphs = chapter_text.split("\n\n")
    chunks: list[Chunk] = []
    idx = 0

    for para_idx, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue

        sentences = _split_sentences(para)
        current: list[str] = []
        current_len = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If this single sentence exceeds max_chars, hard-split it
            if len(sentence) > max_chars:
                if current:
                    chunks.append(Chunk(" ".join(current), "sentence", idx))
                    idx += 1
                    current, current_len = [], 0
                for hard_chunk in _hard_split(sentence, max_chars):
                    chunks.append(Chunk(hard_chunk, "sentence", idx))
                    idx += 1
                continue

            # Would adding this sentence exceed the limit?
            additional = len(sentence) + (1 if current else 0)  # +1 for space
            if current and current_len + additional > max_chars:
                chunks.append(Chunk(" ".join(current), "sentence", idx))
                idx += 1
                current, current_len = [], 0

            current.append(sentence)
            current_len += additional

        if current:
            chunks.append(Chunk(" ".join(current), "sentence", idx))
            idx += 1

        # Insert paragraph break sentinel between paragraphs (not after the last)
        if para_idx < len(paragraphs) - 1:
            chunks.append(Chunk("", "paragraph_break", idx))
            idx += 1

    return _merge_small_chunks(chunks, min_chars, max_chars)


def _merge_small_chunks(
    chunks: list[Chunk], min_chars: int, max_chars: int
) -> list[Chunk]:
    """Merge text chunks shorter than *min_chars* into the preceding sentence chunk.

    Paragraph-break sentinels are kept unchanged.  If a tiny chunk cannot be
    merged backward (e.g. it is the very first chunk), it is kept as-is so
    the model still receives valid input.
    """
    if min_chars <= 0:
        return chunks

    result: list[Chunk] = []
    for chunk in chunks:
        is_tiny = (
            chunk.chunk_type == "sentence"
            and chunk.text.strip()
            and len(chunk.text) < min_chars
        )
        if is_tiny and result:
            # Walk backward to find the nearest sentence chunk
            for i in range(len(result) - 1, -1, -1):
                if result[i].chunk_type == "sentence":
                    candidate = result[i].text + " " + chunk.text
                    if len(candidate) <= max_chars:
                        result[i] = Chunk(candidate, "sentence", result[i].index)
                        break  # merged — do NOT append this chunk again
            else:
                result.append(chunk)  # nothing to merge into → keep as-is
        else:
            result.append(chunk)

    return result


def _split_sentences(text: str) -> list[str]:
    """Split text at sentence boundaries (best-effort, Vietnamese-aware)."""
    parts = _SENTENCE_END.split(text)
    # Re-attach punctuation that got separated
    result = []
    for part in parts:
        part = part.strip()
        if part:
            result.append(part)
    return result


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Split oversized text at word boundaries, never exceeding max_chars."""
    words = text.split()
    parts: list[str] = []
    current: list[str] = []
    length = 0

    for word in words:
        if length + len(word) + (1 if current else 0) > max_chars and current:
            parts.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += len(word) + (1 if len(current) > 1 else 0)

    if current:
        parts.append(" ".join(current))

    return parts
