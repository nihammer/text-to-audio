"""Vietnamese text cleaning and normalization for TTS."""
import re
import unicodedata


# Common Vietnamese abbreviations to expand before chunking
_ABBREVIATIONS = {
    "TP.": "Thành phố",
    "T.P.": "Thành phố",
    "GS.": "Giáo sư",
    "PGS.": "Phó Giáo sư",
    "TS.": "Tiến sĩ",
    "ThS.": "Thạc sĩ",
    "BS.": "Bác sĩ",
    "KS.": "Kỹ sư",
    "NXB.": "Nhà xuất bản",
    "tr.": "trang",
    "đ.": "đồng",
    "km.": "ki-lô-mét",
    "kg.": "ki-lô-gam",
}

# Vietnamese number words
_VI_UNITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
_VI_TENS = [
    "", "mười", "hai mươi", "ba mươi", "bốn mươi", "năm mươi",
    "sáu mươi", "bảy mươi", "tám mươi", "chín mươi",
]


def _num_to_vi(n: int) -> str:
    """Convert non-negative integer to Vietnamese spoken form (up to 999,999,999)."""
    if n < 0:
        return "âm " + _num_to_vi(-n)
    if n == 0:
        return "không"
    if n <= 9:
        return _VI_UNITS[n]
    if n <= 19:
        ones = n % 10
        if ones == 0:
            return "mười"
        return "mười " + ("lăm" if ones == 5 else _VI_UNITS[ones])
    if n <= 99:
        tens, ones = divmod(n, 10)
        base = _VI_TENS[tens]
        if ones == 0:
            return base
        if ones == 1:
            return base + " mốt"
        if ones == 4:
            return base + " tư"
        if ones == 5:
            return base + " lăm"
        return base + " " + _VI_UNITS[ones]
    if n <= 999:
        hundreds, rem = divmod(n, 100)
        base = _VI_UNITS[hundreds] + " trăm"
        if rem == 0:
            return base
        if rem < 10:
            return base + " lẻ " + _VI_UNITS[rem]
        return base + " " + _num_to_vi(rem)
    if n <= 999_999:
        thousands, rem = divmod(n, 1000)
        base = _num_to_vi(thousands) + " nghìn"
        if rem == 0:
            return base
        if rem < 100:
            return base + " " + _num_to_vi(rem)
        return base + " " + _num_to_vi(rem)
    if n <= 999_999_999:
        millions, rem = divmod(n, 1_000_000)
        base = _num_to_vi(millions) + " triệu"
        if rem == 0:
            return base
        return base + " " + _num_to_vi(rem)
    return str(n)  # fallback for numbers >= 1 billion


def _expand_time(text: str) -> str:
    """Convert time formats (H:MM:SS, MM:SS) to Vietnamese spoken form."""
    def _hms(m: re.Match) -> str:
        h, mins, secs = int(m.group(1)), int(m.group(2)), int(m.group(3))
        parts = []
        if h > 0:
            parts.append(_num_to_vi(h) + " giờ")
        parts.append(_num_to_vi(mins) + " phút")
        if secs > 0 or (h == 0 and mins == 0):
            parts.append(_num_to_vi(secs) + " giây")
        return " ".join(parts)

    def _ms(m: re.Match) -> str:
        mins, secs = int(m.group(1)), int(m.group(2))
        return _num_to_vi(mins) + " phút " + _num_to_vi(secs) + " giây"

    # H:MM:SS or HH:MM:SS (must come before MM:SS pattern)
    text = re.sub(r"\b(\d{1,2}):(\d{2}):(\d{2})\b", _hms, text)
    # MM:SS
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", _ms, text)
    return text


def _expand_standalone_numbers(text: str) -> str:
    """Convert standalone integers (optionally with comma thousand-separators) to Vietnamese words."""
    def _replace(m: re.Match) -> str:
        raw = m.group(0).replace(",", "")
        try:
            n = int(raw)
            return _num_to_vi(n)
        except ValueError:
            return m.group(0)

    # Match numbers with optional comma separators (e.g. 1,000,000) or plain integers
    return re.sub(r"\b\d{1,3}(?:,\d{3})+\b|\b\d+\b", _replace, text)


def normalize(text: str) -> str:
    """Full pipeline: NFC normalize → expand abbrevs → clean noise → collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = _expand_abbreviations(text)
    text = _clean_noise(text)
    text = _collapse_whitespace(text)
    return text.strip()


def _expand_abbreviations(text: str) -> str:
    for abbr, expansion in _ABBREVIATIONS.items():
        # Only expand when followed by a space or end of string (not inside another word)
        escaped = re.escape(abbr)
        text = re.sub(rf"(?<!\w){escaped}(?=\s|$)", expansion, text)
    return text


def _clean_noise(text: str) -> str:
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove CJK/special bracket pairs used in web novels 【】〔〕「」『』《》〈〉
    text = re.sub(r"[【】〔〕「」『』《》〈〉]", "", text)
    # Replace em-dashes and en-dashes (——, —, –) with a comma pause
    text = re.sub(r"\s*[—–]+\s*", ", ", text)
    # Replace multiple ASCII dashes / hyphens used as dividers with a period
    text = re.sub(r"-{3,}", ".", text)
    text = re.sub(r"_{3,}", ".", text)
    text = re.sub(r"\*{3,}", ".", text)
    # Normalize ellipsis variants
    text = re.sub(r"\.{4,}", "…", text)
    text = re.sub(r"\.{3}", "…", text)
    # Remove zero-width chars and other invisible unicode
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", text)
    # Normalize curly/typographic quotes to straight ASCII quotes
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    # Convert time formats before general number expansion
    text = _expand_time(text)
    # Convert standalone integers to Vietnamese words
    text = _expand_standalone_numbers(text)
    return text


def _collapse_whitespace(text: str) -> str:
    # Replace multiple spaces/tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)
    # Normalise multiple blank lines to at most two newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def clean_chapter_text(text: str) -> str:
    """Apply normalization to a chapter's body text, preserving paragraph breaks."""
    paragraphs = text.split("\n\n")
    cleaned = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para = normalize(para)
        if para:
            cleaned.append(para)
    return "\n\n".join(cleaned)
