"""Piper TTS wrapper (offline VITS-based, no voice cloning required).

Vietnamese models: https://huggingface.co/rhasspy/piper-voices
  vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx              (+ .onnx.json)
  vi/vi_VN/25hours_single/low/vi_VN-25hours_single-low.onnx        (+ .onnx.json)
  vi/vi_VN/vivos/x_low/vi_VN-vivos-x_low.onnx  (65 speakers)      (+ .onnx.json)

--- Foreign name handling (3-model mode) ---
When foreign_model_path_en / foreign_model_path_ja are configured, the engine
splits text by foreign-name segments and routes each segment to the correct
language model:

  Vietnamese text   → main Piper Vietnamese model
  English names     → Piper English model  (e.g. en_US-lessac-medium)
  Japanese names    → Piper Japanese model (e.g. ja_JP-kokoro-medium)

Language detection heuristic: a capitalised ASCII word sequence is classified
as Japanese romaji if it contains no non-Japanese consonant clusters (e.g. "th",
"br", "ght").  Everything else is treated as English.

--- Fallback (IPA injection) ---
When no foreign models are configured, the engine falls back to injecting
[[IPA]] phonemes via Piper's raw-phoneme syntax.  Japanese names use the
built-in Hepburn romaji→IPA converter; English names use en-us espeak.

Users can override individual words via pronunciation_dict in config.yaml.
"""
import re
import numpy as np


# ---------------------------------------------------------------------------
# Foreign-name detection regex
# ---------------------------------------------------------------------------

# Matches one or more consecutive capitalised ASCII-only words (min 4 chars each).
# Vietnamese words almost always contain diacritical marks; those that don't
# (e.g. "Trong", "Theo") are excluded via _VI_PLAIN_WORDS below.
#
# \b is Unicode-aware in Python: it fails between two \w chars, so it will
# NOT match the ASCII prefix of a diacritical word — e.g. in "Chính" the
# boundary between 'h' and 'í' (both Unicode \w) is not a \b, so "Ch" is
# never captured.
#
# Examples matched:  "Takagi Hiroshi", "Thor", "Sano Ichiro"
# Examples not matched: "Chính" (í blocks \b), "Không" (ô blocks \b),
#                       "Anh" (only 3 chars), "Trong" (in exclusion list)
_FOREIGN_PHRASE_RE = re.compile(r'[A-Z][a-zA-Z]{3,}\b(?:\s+[A-Z][a-zA-Z]{3,}\b)*')

# Common Vietnamese words that are pure ASCII (no diacritics) and may appear
# capitalised at the start of a sentence.  Exclude them from foreign detection.
_VI_PLAIN_WORDS = frozenset({
    "Trong", "Theo", "Tren", "Toan", "Tung", "Dong",
    "Minh", "Hung", "Dung", "Tuan", "Quang", "Long", "Phong",
    "Nguyen", "Ngan", "Bang",
})

# ---------------------------------------------------------------------------
# Hepburn romanization → IPA conversion (fallback when no Japanese model)
# ---------------------------------------------------------------------------

_ROMAJI_TO_IPA: dict[str, str] = {
    # ---- 3-char clusters ----
    "sha": "ʃa",  "shi": "ʃi",  "shu": "ʃɯ", "she": "ʃe", "sho": "ʃo",
    "chi": "tʃi", "cha": "tʃa", "chu": "tʃɯ","che": "tʃe","cho": "tʃo",
    "tsu": "tsɯ",
    "kya": "kja", "kyu": "kjɯ","kyo": "kjo",
    "gya": "ɡja", "gyu": "ɡjɯ","gyo": "ɡjo",
    "hya": "hja", "hyu": "hjɯ","hyo": "hjo",
    "bya": "bja", "byu": "bjɯ","byo": "bjo",
    "pya": "pja", "pyu": "pjɯ","pyo": "pjo",
    "mya": "mja", "myu": "mjɯ","myo": "mjo",
    "nya": "nja", "nyu": "njɯ","nyo": "njo",
    "rya": "ɾja", "ryu": "ɾjɯ","ryo": "ɾjo",
    # ---- 2-char clusters ----
    "ka": "ka",  "ki": "ki",  "ku": "kɯ", "ke": "ke", "ko": "ko",
    "ga": "ɡa",  "gi": "ɡi",  "gu": "ɡɯ", "ge": "ɡe", "go": "ɡo",
    "sa": "sa",  "si": "ʃi",  "su": "sɯ", "se": "se", "so": "so",
    "za": "za",  "zi": "dʑi", "zu": "zɯ", "ze": "ze", "zo": "zo",
    "ta": "ta",  "ti": "tʃi", "te": "te", "to": "to",
    "da": "da",  "di": "dʑi","du": "dzɯ","de": "de", "do": "do",
    "na": "na",  "ni": "ni",  "nu": "nɯ", "ne": "ne", "no": "no",
    "ha": "ha",  "hi": "hi",  "hu": "ɸɯ", "fu": "ɸɯ","he": "he", "ho": "ho",
    "ba": "ba",  "bi": "bi",  "bu": "bɯ", "be": "be", "bo": "bo",
    "pa": "pa",  "pi": "pi",  "pu": "pɯ", "pe": "pe", "po": "po",
    "ma": "ma",  "mi": "mi",  "mu": "mɯ", "me": "me", "mo": "mo",
    "ya": "ja",  "yu": "jɯ", "yo": "jo",
    "ra": "ɾa",  "ri": "ɾi",  "ru": "ɾɯ", "re": "ɾe", "ro": "ɾo",
    "wa": "wa",  "wi": "wi",  "we": "we", "wo": "o",
    "ji": "dʑi",
    # ---- bare vowels ----
    "a": "a", "i": "i", "u": "ɯ", "e": "e", "o": "o",
    # ---- syllabic n ----
    "n": "n",
}


def _romaji_word_to_ipa(word: str) -> str:
    """Convert a single Hepburn-romanised Japanese word to an IPA approximation.

    Uses greedy longest-match (3→2→1 chars) over _ROMAJI_TO_IPA.
    Double consonants (geminates) are simplified by skipping the first copy.

    Examples::

        Takagi  → takaɡi
        Hiroshi → hiɾoʃi
        Sano    → sano
        Ichiro  → itʃiɾo
    """
    s = word.lower()
    result: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        # Geminate consonant: skip first copy (e.g. "tt" → "t", "kk" → "k")
        if i + 1 < n and s[i] == s[i + 1] and s[i] not in "aeiou":
            i += 1
            continue
        matched = False
        for length in (3, 2, 1):
            chunk = s[i:i + length]
            if chunk in _ROMAJI_TO_IPA:
                result.append(_ROMAJI_TO_IPA[chunk])
                i += length
                matched = True
                break
        if not matched:
            result.append(s[i])
            i += 1
    return "".join(result)


# ---------------------------------------------------------------------------
# Japanese vs English language detection heuristic
# ---------------------------------------------------------------------------

# Two-char consonant clusters that are valid in Hepburn romanization.
# Any other consonant cluster (e.g. "th", "br", "str") marks the word as English.
_JA_DIGRAPHS = frozenset({
    "sh", "ch", "ts", "ky", "gy", "hy", "by", "py", "my", "ny", "ry",
})


def _is_japanese_romaji(word: str) -> bool:
    """Return True if *word* looks like Hepburn-romanised Japanese.

    Heuristic: Japanese romaji has no non-Japanese consonant clusters.
    If we find a pair of adjacent consonants that isn't a known Japanese digraph
    (sh, ch, ts, ky, …), the word is classified as English.

    Examples:
        Takagi  → True   (only CV syllables)
        Hiroshi → True   (contains "sh" digraph)
        Sano    → True   (only CV syllables)
        Thor    → False  ("th" is not a Japanese digraph)
        Steve   → False  ("st" consonant cluster)
        Arthur  → False  ("th", "rt" clusters)
    """
    w = word.lower()
    for i, c in enumerate(w):
        if c in "aeioun":       # vowels and syllabic-n: fine
            continue
        nxt = w[i + 1] if i + 1 < len(w) else ""
        if nxt and nxt not in "aeiou":
            # Two consecutive consonants: must be a known Japanese digraph
            if w[i:i + 2] not in _JA_DIGRAPHS:
                return False
    return True


class TTSEnginePiper:
    """
    Offline TTS using Piper (ONNX Runtime + VITS model).

    Supports up to 3 simultaneous Piper models:
      • model_path            – Vietnamese (primary)
      • foreign_model_path_en – English names
      • foreign_model_path_ja – Japanese names

    When foreign models are configured the engine auto-detects the language
    of each foreign-name segment and routes it to the correct model.
    Without foreign models it falls back to [[IPA]] injection.
    """

    def __init__(
        self,
        model_path: str,
        speaker_id: int | None = None,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w_scale: float = 0.8,
        foreign_phoneme: bool = True,
        pronunciation_dict: dict[str, str] | None = None,
        foreign_lang: str = "ja",
        foreign_model_path_en: str = "",
        foreign_model_path_ja: str = "",
    ):
        self.model_path = model_path
        self.speaker_id = speaker_id
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w_scale = noise_w_scale
        self.foreign_phoneme = foreign_phoneme
        self.pronunciation_dict: dict[str, str] = pronunciation_dict or {}
        self.foreign_lang = foreign_lang          # IPA fallback: "ja" romaji or "en" espeak
        self.foreign_model_path_en = foreign_model_path_en
        self.foreign_model_path_ja = foreign_model_path_ja

        self._voice = None
        self._sample_rate: int = 22050

        self._en_voice = None
        self._en_sample_rate: int = 22050

        self._ja_voice = None
        self._ja_sample_rate: int = 22050

        self._espeak = None   # lazy-loaded EspeakPhonemizer (IPA fallback only)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        from piper import PiperVoice  # type: ignore

        print(f"[PiperTTS] Loading VI model: {self.model_path}")
        self._voice = PiperVoice.load(self.model_path)
        self._sample_rate = self._voice.config.sample_rate
        print(f"[PiperTTS] VI model loaded  ({self._sample_rate} Hz, "
              f"{self._voice.config.num_speakers} speaker(s))")

    def _load_en(self) -> None:
        from piper import PiperVoice  # type: ignore

        print(f"[PiperTTS] Loading EN model: {self.foreign_model_path_en}")
        self._en_voice = PiperVoice.load(self.foreign_model_path_en)
        self._en_sample_rate = self._en_voice.config.sample_rate
        print(f"[PiperTTS] EN model loaded  ({self._en_sample_rate} Hz)")

    def _load_ja(self) -> None:
        from piper import PiperVoice  # type: ignore

        print(f"[PiperTTS] Loading JA model: {self.foreign_model_path_ja}")
        self._ja_voice = PiperVoice.load(self.foreign_model_path_ja)
        self._ja_sample_rate = self._ja_voice.config.sample_rate
        print(f"[PiperTTS] JA model loaded  ({self._ja_sample_rate} Hz)")

    # ------------------------------------------------------------------
    # IPA fallback helpers (used when no foreign model is configured)
    # ------------------------------------------------------------------

    def _get_ipa_en(self, phrase: str) -> str:
        """Return IPA string for *phrase* using en-us espeak voice."""
        if self._espeak is None:
            from piper.voice import EspeakPhonemizer  # type: ignore
            self._espeak = EspeakPhonemizer()
        phonemes = self._espeak.phonemize("en-us", phrase)
        return "".join("".join(s) for s in phonemes).strip()

    def _get_ipa(self, phrase: str) -> str:
        """Return IPA for *phrase* using foreign_lang setting.

        "ja" → Hepburn romaji→IPA (word-by-word).
        "en" → en-us espeak.
        """
        if self.foreign_lang == "ja":
            return " ".join(_romaji_word_to_ipa(w) for w in phrase.split())
        return self._get_ipa_en(phrase)

    def _inject_foreign_phonemes(self, text: str) -> str:
        """Replace capitalised ASCII-only word sequences with [[IPA]] notation.

        Priority order for each matched phrase:
          1. Exact phrase match in pronunciation_dict
          2. Word-by-word lookup in pronunciation_dict
          3. Auto IPA via _get_ipa() (romaji or en-us espeak)
          4. Leave unchanged
        """
        if not self.foreign_phoneme and not self.pronunciation_dict:
            return text

        def replace(m: re.Match) -> str:
            phrase = m.group(0)

            if all(w in _VI_PLAIN_WORDS for w in phrase.split()):
                return phrase

            if phrase in self.pronunciation_dict:
                return f"[[{self.pronunciation_dict[phrase]}]]"

            words = phrase.split()
            if any(w in self.pronunciation_dict for w in words):
                parts = []
                for word in words:
                    if word in self.pronunciation_dict:
                        parts.append(f"[[{self.pronunciation_dict[word]}]]")
                    elif self.foreign_phoneme:
                        parts.append(f"[[{self._get_ipa(word)}]]")
                    else:
                        parts.append(word)
                return " ".join(parts)

            if self.foreign_phoneme:
                return f"[[{self._get_ipa(phrase)}]]"

            return phrase

        return _FOREIGN_PHRASE_RE.sub(replace, text)

    # ------------------------------------------------------------------
    # 3-model mode: language-segment splitting
    # ------------------------------------------------------------------

    def _apply_dict_overrides(self, phrase: str) -> str:
        """Apply pronunciation_dict IPA overrides to *phrase*.

        Used when routing to a foreign model so user overrides still apply.
        """
        if not self.pronunciation_dict:
            return phrase
        if phrase in self.pronunciation_dict:
            return f"[[{self.pronunciation_dict[phrase]}]]"
        words = phrase.split()
        if any(w in self.pronunciation_dict for w in words):
            parts = []
            for word in words:
                if word in self.pronunciation_dict:
                    parts.append(f"[[{self.pronunciation_dict[word]}]]")
                else:
                    parts.append(word)
            return " ".join(parts)
        return phrase

    def _split_into_language_segments(self, text: str) -> list[tuple[str, str]]:
        """Split *text* into (segment_text, lang) pairs.

        lang is one of: "vi" | "en" | "ja"

        Foreign-name detection uses _FOREIGN_PHRASE_RE; language classification
        uses _is_japanese_romaji() on the non-Vietnamese words in each phrase.
        """
        segments: list[tuple[str, str]] = []
        last_end = 0

        for m in _FOREIGN_PHRASE_RE.finditer(text):
            phrase = m.group(0)
            words = phrase.split()

            # Skip Vietnamese plain-ASCII words
            if all(w in _VI_PLAIN_WORDS for w in words):
                continue

            start, end = m.span()

            # Vietnamese segment before this foreign phrase
            if start > last_end:
                segments.append((text[last_end:start], "vi"))

            # Classify language of this phrase
            non_vi = [w for w in words if w not in _VI_PLAIN_WORDS]
            lang = "ja" if any(_is_japanese_romaji(w) for w in non_vi) else "en"
            segments.append((phrase, lang))
            last_end = end

        # Remaining text
        if last_end < len(text):
            segments.append((text[last_end:], "vi"))

        return segments

    # ------------------------------------------------------------------
    # Low-level synthesis helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resample(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        """Resample *wav* from *src_sr* to *dst_sr* using linear interpolation."""
        if src_sr == dst_sr:
            return wav
        new_len = max(1, int(len(wav) * dst_sr / src_sr))
        x_old = np.linspace(0.0, 1.0, len(wav))
        x_new = np.linspace(0.0, 1.0, new_len)
        return np.interp(x_new, x_old, wav).astype(np.float32)

    def _synth_wav(self, voice, text: str, speaker_id: int | None = None) -> np.ndarray:
        """Synthesize *text* with *voice* and return a float32 wav array."""
        from piper.config import SynthesisConfig  # type: ignore

        syn_config = SynthesisConfig(
            speaker_id=speaker_id,
            length_scale=self.length_scale,
            noise_scale=self.noise_scale,
            noise_w_scale=self.noise_w_scale,
        )
        chunks: list[np.ndarray] = []
        for audio_chunk in voice.synthesize(text, syn_config):
            chunks.append(audio_chunk.audio_int16_array)
        wav_int16 = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
        return wav_int16.astype(np.float32) / 32768.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Convert *text* to audio via Piper.

        3-model mode (foreign_model_path_en / foreign_model_path_ja set):
            Splits text by language segment and routes each to the correct model.

        Fallback (no foreign models):
            Injects [[IPA]] phonemes for foreign names into the Vietnamese model.

        Returns:
            (wav, sample_rate) where wav is a 1-D float32 numpy array.
        """
        if self._voice is None:
            self._load()

        use_foreign_models = bool(
            self.foreign_model_path_en or self.foreign_model_path_ja
        )

        if not use_foreign_models:
            # Fallback: optional IPA injection into the Vietnamese model
            if self.foreign_phoneme:
                text = self._inject_foreign_phonemes(text)
            wav = self._synth_wav(self._voice, text, self.speaker_id)
            return wav, self._sample_rate

        # --- 3-model mode ---
        segments = self._split_into_language_segments(text)

        # No foreign names detected: synthesise whole text with VI model
        if all(lang == "vi" for _, lang in segments):
            text = self._inject_foreign_phonemes(text)
            wav = self._synth_wav(self._voice, text, self.speaker_id)
            return wav, self._sample_rate

        parts: list[np.ndarray] = []
        for seg_text, lang in segments:
            seg_text = seg_text.strip()
            if not seg_text or not any(c.isalpha() for c in seg_text):
                continue

            if lang == "ja" and self.foreign_model_path_ja:
                # Dedicated Japanese model
                if self._ja_voice is None:
                    self._load_ja()
                seg_text = self._apply_dict_overrides(seg_text)
                wav = self._synth_wav(self._ja_voice, seg_text, None)
                wav = self._resample(wav, self._ja_sample_rate, self._sample_rate)

            elif lang in ("en", "ja") and self.foreign_model_path_en:
                # English model handles English names; also used as JA fallback
                # when no Japanese model is configured.
                if self._en_voice is None:
                    self._load_en()
                seg_text = self._apply_dict_overrides(seg_text)
                wav = self._synth_wav(self._en_voice, seg_text, None)
                wav = self._resample(wav, self._en_sample_rate, self._sample_rate)

            else:
                # No foreign model available: optional IPA injection into VI model
                if self.foreign_phoneme:
                    seg_text = self._inject_foreign_phonemes(seg_text)
                wav = self._synth_wav(self._voice, seg_text, self.speaker_id)

            parts.append(wav)

        wav = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        return wav, self._sample_rate

    def unload(self) -> None:
        """Release all models from memory."""
        self._voice = None
        self._en_voice = None
        self._ja_voice = None
        self._espeak = None
