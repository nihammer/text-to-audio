"""F5-TTS wrapper with lazy model loading and Apple Silicon (MPS) support."""
import logging
import warnings
import numpy as np

# Vietnamese average speaking rate (characters per second).
# Used to compute fix_duration: caps the mel-sequence length fed to F5-TTS,
# preventing the default formula from over-generating audio (≈2× too long).
_CHARS_PER_SEC = 13.0

# Safety buffer on top of the expected generation duration.
# 1.4 = 40 % extra to avoid clipping the last syllable.
_DURATION_BUFFER = 1.4

# Duration (seconds) of reference audio after trimming in prepare_reference_wav.
# Must match audio_merger.prepare_reference_wav(max_duration_ms=...).
_REF_AUDIO_SEC = 4.0


class TTSEngine:
    """Thin wrapper around F5-TTS that lazy-loads the model on first use."""

    def __init__(
        self,
        ref_audio_path: str,
        device: str = "mps",
        nfe_step: int = 16,
        speed: float = 1.0,
        chars_per_sec: float = _CHARS_PER_SEC,
        duration_buffer: float = _DURATION_BUFFER,
        ckpt_file: str = "",
        vocab_file: str = "",
    ):
        self.ref_audio = ref_audio_path
        self.ref_text = ""  # filled once in _load() via F5-TTS internal Whisper
        self._preprocessed_ref_audio = None  # path to preprocessed temp WAV (set in _load())
        self.device = device
        self.nfe_step = nfe_step
        self.speed = speed
        self.chars_per_sec = chars_per_sec
        self.duration_buffer = duration_buffer
        self.ckpt_file = ckpt_file   # path to fine-tuned checkpoint, or "" for default
        self.vocab_file = vocab_file  # path to vocab.txt, or "" for default
        self._model = None
        self._vocoder = None       # stored to call infer_process() directly
        self._mel_spec_type = None  # stored to call infer_process() directly

    def _load(self) -> None:
        # --- Suppress noisy library warnings before any imports ---------------
        # 1. transformers uses torch_dtype= kwarg (deprecated → use dtype=).
        #    The warning is emitted via transformers' own logger (warning_once),
        #    so we must use its API — plain logging.getLogger() is not enough.
        import transformers as _transformers
        _transformers.logging.set_verbosity_error()

        # 2. HuggingFace Hub unauthenticated-request notice: suppress via
        #    huggingface_hub's own logging API.
        import huggingface_hub.utils.logging as _hf_log
        _hf_log.set_verbosity_error()

        # 3. Catch any remaining warnings emitted via Python warnings module
        #    (e.g. from older transformers / third-party code).
        warnings.filterwarnings("ignore", message=r".*torch_dtype.*deprecated.*")
        warnings.filterwarnings("ignore", message=r".*chunk_length_s.*experimental.*")
        warnings.filterwarnings("ignore", message=r".*logits_process.*")
        # 4. PyTorch stft() output-resize deprecation (triggered by pydub silence
        #    detection inside preprocess_ref_audio_text).
        warnings.filterwarnings("ignore", message=r".*resized since it had shape.*",
                                category=UserWarning)
        # ----------------------------------------------------------------------

        # --- Patch rjieba BEFORE importing F5TTS ----------------------------
        # F5-TTS uses rjieba (Chinese word segmenter) inside
        # convert_char_to_pinyin(). For Vietnamese text, rjieba incorrectly
        # splits digraphs like "ng" and "nh" from their preceding vowel:
        #   "không" → "khô" + "ng"
        #   "hương" → "hươ" + "ng"
        # Then inserts a space between them, producing garbled phoneme input.
        # Fix: replace cut() with a no-op that returns the text as a single
        # segment, letting convert_char_to_pinyin handle each character
        # individually in its "mixed characters" branch (correct for Vietnamese).
        import rjieba as _rjieba
        _rjieba.cut = lambda text: [text]
        # -------------------------------------------------------------------

        from f5_tts.api import F5TTS  # type: ignore

        print(f"[TTS] Loading F5-TTS model on {self.device}…")
        kwargs: dict = {"device": self.device}
        if self.ckpt_file:
            # Vietnamese fine-tuned checkpoint (F5TTS_Base architecture)
            kwargs["model"] = "F5TTS_Base"
            kwargs["ckpt_file"] = self.ckpt_file
        if self.vocab_file:
            kwargs["vocab_file"] = self.vocab_file
        self._model = F5TTS(**kwargs)

        # Cache vocoder and mel_spec_type so synthesize() can call
        # infer_process() directly without going through F5TTS.infer().
        self._vocoder = self._model.vocoder
        self._mel_spec_type = self._model.mel_spec_type

        # Transcribe reference audio ONCE so every synthesize() call uses the
        # correct ref_text.  When ref_text="" F5-TTS re-transcribes per call,
        # which (a) wastes time and (b) can mis-align the ref/gen boundary if
        # the Whisper result is wrong, causing reference audio to bleed into
        # the generated output.
        from f5_tts.infer.utils_infer import transcribe, preprocess_ref_audio_text  # type: ignore
        self.ref_text = transcribe(self.ref_audio, language="vi")
        print(f"[TTS] ref_text: {self.ref_text!r}")

        # Pre-process reference audio ONCE.  preprocess_ref_audio_text() trims
        # silence, exports a clean temp WAV, and normalises ref_text.  The
        # result is stored so synthesize() passes the pre-processed audio
        # directly to infer_process(), skipping this step on every chunk call.
        self._preprocessed_ref_audio, self.ref_text = preprocess_ref_audio_text(
            self.ref_audio, self.ref_text
        )
        print("[TTS] Model loaded.")

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """
        Convert *text* to audio.

        fix_duration caps the total mel-sequence length (ref + gen) so F5-TTS
        does not over-generate.  Without it, F5-TTS uses a byte-ratio formula
        that produces ~2× more audio than needed, making sequences ~2627 mel
        frames long.  With fix_duration the sequence is ~1125 frames → ~5× less
        attention computation (O(n²) attention).

        Returns:
            (wav, sample_rate) where wav is a 1-D float32 numpy array.
        """
        if not self._model:
            self._load()

        expected_gen_s = len(text) / self.chars_per_sec * self.duration_buffer
        fix_duration = _REF_AUDIO_SEC + expected_gen_s  # total: ref + gen

        # Call infer_process() directly with the pre-processed reference audio,
        # bypassing F5TTS.infer() which would re-run preprocess_ref_audio_text()
        # (and print "Converting audio..." / "ref_text ..." / "Using custom
        # reference text...") on every single chunk.
        # infer_process() also hard-codes print("gen_text N", ...) which we
        # suppress by temporarily redirecting stdout.
        import io, sys as _sys
        from f5_tts.infer.utils_infer import infer_process  # type: ignore

        _old_stdout = _sys.stdout
        _sys.stdout = io.StringIO()
        try:
            wav, sr, _ = infer_process(
                self._preprocessed_ref_audio,
                self.ref_text,
                text,
                self._model.ema_model,
                self._vocoder,
                self._mel_spec_type,
                show_info=lambda *a: None,  # suppress "Generating audio in N batches…"
                progress=None,              # suppress inner tqdm (we track at chunk level)
                nfe_step=self.nfe_step,
                speed=self.speed,
                fix_duration=fix_duration,
                device=self.device,         # must match device model was loaded on
            )
        finally:
            _sys.stdout = _old_stdout

        # Ensure 1-D float32 array
        wav = np.array(wav, dtype=np.float32).flatten()
        return wav, int(sr)

    def unload(self) -> None:
        """Release model from memory (useful between files to free VRAM)."""
        if self._model is not None:
            del self._model
            self._model = None
            self._vocoder = None
            self._mel_spec_type = None
            self._preprocessed_ref_audio = None
            try:
                import torch

                if self.device == "mps":
                    torch.mps.empty_cache()
                elif self.device == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass
