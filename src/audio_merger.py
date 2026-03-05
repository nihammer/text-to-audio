"""Concatenate WAV chunks into a single MP3 file using pydub."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from pydub import AudioSegment


def numpy_to_segment(wav: np.ndarray, sample_rate: int) -> AudioSegment:
    """Convert a float32 numpy array to a pydub AudioSegment."""
    # Normalise to int16 PCM
    wav_int16 = (wav * 32767).clip(-32768, 32767).astype(np.int16)
    return AudioSegment(
        wav_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,  # int16 → 2 bytes
        channels=1,
    )


def silence(ms: int, sample_rate: int = 24000) -> AudioSegment:
    return AudioSegment.silent(duration=ms, frame_rate=sample_rate)


def merge_chunks(
    chunk_wavs: list[tuple[np.ndarray, int, str]],  # (wav, sr, chunk_type)
    output_path: str | Path,
    bitrate: str = "192k",
    silence_sentence_ms: int = 200,
    silence_paragraph_ms: int = 500,
) -> None:
    """
    Merge synthesized chunks into a single MP3.

    Args:
        chunk_wavs: List of (wav_array, sample_rate, chunk_type).
                    chunk_type is "sentence" | "paragraph_break".
        output_path: Destination .mp3 file path.
        bitrate: MP3 bitrate string (e.g. "192k").
        silence_sentence_ms: Pause between sentence chunks.
        silence_paragraph_ms: Pause at paragraph breaks.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not chunk_wavs:
        raise ValueError("No chunks provided to merge.")

    combined: AudioSegment | None = None
    sr = chunk_wavs[0][1]  # use first chunk's sample rate

    for i, (wav, chunk_sr, chunk_type) in enumerate(chunk_wavs):
        if chunk_type == "paragraph_break":
            # Only add silence; no audio for this sentinel
            seg = silence(silence_paragraph_ms, chunk_sr)
        else:
            seg = numpy_to_segment(wav, chunk_sr)
            # Add inter-sentence silence after each sentence chunk (except last)
            if i < len(chunk_wavs) - 1 and chunk_wavs[i + 1][2] != "paragraph_break":
                seg = seg + silence(silence_sentence_ms, chunk_sr)

        combined = seg if combined is None else combined + seg

    if combined is None:
        raise RuntimeError("Audio merge produced no output.")

    combined.export(str(output_path), format="mp3", bitrate=bitrate)


def prepare_reference_wav(
    src_mp3: str | Path,
    dest_wav: str | Path,
    start_ms: int = 0,
    max_duration_ms: int = 4000,
) -> None:
    """Convert reference MP3 to 24kHz mono WAV expected by F5-TTS.

    Slices [start_ms, start_ms + max_duration_ms] from the source audio,
    then converts to 24kHz mono WAV.  Use start_ms to skip intros (e.g.
    YouTube subscribe phrases) and pick the clean speech portion.
    """
    src_mp3 = Path(src_mp3)
    dest_wav = Path(dest_wav)
    dest_wav.parent.mkdir(parents=True, exist_ok=True)

    if not src_mp3.exists():
        raise FileNotFoundError(f"Reference audio not found: {src_mp3}")

    audio = AudioSegment.from_mp3(str(src_mp3))
    audio = audio.set_frame_rate(24000).set_channels(1)

    if start_ms > 0:
        audio = audio[start_ms:]
        print(f"[ref] Skipped first {start_ms / 1000:.1f}s of reference audio")

    if len(audio) > max_duration_ms:
        audio = audio[:max_duration_ms]

    print(f"[ref] Reference audio: {max_duration_ms / 1000:.1f}s from {start_ms / 1000:.1f}s")
    audio.export(str(dest_wav), format="wav")
    print(f"[ref] Converted reference audio → {dest_wav}")
