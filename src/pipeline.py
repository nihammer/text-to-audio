"""Orchestrator: input file → chapter detection → TTS → MP3 output."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .audio_merger import merge_chunks, prepare_reference_wav
from .chapter_detector import Chapter, detect_chapters
from .chunker import Chunk, split_into_chunks
from .config import get_config
from .preprocessor import clean_chapter_text, normalize
from .tts_engine import TTSEngine
from .tts_engine_edge import TTSEngineEdge
from .tts_engine_piper import TTSEnginePiper


def _checkpoint_path(stem: str, checkpoint_dir: Path) -> Path:
    return checkpoint_dir / f"{stem}.json"


def _load_checkpoint(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_checkpoint(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _chapter_key(chapter: Chapter) -> str:
    return f"chapter_{chapter.number}"


def _ensure_reference_wav(cfg: dict) -> str:
    wav_path = Path(cfg["reference"]["wav_path"])
    if not wav_path.exists():
        src = cfg["reference"]["audio_src"]
        start_ms = cfg["reference"].get("start_ms", 0)
        prepare_reference_wav(src, wav_path, start_ms=start_ms)
    return str(wav_path)


def _build_engine(cfg: dict):
    """Instantiate TTS engine based on cfg['tts']['engine'] setting."""
    engine_type = cfg["tts"].get("engine", "f5")
    if engine_type == "edge":
        edge_cfg = cfg["tts"].get("edge", {})
        print(f"[pipeline] Using Edge-TTS engine (voice: {edge_cfg.get('voice', 'vi-VN-HoaiMyNeural')})")
        return TTSEngineEdge(
            voice=edge_cfg.get("voice", "vi-VN-HoaiMyNeural"),
            rate=edge_cfg.get("rate", "+0%"),
            volume=edge_cfg.get("volume", "+0%"),
        )
    if engine_type == "piper":
        piper_cfg = cfg["tts"].get("piper", {})
        model_path = piper_cfg.get("model_path", "")
        print(f"[pipeline] Using Piper TTS engine (model: {model_path})")
        return TTSEnginePiper(
            model_path=model_path,
            speaker_id=piper_cfg.get("speaker_id"),
            length_scale=piper_cfg.get("length_scale", 1.0),
            noise_scale=piper_cfg.get("noise_scale", 0.667),
            noise_w_scale=piper_cfg.get("noise_w_scale", 0.8),
            foreign_phoneme=piper_cfg.get("foreign_phoneme", True),
            pronunciation_dict=piper_cfg.get("pronunciation_dict") or {},
            foreign_lang=piper_cfg.get("foreign_lang", "ja"),
            foreign_model_path_en=piper_cfg.get("foreign_model_path_en", ""),
            foreign_model_path_ja=piper_cfg.get("foreign_model_path_ja", ""),
        )
    # Default: F5-TTS
    ref_wav = _ensure_reference_wav(cfg)
    print(f"[pipeline] Using F5-TTS engine (device: {cfg['tts']['device']})")
    return TTSEngine(
        ref_audio_path=ref_wav,
        device=cfg["tts"]["device"],
        nfe_step=cfg["tts"]["nfe_step"],
        speed=cfg["tts"]["speed"],
        ckpt_file=cfg["tts"].get("ckpt_file", ""),
        vocab_file=cfg["tts"].get("vocab_file", ""),
    )


def process_file(txt_path: str | Path, cfg: dict | None = None) -> None:
    """
    Convert a single .txt file to per-chapter MP3s.

    Supports resume: if interrupted, re-running will continue from the last
    completed chunk inside an unfinished chapter.
    """
    if cfg is None:
        cfg = get_config()

    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"Input file not found: {txt_path}")

    stem = txt_path.stem
    output_dir = Path(cfg["paths"]["output_dir"]) / stem
    cache_dir = Path(cfg["paths"]["cache_dir"]) / stem
    checkpoint_dir = Path(cfg["paths"]["checkpoint_dir"])

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = _checkpoint_path(stem, checkpoint_dir)
    checkpoint = _load_checkpoint(checkpoint_path)

    # Read and detect chapters
    text = txt_path.read_text(encoding="utf-8")
    chapters = detect_chapters(text)
    print(f"[pipeline] Detected {len(chapters)} chapter(s) in '{txt_path.name}'")

    # Build TTS engine
    engine = _build_engine(cfg)

    all_done = True

    for chapter in chapters:
        key = _chapter_key(chapter)
        chapter_status = checkpoint.get(key, {})

        if chapter_status == "done":
            print(f"[pipeline] Skipping completed: {chapter.title}")
            continue

        all_done = False
        mp3_name = f"chuong_{chapter.number:03d}.mp3"
        mp3_path = output_dir / mp3_name

        print(f"\n[pipeline] Processing: {chapter.title} → {mp3_name}")

        # Build chunk list: title chunk first (standalone, never merged),
        # then a paragraph pause, then the body chunks.
        # chapter_detector stores the heading in chapter.title and strips it
        # from chapter.text, so we must synthesize it separately.
        # When title is empty (no chapter headings found), skip title synthesis.
        body = clean_chapter_text(chapter.text)
        body_chunks = split_into_chunks(
            body,
            max_chars=cfg["chunker"]["max_chars"],
            min_chars=cfg["chunker"].get("min_chars", 50),
        )
        if chapter.title:
            title_chunk = Chunk(normalize(chapter.title), "sentence", -1)
            chunks = [title_chunk, Chunk("", "paragraph_break", -1)] + body_chunks
        else:
            chunks = body_chunks
        # Filter out empty paragraph-break sentinels from counting
        text_chunks = [c for c in chunks if c.chunk_type != "paragraph_break"]

        done_chunks: list[int] = chapter_status.get("done_chunks", []) if isinstance(chapter_status, dict) else []

        # Synthesize each chunk
        chunk_wavs: list[tuple[np.ndarray, int, str]] = []

        with tqdm(total=len(text_chunks), desc=chapter.title, unit="chunk") as pbar:
            text_chunk_idx = 0
            for chunk in chunks:
                if chunk.chunk_type == "paragraph_break":
                    # Sentinel: just pass through with dummy arrays
                    chunk_wavs.append((np.zeros(1, dtype=np.float32), 24000, "paragraph_break"))
                    continue

                if text_chunk_idx in done_chunks:
                    # Load cached WAV for already-synthesized chunk
                    cached = _load_cached_chunk(cache_dir, chapter.number, text_chunk_idx)
                    if cached is not None:
                        chunk_wavs.append(cached)
                    else:
                        # Cache missing — re-synthesize
                        wav, sr = engine.synthesize(chunk.text)
                        _save_cached_chunk(cache_dir, chapter.number, text_chunk_idx, wav, sr)
                        chunk_wavs.append((wav, sr, "sentence"))
                    pbar.update(1)
                    text_chunk_idx += 1
                    continue

                wav, sr = engine.synthesize(chunk.text)
                _save_cached_chunk(cache_dir, chapter.number, text_chunk_idx, wav, sr)
                chunk_wavs.append((wav, sr, "sentence"))

                done_chunks.append(text_chunk_idx)
                checkpoint[key] = {"done_chunks": done_chunks}
                _save_checkpoint(checkpoint_path, checkpoint)

                pbar.update(1)
                text_chunk_idx += 1

        # Merge all chunks into final MP3
        print(f"[pipeline] Merging {len(chunk_wavs)} segments → {mp3_path}")
        merge_chunks(
            chunk_wavs,
            mp3_path,
            bitrate=cfg["audio"]["bitrate"],
            silence_sentence_ms=cfg["audio"]["silence_sentence_ms"],
            silence_paragraph_ms=cfg["audio"]["silence_paragraph_ms"],
        )

        # Mark chapter done and clean cache
        checkpoint[key] = "done"
        _save_checkpoint(checkpoint_path, checkpoint)
        _clean_chapter_cache(cache_dir, chapter.number)

        print(f"[pipeline] Done: {mp3_path}")

    engine.unload()

    # If everything is done, remove the checkpoint file
    if all(v == "done" for v in checkpoint.values()) and len(checkpoint) == len(chapters):
        checkpoint_path.unlink(missing_ok=True)
        print(f"[pipeline] All chapters complete. Checkpoint removed.")

    print(f"\n[pipeline] Finished '{txt_path.name}'. Output: {output_dir}")


def _cached_chunk_path(cache_dir: Path, chapter_num: int, chunk_idx: int) -> Path:
    return cache_dir / f"ch{chapter_num:03d}_ck{chunk_idx:05d}.npy"


def _save_cached_chunk(cache_dir: Path, chapter_num: int, chunk_idx: int, wav: np.ndarray, sr: int) -> None:
    path = _cached_chunk_path(cache_dir, chapter_num, chunk_idx)
    np.save(str(path), {"wav": wav, "sr": sr})


def _load_cached_chunk(
    cache_dir: Path, chapter_num: int, chunk_idx: int
) -> tuple[np.ndarray, int, str] | None:
    path = _cached_chunk_path(cache_dir, chapter_num, chunk_idx)
    if not path.exists():
        return None
    data = np.load(str(path), allow_pickle=True).item()
    return data["wav"], data["sr"], "sentence"


def _clean_chapter_cache(cache_dir: Path, chapter_num: int) -> None:
    prefix = f"ch{chapter_num:03d}_"
    for f in cache_dir.glob(f"{prefix}*.npy"):
        f.unlink()


def get_status(txt_path: str | Path, cfg: dict | None = None) -> dict:
    """Return checkpoint status for a given input file."""
    if cfg is None:
        cfg = get_config()

    txt_path = Path(txt_path)
    stem = txt_path.stem
    checkpoint_dir = Path(cfg["paths"]["checkpoint_dir"])
    checkpoint_path = _checkpoint_path(stem, checkpoint_dir)
    checkpoint = _load_checkpoint(checkpoint_path)

    if not checkpoint:
        return {"status": "not_started", "chapters": {}}

    total = len(checkpoint)
    done = sum(1 for v in checkpoint.values() if v == "done")

    return {
        "status": "complete" if done == total else "in_progress",
        "chapters_done": done,
        "chapters_total": total,
        "chapters": checkpoint,
    }
