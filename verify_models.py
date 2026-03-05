#!/usr/bin/env python3
"""
Verify SHA256 integrity of F5-TTS model files after download.

Known-good hashes sourced directly from HuggingFace blob pages:
  https://huggingface.co/SWivid/F5-TTS/blob/main/F5TTS_v1_Base/model_1250000.safetensors
  https://huggingface.co/charactr/vocos-mel-24khz/blob/main/pytorch_model.bin

Usage:
    python verify_models.py
"""
import hashlib
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Known-good SHA256 hashes (fetched from HuggingFace blob metadata, 2025-03)
# ---------------------------------------------------------------------------
KNOWN_HASHES: dict[str, str] = {
    # Primary F5-TTS model (v1, default used by f5-tts==1.1.x)
    "models--SWivid--F5-TTS/blobs/F5TTS_v1_Base/model_1250000.safetensors":
        "670900fd14e6c458b95da6e9ed317cdb20dbaf7a1c02ac06a05475a9d32b6a38",
    # Legacy base model (kept for reference)
    "models--SWivid--F5-TTS/blobs/F5TTS_Base/model_1200000.safetensors":
        "4180310f91d592cee4bc14998cd37c781f779cf105e8ca8744d9bd48ca7046ae",
    # Vocabulary file
    "models--SWivid--F5-TTS/blobs/F5TTS_v1_Base/vocab.txt":
        "2a05f992e00af9b0bd3800a8d23e78d520dbd705284ed2eedb5f4bd29398fa3c",
    # Vocos vocoder (used by F5-TTS for waveform synthesis)
    "models--charactr--vocos-mel-24khz/blobs/pytorch_model.bin":
        "97ec976ad1fd67a33ab2682d29c0ac7df85234fae875aefcc5fb215681a91b2a",
}

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def find_model_files(cache_dir: Path) -> dict[str, Path]:
    """
    Walk the HuggingFace cache and collect all .safetensors, .pt, .bin, and
    vocab.txt files, keyed by a short logical name for display.
    """
    found: dict[str, Path] = {}
    for path in cache_dir.rglob("*"):
        if path.is_file() and path.suffix in (".safetensors", ".bin", ".pt") or (
            path.is_file() and path.name == "vocab.txt"
        ):
            # Build a short key relative to cache_dir for matching
            rel = str(path.relative_to(cache_dir))
            found[rel] = path
    return found


def main() -> None:
    if not HF_CACHE.exists():
        print(f"HuggingFace cache not found at {HF_CACHE}")
        print("Run the tool once first so models are downloaded, then re-run this script.")
        sys.exit(0)

    print(f"Scanning: {HF_CACHE}\n")

    cached_files = find_model_files(HF_CACHE)
    if not cached_files:
        print("No model files found in cache. Download models by running:")
        print("  python main.py process input/<any>.txt")
        sys.exit(0)

    passed = 0
    failed = 0
    skipped = 0

    for logical_key, expected_hash in KNOWN_HASHES.items():
        # Match by the final component of the logical key (filename)
        filename = Path(logical_key).name
        matches = [
            (rel, path)
            for rel, path in cached_files.items()
            if Path(rel).name == filename
        ]

        if not matches:
            print(f"  [SKIP] Not downloaded yet: {filename}")
            skipped += 1
            continue

        for rel, path in matches:
            print(f"  Checking {path.name} ...", end=" ", flush=True)
            actual = sha256_file(path)
            if actual == expected_hash:
                print("OK")
                passed += 1
            else:
                print("FAIL")
                print(f"    Expected: {expected_hash}")
                print(f"    Got:      {actual}")
                print(f"    Path:     {path}")
                failed += 1

    print(f"\nResults: {passed} passed, {failed} failed, {skipped} not yet downloaded")

    if failed > 0:
        print("\nWARNING: One or more model files do not match known-good hashes.")
        print("Delete the affected files and re-download:")
        print(f"  rm -rf {HF_CACHE}/models--SWivid--F5-TTS")
        print(f"  rm -rf {HF_CACHE}/models--charactr--vocos-mel-24khz")
        sys.exit(1)
    elif skipped > 0 and passed == 0:
        print("\nNo models verified yet — run the tool first to trigger download.")
    else:
        print("\nAll downloaded model files are intact.")


if __name__ == "__main__":
    main()
