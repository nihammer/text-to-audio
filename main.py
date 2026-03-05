#!/usr/bin/env python3
"""
Vietnamese Novel Text-to-Audio CLI

Usage:
    python main.py process input/novel_part1.txt
    python main.py process-all
    python main.py status input/novel_part1.txt
"""
import argparse
import json
import sys
from pathlib import Path


def cmd_process(args: argparse.Namespace) -> None:
    from src.config import load_config
    from src.pipeline import process_file

    cfg = load_config()
    process_file(args.file, cfg=cfg)


def cmd_process_all(args: argparse.Namespace) -> None:
    from src.config import load_config
    from src.pipeline import process_file

    cfg = load_config()
    input_dir = Path(cfg["paths"]["input_dir"])

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in '{input_dir}'.")
        return

    print(f"Found {len(txt_files)} file(s) to process.")
    for txt in txt_files:
        print(f"\n{'='*60}")
        print(f"Processing: {txt}")
        print("=" * 60)
        try:
            process_file(txt, cfg=cfg)
        except Exception as e:
            print(f"[ERROR] Failed to process '{txt}': {e}", file=sys.stderr)


def cmd_status(args: argparse.Namespace) -> None:
    from src.config import load_config
    from src.pipeline import get_status

    cfg = load_config()
    status = get_status(args.file, cfg=cfg)

    print(f"\nStatus for: {args.file}")
    print(f"  Overall: {status['status']}")

    if status["status"] == "not_started":
        return

    print(f"  Chapters done: {status.get('chapters_done', 0)} / {status.get('chapters_total', 0)}")
    print("\n  Chapter breakdown:")
    for chapter_key, chapter_status in status.get("chapters", {}).items():
        if chapter_status == "done":
            marker = "✓"
            detail = "done"
        elif isinstance(chapter_status, dict):
            done_chunks = len(chapter_status.get("done_chunks", []))
            marker = "~"
            detail = f"in progress ({done_chunks} chunks synthesized)"
        else:
            marker = " "
            detail = str(chapter_status)
        print(f"    [{marker}] {chapter_key}: {detail}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="text-to-audio",
        description="Convert Vietnamese novel .txt files to per-chapter MP3s using F5-TTS.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # process
    p_process = subparsers.add_parser("process", help="Convert a single .txt file.")
    p_process.add_argument("file", help="Path to the input .txt file.")
    p_process.set_defaults(func=cmd_process)

    # process-all
    p_all = subparsers.add_parser("process-all", help="Convert all .txt files in the input/ directory.")
    p_all.set_defaults(func=cmd_process_all)

    # status
    p_status = subparsers.add_parser("status", help="Show progress for a given input file.")
    p_status.add_argument("file", help="Path to the input .txt file.")
    p_status.set_defaults(func=cmd_status)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
