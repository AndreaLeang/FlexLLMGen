#!/usr/bin/env python3
"""Recursively compress all .json files under a directory using zstandard.

Each file's compressed output replaces the .json extension with .zst
(e.g. data.json -> data.zst). Originals are deleted after a successful
compression by default; pass --keep-original to keep them.

Usage:
    python compress_json.py path/
    python compress_json.py path/ --level 9 --keep-original
"""

import argparse
import sys
from pathlib import Path

import zstandard as zstd


def compress_file(src: Path, level: int) -> Path:
    dst = src.with_suffix(".zst")
    cctx = zstd.ZstdCompressor(level=level)
    with open(src, "rb") as f_in, open(dst, "wb") as f_out:
        cctx.copy_stream(f_in, f_out)
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursively compress .json files to .zst"
    )
    parser.add_argument("path", type=Path, help="Root folder to search")
    parser.add_argument(
        "--level", type=int, default=3, help="Zstd compression level (default: 3)"
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Keep the original .json files after compressing",
    )
    args = parser.parse_args()

    if not args.path.is_dir():
        print(f"Not a directory: {args.path}", file=sys.stderr)
        sys.exit(1)

    json_files = list(args.path.rglob("*.json"))
    if not json_files:
        print(f"No .json files found under {args.path}")
        return

    ok, failed = 0, 0
    for src in json_files:
        try:
            dst = compress_file(src, level=args.level)
            print(f"Compressed: {src} -> {dst}")
            if not args.keep_original:
                src.unlink()
            ok += 1
        except Exception as e:
            print(f"Failed to compress {src}: {e}", file=sys.stderr)
            failed += 1

    print(f"\nDone. {ok} compressed, {failed} failed.")


if __name__ == "__main__":
    main()