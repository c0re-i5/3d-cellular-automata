#!/usr/bin/env python3
"""Salvage a truncated discoveries.json by keeping every complete array entry.

The file is a JSON array of objects written with ``json.dump(..., indent=1)``.
A truncated write leaves a final partial object and an unterminated array. We
recover by trimming back to the last cleanly-closed element (its close brace is
``\\n }`` at one-space indent) and re-terminating the array.

Prints the salvaged entry count. Does NOT overwrite in place unless --write is
given and an explicit --backup target is written first.
"""
import argparse
import json
import os
import shutil
import sys
import time


def salvage(raw: str):
    # Fast path: already valid.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Trim to the last complete element. Elements close with "\n }" (1-space
    # indent) followed by "," (or the array's final "]"). Keep through that
    # brace, drop the truncated remainder, and re-close the array.
    for needle in ("\n },", "\n }"):
        idx = raw.rfind(needle)
        if idx == -1:
            continue
        end = idx + len("\n }")  # position just past the closing brace
        candidate = raw[:end] + "\n]"
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("could not salvage any complete entries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--write", action="store_true",
                    help="Overwrite path with the salvaged data (after backup).")
    args = ap.parse_args()

    with open(args.path, encoding="utf-8") as f:
        raw = f.read()

    before_ok = True
    try:
        json.loads(raw)
    except json.JSONDecodeError as e:
        before_ok = False
        print(f"input INVALID: {e}")

    data = salvage(raw)
    print(f"salvaged entries: {len(data)}")

    if args.write:
        if before_ok:
            print("input already valid — nothing to write.")
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup = f"{args.path}.corrupt_{ts}.bak"
        shutil.copy2(args.path, backup)
        print(f"backed up corrupt original -> {backup}")
        # Atomic replace.
        tmp = f"{args.path}.tmp_{ts}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, args.path)
        print(f"wrote salvaged data -> {args.path} ({len(data)} entries)")


if __name__ == "__main__":
    sys.exit(main())
