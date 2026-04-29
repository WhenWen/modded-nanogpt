#!/usr/bin/env python3
"""Summarize records/track_3_optimization/modal_results.jsonl.

Usage:
    python3 notes/summarize_results.py [name_pattern]
"""
import json
import sys
from pathlib import Path


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    rows = []
    p = Path("records/track_3_optimization/modal_results.jsonl")
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows = [r for r in rows if pattern in r.get("name", "")]
    print(f"# rows matching '{pattern}': {len(rows)}")
    print()

    hit_rows = [r for r in rows if r.get("target_step") is not None and r.get("returncode") == 0]
    hit_rows.sort(key=lambda r: r["target_step"])
    print(f"## Configs that hit target_loss (sorted by target_step)")
    for r in hit_rows[:20]:
        cfg = r.get("config", {})
        line = f" K={r['target_step']:>4d} final={r.get('final_val_loss'):.5f}  {r.get('name', '')[:120]}"
        print(line)
    print()

    miss_rows = [r for r in rows if r.get("target_step") is None and r.get("final_val_loss") is not None and r.get("returncode") == 0]
    miss_rows = [r for r in miss_rows if r.get("final_val_loss", 99) < 5.0]
    miss_rows.sort(key=lambda r: r.get("final_val_loss", 99))
    print(f"## Configs that did not hit (sorted by final val_loss, top 20)")
    for r in miss_rows[:20]:
        line = f" final={r.get('final_val_loss'):.5f} step={r.get('final_step')}  {r.get('name', '')[:120]}"
        print(line)
    print()

    err_rows = [r for r in rows if r.get("returncode") != 0]
    print(f"## Failed runs ({len(err_rows)})")
    for r in err_rows[-5:]:
        print(" ", r.get("name", ""))


if __name__ == "__main__":
    main()
