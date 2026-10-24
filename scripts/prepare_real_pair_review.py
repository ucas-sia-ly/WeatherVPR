#!/usr/bin/env python3
"""Create an unreviewed, source-bound DarkDriving JSONL review template.

This does not approve samples or infer routes from filename prefixes. Fill in
scene groups from documented route/near-duplicate review and assign each whole
scene to one split. ``automatic`` labels mean automatic screening; use
``manual`` only for records that were actually checked by a person.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from weather_vpr.data.real_pairs import load_review_source


def prepare_review(root: Path | str, output: Path | str, sample_ids=None) -> dict:
    """Write a new template; optionally restrict to explicit known sample IDs."""
    root, rows, manifest_hash = load_review_source(root)
    if sample_ids is not None:
        sample_ids = list(sample_ids)
        if (any(not isinstance(name, str) or not name.strip() for name in sample_ids)
                or len(set(sample_ids)) != len(sample_ids)):
            raise ValueError("sample_ids must contain unique nonempty strings")
        missing = set(sample_ids) - {row["sample_id"] for row in rows}
        if missing:
            raise ValueError(f"Unknown sample IDs: {sorted(missing)}")
        selected = set(sample_ids)
        rows = [row for row in rows if row["sample_id"] in selected]
    if not rows:
        raise ValueError("Cannot create an empty review template")
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting completed review work.
    with output.open("x", encoding="utf-8") as stream:
        for row in rows:
            review = {
                "sample_id": row["sample_id"], "scene_group": None, "split": None,
                "pair_valid": None, "visibility_valid": None,
                "review_status": "unreviewed", "reject_reason": None,
                "source_manifest_sha256": manifest_hash,
                "day_sha256": row["day_sha256"], "night_sha256": row["night_sha256"],
                "difficulty": None, "visibility_status": None, "review_notes": None,
            }
            stream.write(json.dumps(review, ensure_ascii=False, allow_nan=False) + "\n")
    return {"review_manifest": str(output), "pairs": len(rows), "accepted_pairs": 0,
            "source_manifest_sha256": manifest_hash}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/DarkDriving_Base"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-ids", type=Path,
                        help="Optional text file of explicit sample IDs, one per line")
    args = parser.parse_args()
    sample_ids = (args.sample_ids.read_text(encoding="utf-8").splitlines()
                  if args.sample_ids else None)
    print(json.dumps(prepare_review(args.root, args.output, sample_ids), indent=2))


if __name__ == "__main__":
    main()
