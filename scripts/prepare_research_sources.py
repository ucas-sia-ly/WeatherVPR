#!/usr/bin/env python3
"""Audit local training sources and create provenance-bound research manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from weather_vpr.data.research_sources import prepare_research_sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory; existing output is never overwritten")
    parser.add_argument("--qwen-manifest", type=Path)
    parser.add_argument("--real-review", type=Path, help="Explicit source-bound DarkDriving review; unreviewed rows excluded")
    parser.add_argument("--include-bdd", action="store_true", help="Original official BDD train-night annotation intersection only")
    parser.add_argument("--include-nightcity", action="store_true", help="Original NightCity train-label/image intersection only")
    parser.add_argument("--include-gsv", action="store_true", help="Build annotation-backed G1/G2 independent Day source pools")
    parser.add_argument("--gsv-new-places", type=int, default=5000)
    parser.add_argument("--gsv-views-per-place", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-images", action="store_true", help="Strict decode and SHA every selected manifest image (can be expensive)")
    parser.add_argument("--exclude-places", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    report = prepare_research_sources(args.project_root, args.output,
        qwen_manifest=args.qwen_manifest, real_review=args.real_review,
        include_bdd=args.include_bdd, include_nightcity=args.include_nightcity,
        include_gsv=args.include_gsv, gsv_new_places=args.gsv_new_places,
        gsv_views_per_place=args.gsv_views_per_place, seed=args.seed,
        verify_images=args.verify_images, extra_exclusions=args.exclude_places)
    print(json.dumps({"output": str(args.output.resolve()), "source_counts": report["source_counts"],
                      "gsv": report["sources"].get("gsv", {}).get("g2"), "training_started": False},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
