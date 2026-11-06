"""准备本地NEU-DET图像与VOC标注。"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from steel_defect.data import DataError, prepare_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-partial", action="store_true", help="fixture mode; skips NEU size/count completeness checks")
    parser.add_argument("--curation", type=Path, help="hash-bound quarantine and indivisible reviewed groups")
    args = parser.parse_args()
    try:
        report = prepare_dataset(args.source, args.output, args.seed, args.allow_partial, args.curation)
    except (DataError, OSError) as exc:
        parser.exit(2, f"Data preparation failed: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
