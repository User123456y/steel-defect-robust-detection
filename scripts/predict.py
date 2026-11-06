#!/usr/bin/env python3
"""单张或批量本地图像检测与结果导出。"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from steel_defect.inference import Detector, InferenceError
from steel_defect.export import export_results
from steel_defect.profile import load_profile, resolve_parameters, attach_profile_metadata


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--profile", type=Path, help="绑定权重SHA的推荐推理配置；显式CLI参数优先")
    parser.add_argument("--confidence", type=float, help="手动覆盖profile推荐置信度")
    parser.add_argument("--iou", type=float, help="手动覆盖profile推荐NMS IoU")
    parser.add_argument("--imgsz", type=int, help="手动覆盖profile推荐输入尺寸")
    parser.add_argument("--output", type=Path, default=Path("runs/predictions"))
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        profile = load_profile(args.profile, args.weights)
        actual = resolve_parameters(profile, confidence=args.confidence, iou=args.iou, imgsz=args.imgsz)
        detector = Detector(args.weights, args.device)
        result = detector.predict(args.images, **actual)
        attach_profile_metadata(result, profile, actual)
        bundle = export_results(result, args.images, args.output)
    except (InferenceError, OSError, ImportError) as exc:
        parser.exit(2, f"推理未完成：{exc}\n")
    print(json.dumps(dict(folder=bundle["folder"], images=bundle["result"]["images"]), ensure_ascii=False, indent=2))
    return 1 if any(r["status"] != "success" or r["export_error"] for r in bundle["result"]["images"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
