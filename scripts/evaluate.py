#!/usr/bin/env python3
"""验证集或已冻结选择方案的测试集评估。"""
import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from steel_defect.data import CLASSES
from steel_defect.inference import Detector, file_sha256
from steel_defect.evaluation import check_test_gate, match_image, official_metrics, read_yolo_labels, summarize


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="Prepared data.yaml; manifest.json must be beside it")
    parser.add_argument("--config", type=Path, required=True, help="Training configuration retained for provenance")
    parser.add_argument("--output", type=Path, required=True, help="New directory; refuses overwrite")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=16, help="Official validator batch; fixed-point Detector uses batch 1")
    parser.add_argument("--confidence", type=float, default=.25)
    parser.add_argument("--nms-iou", type=float, default=.7)
    parser.add_argument("--match-iou", type=float, default=.5)
    parser.add_argument("--preprocessing", choices=["none"], default="none", help="Reserved hook; unsupported transforms rejected")
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--frozen-selection", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    import math
    if args.batch < 1 or args.imgsz < 32 or args.imgsz > 2048 or args.imgsz % 32:
        raise ValueError("Invalid batch/imgsz")
    if any(not math.isfinite(v) or not 0 < v <= 1 for v in (args.confidence, args.nms_iou, args.match_iou)):
        raise ValueError("All thresholds must be in (0, 1]")
    root = args.data.resolve().parent
    manifest_path = root / "manifest.json"
    hashes = {"weights_sha256": file_sha256(args.weights), "manifest_sha256": file_sha256(manifest_path),
              "config_sha256": file_sha256(args.config)}
    protocol = dict(imgsz=args.imgsz, confidence=args.confidence, nms_iou=args.nms_iou,
                    match_iou=args.match_iou, preprocessing=args.preprocessing)
    frozen = json.loads(args.frozen_selection.read_text()) if args.frozen_selection else None
    check_test_gate(args.split, args.final_test, frozen, dict(hashes, protocol=protocol))
    manifest = json.loads(manifest_path.read_text())
    if manifest["classes"] != list(CLASSES):
        raise ValueError("Manifest class mapping mismatch")
    report = json.loads((root / "validation_report.json").read_text())
    if not report.get("g1_passed") or not report.get("export_complete"):
        raise ValueError("Dataset must pass G1 and export before evaluation")
    records = [r for r in manifest["records"] if r["split"] == args.split]
    if not records:
        raise ValueError("Selected split is empty")
    paths, ground_truth, evidence = [], [], []
    for record in records:
        image, label = (root / record[key] for key in ("output_image", "output_label"))
        if not image.resolve().is_relative_to(root) or not label.resolve().is_relative_to(root):
            raise ValueError("Manifest path escapes dataset root")
        if file_sha256(image) != record["image_sha256"]:
            raise ValueError(f"Image hash mismatch: {image}")
        boxes = read_yolo_labels(label, record["width"], record["height"], len(CLASSES))
        original = record["boxes"]
        if len(boxes) != len(original) or any(b["class_id"] != g[0] or any(abs(x-y) > 1e-5 for x, y in zip(b["xyxy"], g[1:])) for b, g in zip(boxes, original)):
            raise ValueError(f"Exported GT differs from manifest: {label}")
        paths.append(image.resolve())
        ground_truth.append(boxes)
        evidence.append(dict(output_image=record["output_image"], image_sha256=file_sha256(image),
                             output_label=record["output_label"], label_sha256=file_sha256(label)))
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate output image in manifest")
    version = importlib.metadata.version("ultralytics")
    if version != "8.3.200":
        raise ValueError(f"Expected verified Ultralytics 8.3.200, got {version}")
    detector = Detector(args.weights, args.device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(status="started", started_at=datetime.now(timezone.utc).isoformat(), **hashes,
                    data_yaml_sha256=file_sha256(args.data), protocol=protocol, split=args.split,
                    image_count=len(records), device=detector.device, official_batch=args.batch,
                    fixed_point_batch=1, official_ap_confidence_floor=.001, max_det=300,
                    official_rect=True, half=False, augment=False, agnostic_nms=False,
                    versions={name: importlib.metadata.version(name) for name in ("ultralytics", "torch", "torchvision", "numpy", "Pillow")},
                    python=platform.python_version(),
                    evaluator_sha256=file_sha256(Path(__file__)),
                    matching_module_sha256=file_sha256(Path(__file__).resolve().parents[1] / "src/steel_defect/evaluation.py"),
                    frozen_selection=frozen,
                    frozen_selection_sha256=file_sha256(args.frozen_selection) if args.frozen_selection else None)
    write_json(output / "evaluation_metadata.json", metadata)
    try:

        import yaml
        image_list = output / "evaluated_images.txt"
        image_list.write_text("\n".join(map(str, paths)) + "\n", encoding="utf-8")
        local_data = output / "evaluation_data.yaml"
        local_data.write_text(yaml.safe_dump(dict(path=str(root), train=str(image_list), val=str(image_list),
                                                  test=str(image_list), names=list(CLASSES))), encoding="utf-8")
        metadata["evaluation_data_yaml_sha256"] = file_sha256(local_data)
        official = detector.model.val(data=str(local_data), split=args.split, imgsz=args.imgsz,
                     batch=args.batch, device=detector.device, conf=.001, iou=args.nms_iou,
                     max_det=300, rect=True, half=False, augment=False, agnostic_nms=False,
                     workers=0, plots=True, save_json=False, project=str(output), name="official", exist_ok=False)
        images = []
        for start in range(0, len(paths), 100):
            batch = detector.predict(paths[start:start + 100], confidence=args.confidence,
                                     iou=args.nms_iou, imgsz=args.imgsz)
            for offset, image in enumerate(batch["images"]):
                i = start + offset
                if image["status"] != "success":
                    raise RuntimeError(f"Prediction failed for {paths[i]}: {image['error']}")
                if (image["width"], image["height"]) != (records[i]["width"], records[i]["height"]):
                    raise ValueError("Decoded image dimensions differ from manifest")
                image.update(image_id=f"{i:06d}", input_index=i, **evidence[i], ground_truth=ground_truth[i])
                image["analysis"] = match_image(ground_truth[i], image["detections"], args.match_iou)
                images.append(image)
        fixed = summarize(images, CLASSES)
        ap = official_metrics(official, CLASSES, [row["gt_count"] for row in fixed["per_class"]])
        result = dict(schema_version=1, official=ap, fixed_operating_point=fixed,
                      matching="Confidence descending; highest-IoU unmatched same-class GT; stable index ties; duplicate predictions FP",
                      protocol=protocol)
        write_json(output / "metrics.json", result)
        write_json(output / "predictions.json", dict(metadata={k: v for k, v in metadata.items() if k != "status"}, images=images))
        write_csv(output / "fixed_metrics.csv", [dict(class_id="all", class_name="all", **fixed["overall"])] + fixed["per_class"])
        write_csv(output / "official_metrics.csv", [dict(class_id="all", class_name="all", gt_count=sum(r["gt_count"] for r in ap["per_class"]), precision_best_f1=None, recall_best_f1=None, ap50=ap["map50"], ap50_95=ap["map50_95"])] + ap["per_class"])
        errors = [dict(image_id=i["image_id"], output_image=i["output_image"], fp=len(i["analysis"]["false_positive_indices"]), fn=len(i["analysis"]["false_negative_indices"]), class_confusion_candidates=len(i["analysis"]["class_confusion_candidates"]), localization_candidates=len(i["analysis"]["localization_candidates"])) for i in images]
        write_csv(output / "image_errors.csv", errors)
        write_json(output / "error_images.json", [row for row in errors if row["fp"] or row["fn"]])
        metadata["status"] = "completed"
    except BaseException as exc:
        metadata.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "evaluation_metadata.json", metadata)
    print(f"Evaluation completed: {output}")


if __name__ == "__main__":
    main()
