#!/usr/bin/env python3
"""仅使用验证集为已选模型校准置信度阈值；不改变AP或模型选择。"""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from steel_defect.data import CLASSES
from steel_defect.evaluation import match_image, read_yolo_labels, summarize
from steel_defect.inference import Detector, file_sha256
from evaluate import write_csv, write_json

GRID = tuple(i / 100 for i in range(5, 76, 5))
RULE = "Maximum micro F1; exact ties: higher micro precision, then higher threshold. Null F1 is ineligible; null precision ranks below defined precision."
AP_NOTE = "Validation-set operating-point calibration only. Does not change official AP (confidence floor .001), does not reselect the model, and does not establish test improvement."


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def selection_key(row):
    """使用固定协议完成selection_key操作。"""
    tp, fp, fn = row["tp"], row["fp"], row["fn"]
    denominator = 2 * tp + fp + fn
    if not denominator:
        return None
    precision = Fraction(tp, tp + fp) if tp + fp else Fraction(-1)
    return Fraction(2 * tp, denominator), precision, Fraction(str(row["threshold"]))


def select_threshold(rows):
    eligible = [row for row in rows if selection_key(row) is not None]
    return max(eligible, key=selection_key) if eligible else None


def calibrate_predictions(images, classes=CLASSES):
    """使用固定协议完成calibrate_predictions操作。"""
    rows, per_class = [], []
    for threshold in GRID:
        filtered_images = []
        for image in images:
            predictions = [p for p in image["detections"] if p["confidence"] > threshold]
            filtered_images.append(dict(ground_truth=image["ground_truth"], detections=predictions,
                                        analysis=match_image(image["ground_truth"], predictions, .5)))
        metrics = summarize(filtered_images, classes)
        rows.append(dict(threshold=threshold, **metrics["overall"]))
        per_class.extend(dict(threshold=threshold, **row) for row in metrics["per_class"])
    selected = select_threshold(rows)
    baseline = next(row for row in rows if row["threshold"] == .25)
    differences = None if selected is None else {
        key: selected[key] - baseline[key] if selected[key] is not None and baseline[key] is not None else None
        for key in ("tp", "fp", "fn", "precision", "recall", "f1")}
    return dict(grid=list(GRID), selection_rule=RULE, selected=selected,
                baseline_confidence_025=baseline, selected_minus_025=differences,
                curve=rows, per_class_curve=per_class,
                undefined_policy="Zero denominators remain null; no all-empty artificial perfect scores.",
                interpretation=AP_NOTE)


def load_val_records(manifest_path):
    """仅校验和读取验证集工件，拒绝指向其他划分的路径。"""
    root = manifest_path.resolve().parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["classes"] != list(CLASSES):
        raise ValueError("Manifest class mapping mismatch")
    report = json.loads((root / "validation_report.json").read_text(encoding="utf-8"))
    if not report.get("g1_passed") or not report.get("export_complete"):
        raise ValueError("Dataset must pass G1 and export before calibration")
    records = [record for record in manifest["records"] if record["split"] == "val"]
    if not records:
        raise ValueError("Validation split is empty")
    paths, evidence = [], []
    for record in records:
        image, label = (root / record[key] for key in ("output_image", "output_label"))

        if not (image.resolve().is_relative_to(root / "验证集") and label.resolve().is_relative_to(root / "验证集")) and not (image.resolve().is_relative_to(root / "images/val") and label.resolve().is_relative_to(root / "labels/val")):
            raise ValueError("标定仅允许读取验证集目录")
        digest = file_sha256(image)
        if digest != record["image_sha256"]:
            raise ValueError(f"Image hash mismatch: {image}")
        boxes = read_yolo_labels(label, record["width"], record["height"], len(CLASSES))
        original = record["boxes"]
        if len(boxes) != len(original) or any(b["class_id"] != g[0] or any(abs(x-y) > 1e-5 for x, y in zip(b["xyxy"], g[1:])) for b, g in zip(boxes, original)):
            raise ValueError(f"Exported GT differs from manifest: {label}")
        paths.append(image.resolve())
        evidence.append(dict(output_image=record["output_image"], image_sha256=digest,
                             output_label=record["output_label"], label_sha256=file_sha256(label),
                             width=record["width"], height=record["height"], ground_truth=boxes))
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate output image in validation manifest")
    return paths, evidence


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Single checkpoint already selected by validation mAP, before calibration")
    parser.add_argument("--manifest", type=Path, required=True, help="Prepared manifest.json (only val artifacts read)")
    parser.add_argument("--config", type=Path, required=True, help="Selected model training configuration")
    parser.add_argument("--output", type=Path, required=True, help="New directory; refuses overwrite")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=320)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.imgsz < 32 or args.imgsz > 2048 or args.imgsz % 32:
        raise ValueError("Invalid imgsz")
    hashes = dict(weights_sha256=file_sha256(args.weights), manifest_sha256=file_sha256(args.manifest),
                  config_sha256=file_sha256(args.config))
    paths, evidence = load_val_records(args.manifest)
    version = importlib.metadata.version("ultralytics")
    if version != "8.3.200":
        raise ValueError(f"Expected verified Ultralytics 8.3.200, got {version}")
    detector = Detector(args.weights, args.device)
    protocol = dict(split="val", imgsz=args.imgsz, prediction_confidence_floor=.001, nms_iou=.7,
                    match_iou=.5, preprocessing="none", grid=list(GRID), selection_rule=RULE,
                    threshold_comparison="confidence > threshold", batch=1, max_det=300,
                    agnostic_nms=False, augment=False, half=False,
                    matching="evaluation.match_image: confidence descending, same-class one-to-one; duplicates FP")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(status="started", started_at=datetime.now(timezone.utc).isoformat(), **hashes,
                    protocol=protocol, protocol_sha256=canonical_hash(protocol), device=detector.device,
                    image_count=len(paths), calibration_script_sha256=file_sha256(Path(__file__)),
                    matching_module_sha256=file_sha256(Path(__file__).resolve().parents[1] / "src/steel_defect/evaluation.py"),
                    inference_module_sha256=file_sha256(Path(__file__).resolve().parents[1] / "src/steel_defect/inference.py"),
                    versions={name: importlib.metadata.version(name) for name in ("ultralytics", "torch", "torchvision", "numpy", "Pillow")},
                    python=platform.python_version(), interpretation=AP_NOTE)
    write_json(output / "calibration_metadata.json", metadata)
    try:
        images = []
        for start in range(0, len(paths), 100):
            batch = detector.predict(paths[start:start + 100], confidence=.001, iou=.7, imgsz=args.imgsz)
            if len(batch["images"]) != len(paths[start:start + 100]):
                raise RuntimeError("Prediction count mismatch")
            for offset, image in enumerate(batch["images"]):
                index = start + offset
                expected = evidence[index]
                if image["status"] != "success":
                    raise RuntimeError(f"Prediction failed for {paths[index]}: {image['error']}")
                if (image["width"], image["height"], image["image_sha256"]) != (expected["width"], expected["height"], expected["image_sha256"]):
                    raise ValueError("Decoded image differs from validated manifest")
                image.update(image_id=f"{index:06d}", input_index=index, **expected)
                images.append(image)
        result = calibrate_predictions(images)

        prediction_path = output / "val_predictions.json"
        write_json(prediction_path, dict(schema_version=1, protocol=protocol, **hashes, images=images))
        metadata["prediction_artifact_sha256"] = file_sha256(prediction_path)
        write_csv(output / "curve.csv", result["curve"])
        write_csv(output / "per_class_curve.csv", result["per_class_curve"])
        metadata.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat())
        write_json(output / "calibration.json", dict(schema_version=1, metadata=metadata, **result))
    except BaseException as exc:
        metadata.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "calibration_metadata.json", metadata)
    print(f"Validation-only confidence calibration completed: {output}")


if __name__ == "__main__":
    main()
