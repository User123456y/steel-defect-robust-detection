"""导出检测结果、标注图像、CSV、JSON和ZIP文件。"""
from __future__ import annotations

import copy
import csv
import json
import tempfile
import zipfile
from pathlib import Path

from .inference import read_image, file_sha256


def export_results(result: dict, paths, output_root) -> dict:
    from PIL import ImageDraw
    paths = [Path(p) for p in paths]
    if len(paths) != len(result["images"]):
        raise ValueError("Input/result counts differ")
    for index, (path, record) in enumerate(zip(paths, result["images"])):
        if record["input_index"] != index or record["filename"] != path.name or record["image_id"] != f"{index:04d}":
            raise ValueError("Input/result order differs")
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="prediction-", dir=root))
    payload = copy.deepcopy(result)
    gallery, originals = [], []
    for path, record in zip(paths, payload["images"]):
        record.update(original_image=None, annotated_image=None, export_error=None)
        if record["status"] != "success":
            continue
        try:
            if file_sha256(path) != record["image_sha256"]:
                raise ValueError("Image content changed after inference")
            image = read_image(path)
            if image.size != (record["width"], record["height"]):
                raise ValueError("Image dimensions changed after inference")

            image.info.clear()
            original_name = f"{record['image_id']}_original.png"
            image.save(folder / original_name)
            draw = ImageDraw.Draw(image)
            for detection in record["detections"]:
                box = detection["xyxy"]
                draw.rectangle(box, outline="#ff3b30", width=2)
                draw.text((box[0], max(0, box[1] - 12)),
                          f"{detection['class_name']} {detection['confidence']:.2f}", fill="#ff3b30")
            name = f"{record['image_id']}_annotated.png"
            image.save(folder / name)
            record["annotated_image"] = name
            record["original_image"] = original_name
            originals.append((str(folder / original_name), f"{record['image_id']} · {record['filename']}"))
            gallery.append((str(folder / name), f"{record['image_id']} · {record['filename']} · {len(record['detections'])} 框"))
        except Exception as exc:
            record["export_error"] = f"{type(exc).__name__}: {exc}"
    json_path, csv_path = folder / "results.json", folder / "detections.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = ["image_id", "input_index", "filename", "image_sha256", "status", "error", "width", "height", "detection_count",
              "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "original_image", "annotated_image", "export_error",
              "model_name", "model_sha256", "confidence_threshold", "nms_iou", "imgsz", "device", "preprocessing", "started_at", "elapsed_ms"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in payload["images"]:
            base = {k: record.get(k) for k in fields if k in record}
            base.update({k: v for k, v in payload["metadata"].items() if k in fields and k != "elapsed_ms"})
            base["detection_count"] = len(record["detections"])
            for detection in record["detections"] or [None]:
                row = dict(base)
                if detection:
                    row.update({k: detection[k] for k in ("class_id", "class_name", "confidence")})
                    row.update(zip(("x1", "y1", "x2", "y2"), detection["xyxy"]))

                row = {k: "'" + v if isinstance(v, str) and v[:1] in {"=", "+", "-", "@", "\t", "\r"} else v for k, v in row.items()}
                writer.writerow(row)
    zip_path = folder / "results.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(folder.iterdir()):
            if item != zip_path:
                archive.write(item, arcname=item.name)
    return dict(result=payload, folder=str(folder), gallery=gallery, originals=originals,
                files=[str(json_path), str(csv_path), str(zip_path)])
