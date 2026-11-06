"""验证集亮度、对比度与噪声压力测试；不修改几何标注。"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
from PIL import Image

from .data import CLASSES
from .inference import file_sha256

CONDITIONS = [dict(name="clean", kind="identity"),
              dict(name="contrast_075", kind="mean_centered_contrast", alpha=.75),
              dict(name="contrast_050", kind="mean_centered_contrast", alpha=.5),
              dict(name="gaussian_noise_10", kind="gaussian_noise", sigma_8bit=10.)]
METRICS = dict(imgsz=320, batch=16, confidence=.25, nms_iou=.7, match_iou=.5,
               ap_confidence_floor=.001, max_det=300, rect=True, half=False,
               augment=False, agnostic_nms=False, workers=0)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sample_seed(sample_id, condition, seed):
    payload = json.dumps([seed, sample_id, condition], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def transform(rgb, condition, seed):
    """对RGB uint8执行逐通道中心对比度变换，四舍五入并裁剪。
    高斯噪声以8位像素为单位且RGB共享；合成噪声不等价于真实纹理鲁棒性。"""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected HxWx3 uint8")
    if condition not in CONDITIONS:
        raise ValueError("Condition is not in the frozen suite")
    if condition["kind"] == "identity":
        return rgb.copy()
    x = rgb.astype(np.float64)
    if condition["kind"] == "mean_centered_contrast":
        mean = x.mean(axis=(0, 1), keepdims=True)
        x = mean + condition["alpha"] * (x - mean)
    else:
        x = x + np.random.Generator(np.random.PCG64(seed)).normal(0, condition["sigma_8bit"], (*x.shape[:2], 1))
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def validate_protocol(protocol):
    if protocol.get("status") != "frozen" or not protocol.get("frozen_at"):
        raise ValueError("Protocol must be frozen before materialization or inference")
    if protocol.get("split") != "val" or protocol.get("conditions") != CONDITIONS:
        raise ValueError("Only the pre-registered validation suite is supported")
    if protocol.get("metrics") != METRICS or protocol.get("rng_seed") != 424242:
        raise ValueError("Frozen metric/RNG settings mismatch")


def val_records(manifest):
    if manifest.get("classes") != list(CLASSES):
        raise ValueError("Class mapping mismatch")
    records = [r for r in manifest["records"] if r["split"] == "val"]
    if not records or len(records) != manifest["split_counts"]["val"]:
        raise ValueError("Validation count mismatch")
    ids = [r["image"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate validation source")
    return records


def labels_text(record):
    w, h = record["width"], record["height"]
    lines = []
    for cls, x1, y1, x2, y2 in record["boxes"]:
        if cls not in range(len(CLASSES)) or not 0 <= x1 < x2 <= w or not 0 <= y1 < y2 <= h:
            raise ValueError("Invalid frozen GT")
        lines.append(f"{cls} {(x1+x2)/2/w:.12g} {(y1+y2)/2/h:.12g} {(x2-x1)/w:.12g} {(y2-y1)/h:.12g}")
    return "\n".join(lines) + ("\n" if lines else "")


def prepare_suite(archive, manifest_path, protocol, output):
    """仅读取ZIP内验证图像，沿用冻结真值；不解压覆盖或修改原始样本。
    压缩包SHA256仅用于来源校验。"""
    validate_protocol(protocol)
    if file_sha256(manifest_path) != protocol["manifest_sha256"]:
        raise ValueError("Manifest hash mismatch")
    if file_sha256(archive) != protocol["archive_sha256"]:
        raise ValueError("Archive hash mismatch")
    manifest = json.loads(Path(manifest_path).read_text())
    records = val_records(manifest)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    evidence = {c["name"]: [] for c in CONDITIONS}
    with zipfile.ZipFile(archive) as source:
        for index, record in enumerate(records):
            relative = PurePosixPath(record["image"])
            if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "IMAGES":
                raise ValueError("Unsafe validation image path")
            raw = source.read("NEU-DET/" + str(relative))
            if digest(raw) != record["image_sha256"]:
                raise ValueError("Source image hash mismatch")
            with Image.open(io.BytesIO(raw)) as image:
                rgb = np.array(image.convert("RGB"))
            if rgb.shape != (record["height"], record["width"], 3):
                raise ValueError("Decoded dimensions mismatch")
            if digest(f"{record['width']}x{record['height']}:RGB:".encode() + rgb.tobytes()) != record["pixel_sha256"]:
                raise ValueError("Decoded pixel hash mismatch")
            for condition in CONDITIONS:
                name = condition["name"]
                directory = output / name
                images, labels = directory / "images/val", directory / "labels/val"
                images.mkdir(parents=True, exist_ok=True)
                labels.mkdir(parents=True, exist_ok=True)
                stem = f"{index:06d}_{record['image_sha256'][:12]}"
                image_path, label_path = images / (stem + ".png"), labels / (stem + ".txt")
                seed = sample_seed(record["image"], name, protocol["rng_seed"])
                corrupted = transform(rgb, condition, seed)
                Image.fromarray(corrupted).save(image_path, format="PNG")
                label_path.write_text(labels_text(record))
                evidence[name].append(dict(sample_id=record["image"], image=str(image_path),
                    source_sha256=digest(raw), source_pixel_sha256=digest(rgb.tobytes()),
                    transformed_pixel_sha256=digest(corrupted.tobytes()), image_sha256=file_sha256(image_path),
                    label_sha256=file_sha256(label_path), sample_seed=seed,
                    ground_truth=[dict(class_id=b[0], xyxy=b[1:]) for b in record["boxes"]]))
    import yaml
    for condition in CONDITIONS:
        directory = output / condition["name"]
        (directory / "data.yaml").write_text(yaml.safe_dump(dict(path=str(directory),
            train="images/val", val="images/val", names=list(CLASSES))))
    write_json(output / "sample_hashes.json", evidence)
    return evidence
