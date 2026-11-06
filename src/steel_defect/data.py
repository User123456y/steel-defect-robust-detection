"""本地VOC标注校验、重复样本隔离与分层数据划分。"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ("crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches")


class DataError(ValueError):
    pass


def parse_voc(path: Path) -> dict:
    """将VOC从1开始的闭区间坐标转换为从0开始的边界坐标。"""
    try:
        root = ET.parse(path).getroot()
        width, height = (int(root.findtext(f"size/{key}", "")) for key in ("width", "height"))
        filename = root.findtext("filename", "").strip()
        if width <= 0 or height <= 0 or not filename or Path(filename).name != filename or "\\" in filename:
            raise ValueError("invalid dimensions or image filename")
        stem = Path(filename).stem if Path(filename).suffix else filename
        match = re.fullmatch(r"(.+)_([1-9][0-9]*)", stem)
        if not match or match[1] not in CLASSES or (Path(filename).suffix and Path(filename).suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}):
            raise ValueError("expected canonical NEU filename: class_positive-index[.image-extension]")
        acquisition_class = CLASSES.index(match[1])
        boxes = []
        for obj in root.findall("object"):
            name = obj.findtext("name", "").strip()
            if name not in CLASSES:
                raise ValueError(f"unknown class: {name!r}")
            values = [float(obj.findtext(f"bndbox/{key}", "")) for key in ("xmin", "ymin", "xmax", "ymax")]
            x1, y1, x2, y2 = values
            if not all(math.isfinite(v) and v.is_integer() for v in values):
                raise ValueError("VOC coordinates must be finite integers")
            if not (1 <= x1 <= x2 <= width and 1 <= y1 <= y2 <= height):
                raise ValueError(f"invalid VOC box: {values}")
            boxes.append([CLASSES.index(name), int(x1 - 1), int(y1 - 1), int(x2), int(y2)])
        if not boxes:
            raise ValueError("expected nonempty NEU annotation")


        counts = Counter(tuple(box) for box in boxes)
        return dict(filename=filename, width=width, height=height, class_id=acquisition_class,
                    raw_box_count=len(boxes),
                    removed_duplicate_count=sum(count - 1 for count in counts.values()),
                    removed_duplicates=[dict(box=list(box), removed=count - 1)
                                        for box, count in sorted(counts.items()) if count > 1],
                    boxes=[list(box) for box in sorted(counts)])
    except (ET.ParseError, ValueError, OSError) as exc:
        raise DataError(f"{path}: {exc}") from exc


def resolve_image(filename: str, annotation: Path, image_index: dict, stem_index: dict) -> Path:
    """安全匹配XML文件名；无扩展名时要求文件主名唯一。"""
    stem = Path(filename).stem if Path(filename).suffix else filename
    if Path(filename).name != filename or "\\" in filename or annotation.stem != stem:
        raise DataError(f"{annotation}: XML/image filename stem mismatch")
    matches = image_index.get(filename, []) if Path(filename).suffix else stem_index.get(stem, [])
    if len(matches) != 1:
        raise DataError(f"{annotation}: image missing or filename ambiguous")
    return matches[0]


def split_records(records: list[dict], seed: int = 42) -> dict[str, str]:
    """保持审核样本组不可拆分，同时平衡采集类别比例。"""
    groups, pixels = defaultdict(list), defaultdict(list)
    for record in records:
        pixels[record["pixel_sha256"]].append(record)
        groups[record.get("group_id", record["pixel_sha256"])].append(record)
    for digest, members in pixels.items():
        if len({json.dumps(r["boxes"]) for r in members}) != 1:
            raise DataError(f"identical pixels have conflicting annotations: {digest}")
        if len({r.get("group_id", digest) for r in members}) != 1:
            raise DataError("identical pixels must share one split group")
    totals = Counter(r["class_id"] for r in records)
    targets = {c: [round(n * .7), round(n * .85) - round(n * .7), n - round(n * .85)]
               for c, n in totals.items()}
    actual = {c: [0, 0, 0] for c in totals}
    entries = sorted(groups.items(), key=lambda item: (-len(item[1]),
        hashlib.sha256(f"{seed}:{item[0]}".encode()).digest()))
    result = {}
    for group_id, members in entries:
        counts = Counter(r["class_id"] for r in members)
        def cost(i):
            return sum(((actual[c][i] + n - targets[c][i]) ** 2 -
                        (actual[c][i] - targets[c][i]) ** 2) / max(targets[c][i], 1)
                       for c, n in counts.items())
        index = min(range(3), key=lambda i: (cost(i), i))
        for c, n in counts.items():
            actual[c][index] += n
        for r in members:
            result[r["pixel_sha256"]] = ("train", "val", "test")[index]
    return result


def source_fingerprint(records):
    items = sorted((r["image"], r["image_sha256"], r["annotation"], r["annotation_sha256"])
                   for r in records)
    return hashlib.sha256(json.dumps(items, separators=(",", ":")).encode()).hexdigest()


def apply_curation(records, review):
    """将审核决策绑定原始数据摘要，保持原图与标注不变。"""
    if review.get("source_fingerprint") != source_fingerprint(records):
        raise DataError("curation source fingerprint mismatch")
    if not review.get("reviewer") or not review.get("limitations"):
        raise DataError("curation requires reviewer and limitations")
    index = {r["image"]: r for r in records}
    excluded = {}
    for entry in review.get("quarantine", []):
        name = entry["image"]
        if name not in index or name in excluded or not entry.get("reason"):
            raise DataError("invalid quarantine decision")
        excluded[name] = entry["reason"]

    pixels = defaultdict(list)
    for r in records:
        pixels[r["pixel_sha256"]].append(r)
    for members in pixels.values():
        flags = [r["image"] in excluded for r in members]
        if any(flags) and not all(flags):
            raise DataError("quarantine must include complete identical-pixel group")
    kept = [dict(r, group_id=r["pixel_sha256"]) for r in records if r["image"] not in excluded]
    remaining = {r["image"]: r for r in kept}
    seen = set()
    for entry in review.get("groups", []):
        names = entry.get("images", [])
        if len(names) < 2 or not entry.get("reason") or len(set(names)) != len(names):
            raise DataError("invalid reviewed group")
        if any(n not in remaining or n in seen for n in names):
            raise DataError("unknown, excluded or overlapping reviewed group member")
        group_id = hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()
        for name in names:
            remaining[name]["group_id"] = group_id
        seen.update(names)
    return kept, excluded


def inspect_image(path: Path) -> tuple[int, int, str, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise DataError("Pillow required for decoding and duplicate checks") from exc
    try:
        with Image.open(path) as image:
            image.load()
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("multi-frame image unsupported")
            rgb = image.convert("RGB")
            digest = hashlib.sha256(f"{rgb.width}x{rgb.height}:RGB:".encode() + rgb.tobytes()).hexdigest()
            small = rgb.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
            pixels = list(small.getdata())
            dhash = sum((pixels[y * 9 + x] > pixels[y * 9 + x + 1]) << (y * 8 + x)
                        for y in range(8) for x in range(8))
            return rgb.width, rgb.height, digest, dhash
    except (OSError, ValueError) as exc:
        raise DataError(f"cannot decode {path}: {exc}") from exc


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_dataset(source: Path, output: Path, seed: int = 42, allow_partial: bool = False, curation: Path | None = None) -> dict:
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir():
        raise DataError(f"missing source directory: {source}")
    if source == output or source in output.parents or output in source.parents:
        raise DataError("source/output must be separate, non-nested directories")
    if output.exists():
        raise DataError(f"refusing to overwrite: {output}")
    files = sorted(source.rglob("*"))
    if any(p.is_symlink() for p in files):
        raise DataError("source symlinks are unsupported")
    images = [p for p in files if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    annotations = [p for p in files if p.is_file() and p.suffix.lower() == ".xml"]
    image_index = defaultdict(list)
    stem_index = defaultdict(list)
    for image in images:
        image_index[image.name].append(image)
        stem_index[image.stem].append(image)
    if not annotations:
        raise DataError("no XML annotations found")
    records, used = [], set()
    for annotation in annotations:
        record = parse_voc(annotation)
        image = resolve_image(record.pop("filename"), annotation, image_index, stem_index)
        if image in used:
            raise DataError(f"multiple annotations for {image}")
        used.add(image)
        width, height, digest, dhash = inspect_image(image)
        if (width, height) != (record["width"], record["height"]):
            raise DataError(f"XML/image dimensions disagree: {image}")
        if not allow_partial and (width, height) != (200, 200):
            raise DataError(f"NEU image must be 200x200: {image}")
        record.update(image=str(image.relative_to(source)), annotation=str(annotation.relative_to(source)),
                      image_sha256=_hash(image), annotation_sha256=_hash(annotation),
                      pixel_sha256=digest, dhash=f"{dhash:016x}")
        records.append(record)
    if used != set(images):
        raise DataError("unannotated images present")
    counts = Counter(r["class_id"] for r in records)
    if not allow_partial and (len(records) != 1800 or any(counts[i] != 300 for i in range(6))):
        raise DataError(f"expected 1800 images, 300 per class; got {dict(counts)}")
    raw_records = records
    fingerprint = source_fingerprint(raw_records)
    review = json.loads(Path(curation).read_text()) if curation else None
    if review and allow_partial:
        raise DataError("curation cannot certify partial fixtures")
    records, excluded = apply_curation(raw_records, review) if review else (records, {})
    assignments = split_records(records, seed)
    groups = defaultdict(list)
    for record in records:
        groups[record["pixel_sha256"]].append(record)
        record["split"] = assignments[record["pixel_sha256"]]
    unique = [groups[digest][0] for digest in sorted(groups)]
    pairs, unresolved = [], []
    for i, left in enumerate(unique):
        for right in unique[i + 1:]:
            distance = (int(left["dhash"], 16) ^ int(right["dhash"], 16)).bit_count()
            if distance <= 4:
                pair = dict(left=left["image"], right=right["image"], distance=distance)
                pairs.append(pair)
                if not review or left.get("group_id") != right.get("group_id"):
                    unresolved.append(pair)
    split_class_counts = {split: dict(Counter(r["class_id"] for r in records if r["split"] == split))
                          for split in ("train", "val", "test")}
    if review and any(any(counts.get(c, 0) == 0 for c in range(6)) for counts in split_class_counts.values()):
        raise DataError("curated split is missing an acquisition category")
    passed = bool(review) and not unresolved and not allow_partial
    report = dict(status="prepared_reviewed" if passed else ("blocked_near_duplicates" if unresolved else "prepared_pending_manual_review"),
                  g1_passed=passed, partial_fixture_mode=allow_partial, image_count=len(records),
                  raw_image_count=len(raw_records), raw_class_counts=dict(counts),
                  raw_source_box_count=sum(r["raw_box_count"] for r in raw_records),
                  retained_raw_box_count=sum(r["raw_box_count"] for r in records),
                  canonical_box_count=sum(len(r["boxes"]) for r in records),
                  removed_duplicate_count=sum(r["removed_duplicate_count"] for r in records),
                  split_box_counts={split: sum(len(r["boxes"]) for r in records if r["split"] == split)
                                    for split in ("train", "val", "test")},
                  class_counts=dict(Counter(r["class_id"] for r in records)),
                  source_fingerprint=fingerprint, quarantine=excluded,
                  split_class_counts=split_class_counts,
                  near_duplicate_candidates=pairs, unresolved_near_duplicate_candidates=unresolved,
                  exact_duplicate_groups=[[r["image"] for r in group] for group in groups.values() if len(group) > 1],
                  near_duplicate_method="64-bit grayscale dHash <=4, heuristic only; reviewed candidate components remain indivisible",
                  limitations=review.get("limitations", []) if review else [],
                  pending=[] if passed else ["Visual review and source/curation documentation required"])
    manifest = dict(schema_version=4,
                    box_normalization="Within each image, deduplicate exact (class_id,x1,y1,x2,y2) tuples after lossless VOC conversion; never merge different classes or coordinates; raw XML unchanged", source_fingerprint=fingerprint,
                    curation_sha256=_hash(Path(curation)) if curation else None,
                    class_id_semantics="acquisition category from canonical filename; stratification only; boxes retain object class IDs", source=str(source), seed=seed, ratios=[.7, .15, .15],
                    classes=list(CLASSES), coordinate_convention="VOC 1-based inclusive -> zero-based edges",
                    split_counts=dict(Counter(r["split"] for r in records)), records=records)
    output.mkdir(parents=True)
    _json(output / "validation_report.json", dict(report, g1_passed=False, export_complete=False))
    _json(output / "manifest.json", manifest)
    if review:
        _json(output / "curation.json", review)
    if unresolved:
        raise DataError(f"{len(unresolved)} unresolved near-duplicate candidates; see {output / 'validation_report.json'}; no YOLO export")
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    for index, record in enumerate(records):
        image = source / record["image"]
        name = f"{index:06d}_{record['image_sha256'][:12]}"
        record["output_image"] = f"images/{record['split']}/{name}{image.suffix.lower()}"
        record["output_label"] = f"labels/{record['split']}/{name}.txt"
        shutil.copyfile(image, output / record["output_image"])
        lines = []
        for cls, x1, y1, x2, y2 in record["boxes"]:
            values = ((x1 + x2) / (2 * record["width"]), (y1 + y2) / (2 * record["height"]),
                      (x2 - x1) / record["width"], (y2 - y1) / record["height"])
            lines.append(str(cls) + " " + " ".join(f"{v:.10f}" for v in values))
        (output / record["output_label"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    _json(output / "manifest.json", manifest)
    (output / "data.yaml").write_text("path: " + json.dumps(str(output), ensure_ascii=False) +
        "\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n" +
        "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASSES)), encoding="utf-8")
    report["export_complete"] = True
    _json(output / "validation_report.json", report)
    return report
