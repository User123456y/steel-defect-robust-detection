#!/usr/bin/env python3
"""使用训练集定义对比度阈值，对验证集预测进行误差分析。"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantile(values, q=.25):
    values = sorted(values)
    if not values or not 0 <= q <= 1:
        raise ValueError("Quantile requires values and q in [0,1]")
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def local_contrast(gray, box, padding=5):
    """计算框内均值与外围5像素环带均值之差的绝对值除以255；无环带返回空值。"""
    from PIL import ImageStat
    x1, y1, x2, y2 = box
    if any(int(v) != v for v in box) or not (0 <= x1 < x2 <= gray.width and 0 <= y1 < y2 <= gray.height):
        raise ValueError("Expected valid integer pixel-edge box")
    roi = tuple(map(int, box))
    outer = (max(0, x1-padding), max(0, y1-padding),
             min(gray.width, x2+padding), min(gray.height, y2+padding))
    roi_n = (x2-x1)*(y2-y1)
    ring_n = (outer[2]-outer[0])*(outer[3]-outer[1]) - roi_n
    if ring_n < 1:
        return None
    roi_sum = ImageStat.Stat(gray.crop(roi)).sum[0]
    outer_sum = ImageStat.Stat(gray.crop(outer)).sum[0]
    return abs(roi_sum / roi_n - (outer_sum-roi_sum) / ring_n) / 255


def check_predictions(payload, records, manifest_hash):
    if payload.get("metadata", {}).get("split") != "val":
        raise ValueError("Only val predictions are allowed")
    if payload["metadata"].get("manifest_sha256") != manifest_hash:
        raise ValueError("Prediction manifest hash mismatch")
    index = {}
    expected = {r["output_image"]: r for r in records if r["split"] == "val"}
    for image in payload["images"]:
        name = image["output_image"]
        if name in index or name not in expected:
            raise ValueError("Duplicate or unexpected validation image")
        record = expected[name]
        gt = image["ground_truth"]
        if len(gt) != len(record["boxes"]) or any(
            g["class_id"] != b[0] or any(abs(x-y)>1e-5 for x,y in zip(g["xyxy"], b[1:]))
            for g,b in zip(gt,record["boxes"])):
            raise ValueError("Prediction GT differs from frozen manifest")
        if image.get("image_sha256") != record["image_sha256"]:
            raise ValueError("Prediction image hash mismatch")
        index[name] = image
    if set(index) != set(expected):
        raise ValueError("Incomplete validation predictions")
    return index


def strata(rows, classes):
    result = []
    for cls, name in enumerate(classes):
        for band in ("low", "other", "undefined"):
            subset = [r for r in rows if r["class_id"] == cls and r["band"] == band]
            valid = [r for r in subset if r["missed"] is not None]
            fn = sum(r["missed"] for r in valid)
            result.append(dict(class_id=cls, class_name=name, band=band, gt_boxes=len(subset),
                images=len({r["output_image"] for r in subset}), evaluated_boxes=len(valid),
                false_negatives=fn if valid else None, fn_fraction=fn/len(valid) if valid else None))
    return result


def run(data, output, predictions=None):
    from PIL import Image, ImageStat, ImageDraw
    data, output = Path(data).resolve(), Path(output).resolve()
    root = data.parent
    manifest_path = root/"manifest.json"
    manifest_hash = sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    validation = json.loads((root/"validation_report.json").read_text())
    if not validation.get("g1_passed") or not validation.get("export_complete"):
        raise ValueError("G1-complete exported data required")
    records = [r for r in manifest["records"] if r["split"] in ("train", "val")]
    prediction_data = json.loads(Path(predictions).read_text()) if predictions else None
    pred = check_predictions(prediction_data, records, manifest_hash) if predictions else {}
    boxes, image_rows = [], []
    for record in records:
        path = (root/record["output_image"]).resolve()
        if not path.is_relative_to(root) or sha(path) != record["image_sha256"]:
            raise ValueError("Image path/hash mismatch")
        with Image.open(path) as im:
            gray = im.convert("L")
            if gray.size != (record["width"], record["height"]):
                raise ValueError("Image dimensions differ from manifest")
            image_rows.append(dict(output_image=record["output_image"], source_image=record["image"],
                                   split=record["split"], global_rms_aux=ImageStat.Stat(gray).stddev[0]/255))
            errors = pred.get(record["output_image"], {}).get("analysis", {})
            fn = set(errors.get("false_negative_indices", []))
            for i, box in enumerate(record["boxes"]):
                boxes.append(dict(output_image=record["output_image"], source_image=record["image"],
                    split=record["split"], gt_index=i, class_id=box[0], xyxy=box[1:],
                    contrast=local_contrast(gray, box[1:]),
                    missed=(i in fn) if record["output_image"] in pred else None))
    threshold = quantile([r["contrast"] for r in boxes if r["split"] == "train" and r["contrast"] is not None])
    for row in boxes:
        row["band"] = "undefined" if row["contrast"] is None else ("low" if row["contrast"] <= threshold else "other")
    train = [r for r in boxes if r["split"] == "train"]
    val = [r for r in boxes if r["split"] == "val"]
    summary = dict(status="validation_errors_analyzed" if predictions else "train_threshold_frozen_pending_baseline_val",
        manifest_sha256=manifest_hash, analyzer_sha256=sha(__file__),
        primary_measure="abs(mean grayscale in GT bbox - mean grayscale in clipped 5px outer ring)/255",
        ring_padding_pixels=5, undefined_rule="No surrounding ring -> null; exclude from threshold, retain separate count",
        threshold=dict(value=threshold, rule="train GT box pooled 25th percentile, linear interpolation; val low <= threshold",
                       train_defined_boxes=sum(r["contrast"] is not None for r in train)),
        auxiliary_measure="whole-image grayscale RMS/255; descriptive only, not threshold",
        counts=dict(images=dict(Counter(r["split"] for r in image_rows)), boxes=dict(Counter(r["split"] for r in boxes))),
        train_by_class=strata(train, manifest["classes"]), val_by_class=strata(val, manifest["classes"]),
        prediction_sha256=sha(predictions) if predictions else None,
        protocol=prediction_data["metadata"]["protocol"] if predictions else None,
        facts="Threshold uses train only; error counts use val at recorded operating point; no test image or model inference used.",
        interpretation_limits=["ROI/ring contrast is a proxy, not proof of target visibility or causal mechanism.",
            "Bounding boxes contain background; ring may overlap another defect; full-image boxes have undefined local contrast.",
            "Compare within class first: pooled low/high rates may be confounded by category.",
            "Related images/boxes are not independent; no statistical-significance claim.",
            "Texture interference requires inspecting FP/confusion examples; no texture score is treated as ground truth."],
        hypothesis="Whether deterministic contrast preprocessing helps remains undecided until baseline validation evidence is reviewed.")
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("analysis.json",summary),("box_contrast_errors.json",boxes),("image_statistics.json",image_rows)):
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    with (output/"val_contrast_by_class.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(summary["val_by_class"][0]));writer.writeheader();writer.writerows(summary["val_by_class"])
    if pred:
        counts=defaultdict(lambda:Counter())
        for row in val:
            if row["missed"]:counts[row["output_image"]]["low_fn" if row["band"]=="low" else "other_fn"]+=1
        for name,item in pred.items():
            counts[name]["fp"]=len(item["analysis"]["false_positive_indices"])
            counts[name]["confusion"]=len(item["analysis"]["class_confusion_candidates"])
        selected=[]
        for criterion in ("low_fn","other_fn","fp","confusion"):
            candidates=sorted(counts,key=lambda name:(-counts[name][criterion],name))
            selected.extend((criterion,name) for name in candidates[:4] if counts[name][criterion]>0)
        samples=[]
        for k,(criterion,name) in enumerate(selected):
            item=pred[name]
            with Image.open(root/name) as im: canvas=im.convert("RGB").resize((600,600))
            draw=ImageDraw.Draw(canvas); sx=600/item["width"];sy=600/item["height"]
            fn=set(item["analysis"]["false_negative_indices"]);fp=set(item["analysis"]["false_positive_indices"])
            for i,gt in enumerate(item["ground_truth"]):
                xy=[v*(sx if j%2==0 else sy) for j,v in enumerate(gt["xyxy"])]
                draw.rectangle(xy,outline="red" if i in fn else "lime",width=3)
                draw.text((xy[0],xy[1]),"GT "+str(gt["class_id"]),fill="red" if i in fn else "lime")
            for i in fp:
                detection=item["detections"][i]
                xy=[v*(sx if j%2==0 else sy) for j,v in enumerate(detection["xyxy"])]
                draw.rectangle(xy,outline="magenta",width=2)
                draw.text((xy[0],xy[1]+12),"FP "+str(detection["class_id"]),fill="magenta")
            target=f"example_{k:02d}_{criterion}.png";canvas.save(output/target)
            samples.append(dict(path=target,criterion=criterion,output_image=name,counts=dict(counts[name])))
        (output/"examples.json").write_text(json.dumps(dict(legend="red=missed GT, green=matched GT, magenta=unmatched prediction; candidate confusion not causal evidence",samples=samples),indent=2)+"\n")
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--predictions",type=Path)
    args=parser.parse_args()
    result=run(args.data,args.output,args.predictions)
    print(json.dumps(dict(status=result["status"],threshold=result["threshold"],counts=result["counts"]),indent=2))
