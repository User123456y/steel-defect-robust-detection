"""固定阈值检测匹配、分类指标与Ultralytics官方AP结果映射。"""
from __future__ import annotations

import math
from pathlib import Path


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def read_yolo_labels(path, width, height, class_count):
    """将归一化YOLO标注转换为原图像素边界坐标。"""
    boxes = []
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions")
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if len(values) != 5 or not all(math.isfinite(v) for v in values):
            raise ValueError(f"Malformed YOLO label: {path}")
        cls, cx, cy, w, h = values
        if cls != int(cls) or not 0 <= cls < class_count or not (0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"Invalid label class/size: {path}")
        normalized = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        if any(v < -1e-8 or v > 1 + 1e-8 for v in normalized):
            raise ValueError(f"Out-of-bounds label: {path}")
        xyxy = [min(1, max(0, v)) * size for v, size in zip(normalized, [width, height, width, height])]
        boxes.append(dict(class_id=int(cls), xyxy=xyxy))
    return boxes


def match_image(ground_truth, predictions, match_iou=.5):
    """按置信度降序匹配同类别最高IoU的未使用真值框，并列时保持输入顺序。
    跨类别重叠只用于诊断，不改变TP、FP、FN；此规则不替代官方AP计算。"""
    if not math.isfinite(match_iou) or not 0 < match_iou <= 1:
        raise ValueError("match_iou must be in (0, 1]")
    used, matches, fp = set(), [], []
    for pi in sorted(range(len(predictions)), key=lambda i: (-predictions[i]["confidence"], i)):
        prediction = predictions[pi]
        candidates = [(box_iou(prediction["xyxy"], gt["xyxy"]), gi)
                      for gi, gt in enumerate(ground_truth)
                      if gi not in used and gt["class_id"] == prediction["class_id"]]
        best = max(candidates, key=lambda item: (item[0], -item[1]), default=(0, -1))
        if best[0] >= match_iou:
            used.add(best[1])
            matches.append(dict(prediction_index=pi, ground_truth_index=best[1], iou=best[0]))
        else:
            fp.append(pi)
    fn = [i for i in range(len(ground_truth)) if i not in used]
    confusion, localization = [], []
    for pi in fp:
        prediction = predictions[pi]
        for gi, gt in enumerate(ground_truth):
            overlap = box_iou(prediction["xyxy"], gt["xyxy"])
            evidence = dict(prediction_index=pi, ground_truth_index=gi, iou=overlap)
            if overlap >= match_iou and gt["class_id"] != prediction["class_id"]:
                confusion.append(evidence)
            elif 0 < overlap < match_iou and gt["class_id"] == prediction["class_id"]:
                localization.append(evidence)
    return dict(matches=matches, false_positive_indices=fp, false_negative_indices=fn,
                class_confusion_candidates=confusion, localization_candidates=localization)


def count_metrics(tp, fp, fn):
    return dict(tp=tp, fp=fp, fn=fn, gt_count=tp + fn, prediction_count=tp + fp,
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None)


def summarize(images, classes):
    counts = [[0, 0, 0] for _ in classes]
    for image in images:
        gt, predictions, analysis = image["ground_truth"], image["detections"], image["analysis"]
        for pair in analysis["matches"]:
            counts[gt[pair["ground_truth_index"]]["class_id"]][0] += 1
        for index in analysis["false_positive_indices"]:
            counts[predictions[index]["class_id"]][1] += 1
        for index in analysis["false_negative_indices"]:
            counts[gt[index]["class_id"]][2] += 1
    return dict(overall=count_metrics(*(sum(row[i] for row in counts) for i in range(3))),
                per_class=[dict(class_id=i, class_name=name, **count_metrics(*counts[i]))
                           for i, name in enumerate(classes)],
                undefined_policy="Zero denominators -> JSON null / blank CSV; F1 uses 2TP/(2TP+FP+FN). Overall is micro-averaged.")


def official_metrics(metrics, classes, gt_counts):
    """按ap_class_index映射Ultralytics 8.3.200各类别指标。
    不使用会为缺失类别填入总体mAP的maps字段。
    来源：https://github.com/ultralytics/ultralytics/blob/v8.3.200/ultralytics/utils/metrics.py"""
    indexed = {int(cls): tuple(map(float, metrics.box.class_result(row)))
               for row, cls in enumerate(metrics.box.ap_class_index)}
    per_class = []
    for cls, name in enumerate(classes):

        p, r, ap50, ap = indexed.get(cls, (None, None, 0.0, 0.0))
        if not gt_counts[cls]:
            p, r, ap50, ap = None, None, None, None
        per_class.append(dict(class_id=cls, class_name=name, gt_count=gt_counts[cls],
                              precision_best_f1=p, recall_best_f1=r, ap50=ap50, ap50_95=ap))
    return dict(map50=float(metrics.box.map50) if sum(gt_counts) else None,
                map50_95=float(metrics.box.map) if sum(gt_counts) else None,
                per_class=per_class,
                precision_recall_note="Official P/R at library-selected best mean F1 operating point, NOT fixed confidence=.25.",
                absent_class_policy="No GT: AP null. GT but no detections: AP zero; unavailable library P/R null.")


def check_test_gate(split, final_test, frozen, expected):
    if split != "test":
        if final_test:
            raise ValueError("--final-test requires --split test")
        return
    if not final_test or not isinstance(frozen, dict):
        raise ValueError("test is locked: require --final-test and --frozen-selection JSON")
    if frozen.get("status") != "frozen" or not frozen.get("selected_on_val") or not frozen.get("frozen_at"):
        raise ValueError("Frozen selection needs status=frozen, selected_on_val and frozen_at")
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise ValueError(f"Frozen selection mismatch: {key}")
