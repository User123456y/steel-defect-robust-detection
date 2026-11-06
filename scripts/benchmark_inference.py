#!/usr/bin/env python3
"""CPU模型加载与复用请求流程基准；不作为GPU加速证据。"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from steel_defect.inference import Detector, InferenceError, file_sha256
from steel_defect.profile import load_profile, resolve_parameters


def paired_schedule(requests, image_count=3):
    """均衡每张图像的请求数与AB、BA顺序，每对包含两个请求。"""
    if requests < 30 or requests % (2 * image_count):
        raise ValueError("requests per flow must be >=30 and a multiple of twice image_count (6)")
    return [(i % image_count, ("A", "B") if i % 2 == 0 else ("B", "A")) for i in range(requests)]


def equivalent(left, right, *, coordinate_atol=.001, confidence_atol=.00001):
    """逐项比较推理结果；无效或失败结果不得判为等价。"""
    try:
        for key in ("model_sha256", "confidence_threshold", "nms_iou", "imgsz", "device", "preprocessing"):
            if left["metadata"][key] != right["metadata"][key]:
                return False
        if len(left["images"]) != 1 or len(right["images"]) != 1:
            return False
        a, b = left["images"][0], right["images"][0]
        for key in ("filename", "image_sha256", "width", "height"):
            if a[key] != b[key]:
                return False
        if a["status"] != "success" or b["status"] != "success":
            return False
        if len(a["detections"]) != len(b["detections"]):
            return False
        order = lambda d: (d["class_id"], d["class_name"], *d["xyxy"], d["confidence"])
        for da, db in zip(sorted(a["detections"], key=order), sorted(b["detections"], key=order)):
            if da["class_id"] != db["class_id"] or da["class_name"] != db["class_name"]:
                return False
            if len(da["xyxy"]) != 4 or len(db["xyxy"]) != 4:
                return False
            for va, vb, tolerance in [(da["confidence"], db["confidence"], confidence_atol),
                                      *[(x, y, coordinate_atol) for x, y in zip(da["xyxy"], db["xyxy"])]]:
                if not (math.isfinite(va) and math.isfinite(vb) and math.isclose(va, vb, rel_tol=1e-5, abs_tol=tolerance)):
                    return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def summarize(milliseconds):
    ordered = sorted(milliseconds)
    return dict(request_count=len(ordered), median_ms=statistics.median(ordered),
                p95_ms=ordered[math.ceil(.95 * len(ordered)) - 1],
                p95_method="nearest rank", total_request_ms=sum(ordered))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--samples-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data/processed")
    parser.add_argument("--requests", type=int, default=30, help="Measured requests PER flow; >=30, multiple of6")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "06_整理验收/CPU请求基准")
    args = parser.parse_args()
    try:
        schedule = paired_schedule(args.requests)
        profile = load_profile(args.profile, args.weights)
        actual = resolve_parameters(profile)
        if actual["imgsz"] != 320:
            raise ValueError("This fixed protocol requires the selected profile imgsz=320")
        manifest_path = args.samples_dir / "manifest.json"
        records = json.loads(manifest_path.read_text())
        if isinstance(records, dict):
            records = [r for r in records["records"] if r["split"] == "val"][:3]
        if len(records) != 3 or any(r["split"] != "val" for r in records):
            raise ValueError("样本清单须包含三张验证图像")
        paths = [(args.samples_dir / (r["output_image"] if "output_image" in r else Path(r["image"]).name)).resolve() for r in records]
        if len(set(paths)) != 3 or any(file_sha256(p) != r["image_sha256"] for p, r in zip(paths, records)):
            raise ValueError("Sample names/hashes are invalid")

        args.output.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError, InferenceError) as exc:
        parser.exit(2, f"Benchmark not started: {exc}\n")

    wall_start = time.perf_counter()
    import torch
    import torchvision
    import ultralytics
    from ultralytics.utils import NUM_THREADS


    torch.set_num_threads(NUM_THREADS)
    torch.set_num_interop_threads(1)
    threads = torch.get_num_threads()
    import_done = time.perf_counter()
    report = dict(schema_version=1, started_at=datetime.now(timezone.utc).isoformat(),
                  description="单图后端请求流程：对照A每次重新加载模型，B复用模型；A为构造对照，不代表历史部署版本。",
                  host=dict(platform=platform.platform(), machine=platform.machine(), cpu_count=os.cpu_count(),
                            python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__,
                            ultralytics=ultralytics.__version__, threads=threads, interop_threads=torch.get_num_interop_threads()),
                  weights_sha256=file_sha256(args.weights), profile_sha256=profile["sha256"],
                  profile=profile["values"], parameters=actual, device="cpu", batch=1, precision="float32",
                  samples=[dict(filename=p.name, sha256=r["image_sha256"], split=r["split"]) for p, r in zip(paths, records)],
                  sample_manifest_sha256=file_sha256(manifest_path),
                  protocol=dict(clock="time.perf_counter; CPU operations and tensor.tolist complete synchronously",
                                included="A: Detector construction (weight file hash/load/class validation), plus predict; B: predict. Both: image open/decode/RGB, image hash, YOLO preprocess/forward/NMS, Python schema conversion.",
                                excluded="Imports, profile/sample prevalidation, 结果导出, equality checks, garbage collection outside request timer.",
                                caches="Imports loaded once before timings; filesystem cache not flushed. Warm-cache repeated local requests, not cold disk startup.",
                                warmup="One request per preselected image per flow (3 each), alternating order. A warmup models discarded. B same model retained; its lazy predictor setup occurs during warmup.",
                                ordering="Each image has equal AB and BA measured pairs. No concurrent A/B execution.",
                                scope="Only this host/current conditions: CPU process-reuse engineering measurement; no general or GPU acceleration claim, model-selection input, or accuracy improvement claim.",
                                equality=dict(coordinate_atol=.001, confidence_atol=.00001, rtol=.00001)),
                  import_and_thread_setup_ms=(import_done-wall_start)*1000,
                  warmup=[], pairs=[], issues=[])

    start = time.perf_counter()
    persistent = Detector(args.weights, "cpu")
    report["B_model_constructor_ms"] = (time.perf_counter() - start) * 1000

    def request(flow, path):
        gc.collect()
        tick = time.perf_counter()
        detector = Detector(args.weights, "cpu") if flow == "A" else persistent
        result = detector.predict([path], **actual)
        elapsed = (time.perf_counter() - tick) * 1000
        if torch.get_num_threads() != threads or torch.get_num_interop_threads() != 1:
            raise RuntimeError("CPU thread count changed during a request")
        if next(detector.model.model.parameters()).dtype != torch.float32 or detector.model.predictor.model.fp16:
            raise RuntimeError("Unexpected model precision; refusing mixed-precision comparison")
        if result["metadata"]["model_sha256"] != report["weights_sha256"] or result["metadata"]["device"] != "cpu":
            raise RuntimeError("Model/device changed during benchmark")
        return dict(flow=flow, filename=path.name, latency_ms=elapsed, prediction=result)

    try:
        for index, path in enumerate(paths):
            order = ("A", "B") if index % 2 == 0 else ("B", "A")
            runs = {flow: request(flow, path) for flow in order}
            report["warmup"].append(dict(image_index=index, order=list(order),
                                         equivalent=equivalent(runs["A"]["prediction"], runs["B"]["prediction"]), runs=runs))
        for pair_id, (index, order) in enumerate(schedule):
            runs = {flow: request(flow, paths[index]) for flow in order}
            report["pairs"].append(dict(pair_id=pair_id, image_index=index, order=list(order),
                                        equivalent=equivalent(runs["A"]["prediction"], runs["B"]["prediction"]), runs=runs))
        if file_sha256(args.weights) != report["weights_sha256"] or any(file_sha256(p) != r["image_sha256"] for p, r in zip(paths, records)):
            report["issues"].append("Input/model files changed during benchmark")
        all_equal = all(p["equivalent"] for p in report["warmup"] + report["pairs"])
        report["all_predictions_equivalent"] = all_equal
        report["summary"] = {flow: summarize([p["runs"][flow]["latency_ms"] for p in report["pairs"]]) for flow in ("A", "B")}
        b_setup = report["B_model_constructor_ms"] + sum(p["runs"]["B"]["latency_ms"] for p in report["warmup"])
        report["B_first_call_ms"] = report["warmup"][0]["runs"]["B"]["latency_ms"]
        report["B_constructor_plus_warmup_ms"] = b_setup
        report["B_amortized_total_ms"] = b_setup + report["summary"]["B"]["total_request_ms"]
        report["B_amortized_ms_per_request"] = report["B_amortized_total_ms"] / args.requests
        if all_equal and not report["issues"]:
            report["comparison"] = dict(
                median_request_speedup_A_over_B=report["summary"]["A"]["median_ms"] / report["summary"]["B"]["median_ms"],
                total_speedup_including_B_constructor_and_warmup=report["summary"]["A"]["total_request_ms"] / report["B_amortized_total_ms"])
            report["status"] = "completed_equivalent"
        else:
            report["status"] = "not_comparable"
            report["issues"].append("No speedup reported: equality or input integrity failed")
    except Exception as exc:
        report["status"] = "failed"
        report["issues"].append(f"{type(exc).__name__}: {exc}")
    report["whole_benchmark_wall_ms"] = (time.perf_counter() - wall_start) * 1000
    destination = args.output / "benchmark.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(report=str(destination.resolve()), status=report["status"],
                          issues=report["issues"], summary=report.get("summary"), comparison=report.get("comparison")), indent=2))
    return 0 if report["status"] == "completed_equivalent" else 1


if __name__ == "__main__":
    raise SystemExit(main())
