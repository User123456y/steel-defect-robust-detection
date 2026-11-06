#!/usr/bin/env python3
"""执行YOLO检测训练并记录配置、环境和工件校验值。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

from check_environment import inspect_environment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data/processed/data.yaml")
    parser.add_argument("--initial-weights", type=Path, required=True, help="本地初始权重；训练前需自行准备与实验一致的yolo11n.pt")
    parser.add_argument("--device", default="0", help="Single CUDA index, cpu, or mps; never silently falls back")
    parser.add_argument("--project", type=Path, default=Path("runs"))
    parser.add_argument("--verify-image", type=Path, help="Optional train/val image for saved-checkpoint reload/predict verification")
    return parser.parse_args()


def main():
    args = parse_args()
    if not (args.device.isdigit() or args.device in {"cpu", "mps"}):
        raise ValueError("--device must be one CUDA index, cpu, or mps")
    for path in (args.config, args.data, args.verify_image, args.initial_weights):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("请在Python 3.12环境安装requirements.txt依赖。") from exc
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not {"stage", "model", "train"} <= set(config) or set(config) - {"stage", "model", "train", "contrast_augmentation"}:
        raise ValueError("Config requires stage/model/train, optionally contrast_augmentation")
    contrast = config.get("contrast_augmentation")
    if "contrast_augmentation" in config:
        from steel_defect.augmentation import validate_contrast_config
        contrast = validate_contrast_config(contrast)
        if config["stage"] != "control":
            raise ValueError("Contrast is restricted to the control experiment")
    train = config["train"]
    if not isinstance(train, dict) or train.get("epochs", 0) <= 0:
        raise ValueError("train must contain positive epochs")
    if config["stage"] not in {"smoke", "baseline", "control"}:
        raise ValueError("stage must be smoke, baseline, or control")
    if config["stage"] == "smoke" and not 3 <= train["epochs"] <= 5:
        raise ValueError("Smoke must use 3–5 epochs")
    reserved = {"data", "device", "project", "name", "exist_ok", "resume", "model", "task", "mode", "split"}
    if reserved.intersection(train):
        raise ValueError(f"Reserved train options: {sorted(reserved.intersection(train))}")
    dataset = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict) or not all(k in dataset for k in ("train", "val", "names")):
        raise ValueError("Expected a prepared detection data.yaml with train/val/names")
    if "download" in dataset:
        raise ValueError("Use prepared local data; remove any dataset download directive")
    environment = inspect_environment(args.device)
    if args.device.isdigit() and not environment["gpu_probe_passed"]:
        raise RuntimeError("CUDA probe failed. Run scripts/check_environment.py --require-cuda for details.")
    run_name = f"{config['stage']}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    run_dir = args.project.resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {"status": "started", "started_utc": utc_now(), "config": config,
                "config_sha256": sha256(args.config), "data_yaml_sha256": sha256(args.data),
                "data_yaml": str(args.data.resolve()), "environment": environment,
                "checkpoint_prediction": "pending" if args.verify_image else "not_requested",
                "timing_scope": "model loading + train (including val/save) + optional reload/predict; synchronized CUDA"}
    manifest_path = args.data.resolve().parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    metadata["data_manifest_sha256"] = sha256(manifest_path)
    metadata["data_manifest"] = str(manifest_path)
    metadata["train_entry_sha256"] = sha256(Path(__file__))
    (run_dir / "input_config.yaml").write_text(args.config.read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "input_data.yaml").write_text(args.data.read_text(encoding="utf-8"), encoding="utf-8")

    def save_metadata():
        (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    if contrast is not None:
        module = Path(__file__).resolve().parents[1] / "src/steel_defect/augmentation.py"
        proposal = Path(__file__).resolve().parents[1] / "configs/contrast_proposal.json"
        metadata["contrast_augmentation"] = dict(parameters=contrast, module_sha256=sha256(module),
            proposal_sha256=sha256(proposal), rng="worker-seeded Python random; no reseeding",
            placement="train only, after existing augmentations, before Format; rebuilt with close_mosaic")
    save_metadata()
    started = time.perf_counter()
    torch = None
    try:
        import torch
        from ultralytics import YOLO
        cuda_index = int(args.device) if args.device.isdigit() else None
        if cuda_index is not None:
            torch.cuda.synchronize(cuda_index)
            torch.cuda.reset_peak_memory_stats(cuda_index)
        started = time.perf_counter()
        model = YOLO(str(args.initial_weights.resolve()), task="detect")
        initial_path = Path(getattr(model, "ckpt_path", config["model"]))
        if initial_path.is_file():
            metadata["initial_weights_sha256"] = sha256(initial_path)
        trainer_kwargs = {}
        if contrast is not None:
            from functools import partial
            from steel_defect.augmentation import ContrastTrainer
            trainer_kwargs["trainer"] = partial(ContrastTrainer, contrast_config=contrast)
        model.train(**trainer_kwargs, **train, data=str(args.data.resolve()), device=args.device,
                    project=str(args.project.resolve()), name=run_name, exist_ok=True)
        best = Path(model.trainer.best)
        if not best.is_file():
            raise RuntimeError("Training returned without a best checkpoint")
        metadata["best_checkpoint"] = str(best.resolve())
        metadata["best_checkpoint_sha256"] = sha256(best)
        if args.verify_image:
            reloaded = YOLO(str(best), task="detect")
            results = reloaded.predict(source=str(args.verify_image.resolve()), device=args.device,
                                       imgsz=train.get("imgsz", 320), save=False, verbose=False)
            metadata["checkpoint_prediction"] = {"status": "passed", "image_sha256": sha256(args.verify_image),
                                                 "results_count": len(results)}
        if cuda_index is not None:
            torch.cuda.synchronize(cuda_index)
            metadata["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(cuda_index)
            metadata["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(cuda_index)
        metadata["status"] = "completed"
        print(f"Completed training: {run_dir}")
        if not args.verify_image:
            print("Checkpoint reload / single-image prediction remains unverified; provide --verify-image to include it.")
    except BaseException as exc:
        metadata["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        metadata["error_type"] = type(exc).__name__

        raise
    finally:
        metadata["elapsed_seconds"] = time.perf_counter() - started
        metadata["finished_utc"] = utc_now()
        save_metadata()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
