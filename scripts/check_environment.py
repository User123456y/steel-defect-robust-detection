#!/usr/bin/env python3
"""检查Python、依赖版本和实际CUDA运算能力。"""
import argparse
import importlib.metadata
import json
import platform
import sys


def inspect_environment(device="0"):
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "packages": {}, "device": device, "cuda_available": False,
              "gpu_probe_passed": False, "issues": []}
    for package in ("torch", "torchvision", "ultralytics", "PyYAML", "Pillow"):
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
            report["issues"].append(f"Missing package: {package}")
    if sys.version_info[:2] != (3, 12):
        report["issues"].append("已验证环境使用Python 3.12；当前Python版本未经本项目验证。")
    try:
        import torch
        report["cuda_build"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        report["mps_available"] = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        if device.isdigit() and report["cuda_available"]:
            index = int(device)
            report["gpu_name"] = torch.cuda.get_device_name(index)
            capability = torch.cuda.get_device_capability(index)
            report["gpu_capability"] = list(capability)
            report["compiled_arches"] = torch.cuda.get_arch_list()
            if capability[0] >= 10 and tuple(map(int, (torch.version.cuda or "0.0").split(".")[:2])) < (12, 8):
                report["issues"].append("Blackwell显卡需要包含对应计算架构的CUDA构建。")

            with torch.cuda.device(index):
                x = torch.ones((32, 32), device=f"cuda:{index}")
                value = (x @ x).sum().item()
                torch.cuda.synchronize(index)
                report["gpu_probe_passed"] = value == 32768
        elif device.isdigit():
            report["issues"].append("CUDA is unavailable; do not start the planned GPU training.")
    except Exception as exc:
        report["issues"].append(f"Torch/GPU probe failed: {type(exc).__name__}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0", help="Single CUDA index to probe (default: 0)")
    parser.add_argument("--require-cuda", action="store_true", help="Exit nonzero unless the GPU calculation succeeds")
    args = parser.parse_args()
    report = inspect_environment(args.device)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if args.require_cuda and not report["gpu_probe_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
