"""校验与权重SHA256绑定的推理配置并解析参数优先级。"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .inference import InferenceError, file_sha256, validate_parameters

DEFAULTS = dict(confidence=.25, iou=.7, imgsz=320)


def load_profile(path, weights):
    if path is None:
        return None
    try:
        raw = Path(path).read_bytes()
        profile = json.loads(raw)
        if not isinstance(profile, dict):
            raise ValueError("profile must be a JSON object")
        required = {"weights_sha256", "imgsz", "confidence", "nms_iou", "preprocessing"}
        if required - profile.keys():
            raise ValueError(f"missing fields: {sorted(required - profile.keys())}")
        digest = profile["weights_sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError("weights_sha256 must contain 64 hexadecimal characters")
        if not weights or file_sha256(weights) != digest.lower():
            raise ValueError("profile 与当前实际权重的 SHA256 不匹配")
        if profile["preprocessing"] != "none":
            raise ValueError("仅支持 preprocessing=none；未知预处理协议拒绝运行")
        validate_parameters(profile["confidence"], profile["nms_iou"], profile["imgsz"])
        if "model_label" in profile and not isinstance(profile["model_label"], str):
            raise ValueError("model_label must be text")
        return dict(values=profile, filename=Path(path).name, sha256=hashlib.sha256(raw).hexdigest())
    except (OSError, ValueError, TypeError) as exc:
        raise InferenceError(f"推理 profile 无效：{exc}") from exc


def resolve_parameters(profile=None, *, confidence=None, iou=None, imgsz=None):
    values = dict(DEFAULTS)
    if profile is not None:
        p = profile["values"]
        values.update(confidence=p["confidence"], iou=p["nms_iou"], imgsz=p["imgsz"])
    values.update({key: value for key, value in dict(confidence=confidence, iou=iou, imgsz=imgsz).items() if value is not None})
    validate_parameters(**values)
    return values


def attach_profile_metadata(result, profile, actual):
    """分开记录实际推理参数、推荐值和显式覆盖参数。"""
    if profile is None:
        return
    if result["metadata"]["model_sha256"] != profile["values"]["weights_sha256"].lower():
        raise InferenceError("profile 与已加载模型 SHA256 不匹配；请重新加载正确权重。")
    recommended = resolve_parameters(profile)
    result["metadata"]["inference_profile"] = dict(
        filename=profile["filename"], sha256=profile["sha256"],
        weights_sha256=profile["values"]["weights_sha256"].lower(),
        recommended=recommended, source=profile["values"].get("source"),
        model_label=profile["values"].get("model_label"),
        overridden_parameters=[key for key in recommended if actual[key] != recommended[key]])
