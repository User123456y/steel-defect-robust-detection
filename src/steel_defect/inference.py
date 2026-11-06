"""本地图像检测与输入校验；缺失权重时拒绝隐式下载。"""
from __future__ import annotations

import hashlib
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .data import CLASSES

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 20_000_000
MAX_IMAGES = 100


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class InferenceError(ValueError):
    pass


def validate_parameters(confidence, iou, imgsz):
    for value in (confidence, iou):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
            raise InferenceError("置信度与 NMS IoU 必须为 (0, 1] 内的有限数值。")
    if isinstance(imgsz, bool) or not isinstance(imgsz, int) or imgsz < 32 or imgsz > 2048 or imgsz % 32:
        raise InferenceError("imgsz 必须为 32–2048 之间的 32 倍数。")


def resolve_device(device: str) -> str:
    if device == "cpu":
        return device
    try:
        import torch
    except ImportError as exc:
        raise InferenceError("缺少 PyTorch；请安装已验证的推理环境。") from exc
    if device == "mps" and torch.backends.mps.is_available():
        return device
    if device in {"cuda", "cuda:0", "0"} and torch.cuda.is_available():
        return "0"
    raise InferenceError(f"设备 {device!r} 不可用；可改用 --device cpu。")


def read_image(path: Path):
    from PIL import Image
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise InferenceError("仅支持 JPG、JPEG、PNG、BMP。")
    if not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise InferenceError("图片不存在或超过 20 MiB。")
    with Image.open(path) as image:
        if image.width * image.height > MAX_PIXELS or getattr(image, "n_frames", 1) != 1:
            raise InferenceError("图片超过 2000 万像素或含多帧。")

        image.load()
        return image.convert("RGB")


class Detector:
    """加载一次检测模型，复用于后续图像推理。"""

    def __init__(self, weights, device="cpu", *, model_factory=None):
        if not weights or not Path(weights).is_file():
            raise InferenceError("尚未加载已训练权重。请先完成训练，再用 --weights 指定本地 best.pt。")
        path = Path(weights).resolve()
        if path.suffix.lower() != ".pt":
            raise InferenceError("本入口仅接受本地训练生成的 .pt 权重。")
        self.device = resolve_device(device)
        self.model_sha256 = file_sha256(path)
        self.model_name = path.name
        if model_factory is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise InferenceError("缺少 Ultralytics；请安装项目推理依赖。") from exc
            model_factory = YOLO
        try:
            self.model = model_factory(str(path))
            names = self.model.names
            self.names = {int(k): v for k, v in names.items()} if isinstance(names, dict) else dict(enumerate(names))
            if self.names != dict(enumerate(CLASSES)) or self.model.task != "detect":
                raise InferenceError("权重不是本项目六类检测模型；拒绝把通用 COCO 模型当作缺陷模型。")
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(f"权重加载失败：{type(exc).__name__}: {exc}") from exc
        self._lock = threading.Lock()

    def predict(self, paths, *, confidence=.25, iou=.7, imgsz=320):
        paths = [Path(p) for p in paths]
        if not 1 <= len(paths) <= MAX_IMAGES:
            raise InferenceError("每次请选择 1–100 张图片。")
        validate_parameters(confidence, iou, imgsz)
        metadata = dict(model_name=self.model_name, model_sha256=self.model_sha256,
                        confidence_threshold=confidence, nms_iou=iou, imgsz=imgsz, device=self.device,
                        started_at=datetime.now(timezone.utc).isoformat(),
                        coordinates="zero-based xyxy pixel edges; raw stored image orientation",
                        preprocessing="none",
                        input_adaptation="Pillow decode, RGB conversion; Ultralytics letterbox/normalization")
        records = []
        started = time.perf_counter()
        for index, path in enumerate(paths):
            record = dict(image_id=f"{index:04d}", input_index=index, filename=path.name,
                          status="error", error=None, width=None, height=None, image_sha256=None, detections=[])
            tick = time.perf_counter()
            try:
                image = read_image(path)
                record.update(width=image.width, height=image.height, image_sha256=file_sha256(path))
                with self._lock:
                    result = self.model.predict(source=image, conf=confidence, iou=iou, imgsz=imgsz,
                                                device=self.device, verbose=False, save=False)[0]
                if tuple(result.orig_shape) != (image.height, image.width) or result.boxes is None:
                    raise InferenceError("模型返回尺寸或检测结构异常。")
                detections = []
                for row in result.boxes.data.cpu().tolist():
                    if len(row) != 6:
                        raise InferenceError("模型返回了非检测框结构。")
                    x1, y1, x2, y2, score, cls = map(float, row)
                    if not all(math.isfinite(v) for v in row) or cls != int(cls) or int(cls) not in self.names:
                        raise InferenceError("模型返回无效类别或非有限数值。")
                    if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height and 0 <= score <= 1):
                        raise InferenceError("模型返回越界坐标或无效置信度。")
                    detections.append(dict(class_id=int(cls), class_name=self.names[int(cls)],
                                           confidence=score, xyxy=[x1, y1, x2, y2]))
                record.update(status="success", detections=detections)
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            record["elapsed_ms"] = round((time.perf_counter() - tick) * 1000, 3)
            records.append(record)
        metadata["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return dict(schema_version=1, metadata=metadata, images=records)
