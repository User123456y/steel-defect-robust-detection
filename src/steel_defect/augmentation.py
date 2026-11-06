"""训练阶段对比度增强，使用Ultralytics 8.3.200接口。
参考上游ultralytics/data/build.py、data/dataset.py和models/yolo/detect/train.py。"""
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import __version__
from ultralytics.data.augment import Format
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import colorstr
from ultralytics.utils.torch_utils import unwrap_model


def validate_contrast_config(config):
    if not isinstance(config, dict) or set(config) != {"probability", "alpha_min", "alpha_max"}:
        raise ValueError("contrast_augmentation needs probability, alpha_min, alpha_max only")
    for value in config.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Contrast parameters must be finite numbers")
    if not 0 <= config["probability"] <= 1 or not 0 < config["alpha_min"] <= config["alpha_max"]:
        raise ValueError("Invalid contrast probability or alpha range")
    return dict(config)


class MeanCenteredContrast:
    """使用工作进程初始化的Python随机数，执行均值中心对比度增强；不重新设种或修改缓存图像。"""
    def __init__(self, probability, alpha_min, alpha_max):
        self.config = validate_contrast_config(dict(probability=probability, alpha_min=alpha_min, alpha_max=alpha_max))

    def __call__(self, labels):
        p = self.config["probability"]
        if p == 0 or random.random() >= p:
            return labels
        alpha = random.uniform(self.config["alpha_min"], self.config["alpha_max"])
        if alpha == 1:
            return labels
        original = labels["img"]
        if original.dtype != np.uint8 or original.ndim != 3 or original.shape[2] != 3:
            raise ValueError("Contrast expects HWC uint8 BGR image")

        mean = float(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).mean())
        result = labels.copy()
        result["img"] = np.rint(np.clip(mean + alpha * (original.astype(np.float64) - mean), 0, 255)).astype(np.uint8)
        return result


class ContrastDataset(YOLODataset):
    def __init__(self, *args, contrast_config, audit_path=None, **kwargs):
        self.contrast_config = validate_contrast_config(contrast_config)
        self.audit_path = Path(audit_path) if audit_path else None
        self.transform_build_count = 0
        super().__init__(*args, **kwargs)

    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        if self.augment:
            if not isinstance(transforms.transforms[-1], Format):
                raise RuntimeError("Unexpected upstream transform layout")
            transforms.insert(-1, MeanCenteredContrast(**self.contrast_config))
        self.transform_build_count += 1
        count = sum(isinstance(t, MeanCenteredContrast) for t in transforms.transforms)
        assert count == int(bool(self.augment))
        if self.audit_path:
            event = dict(build=self.transform_build_count, augment=self.augment,
                         mosaic=hyp.mosaic, contrast_count=count,
                         transforms=[type(t).__name__ for t in transforms.transforms])
            with self.audit_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event) + "\n")
        return transforms


class ContrastTrainer(DetectionTrainer):
    def __init__(self, *args, contrast_config, **kwargs):
        if __version__ != "8.3.200":
            raise RuntimeError("Contrast integration verified only for ultralytics 8.3.200")
        self.contrast_config = validate_contrast_config(contrast_config)
        super().__init__(*args, **kwargs)

    def build_dataset(self, img_path, mode="train", batch=None):
        if mode != "train":
            return super().build_dataset(img_path, mode, batch)
        cfg = self.args
        stride = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)

        return ContrastDataset(img_path=img_path, imgsz=cfg.imgsz, batch_size=batch,
            augment=True, hyp=cfg, rect=cfg.rect, cache=cfg.cache or None,
            single_cls=cfg.single_cls or False, stride=stride, pad=0.0,
            prefix=colorstr("train: "), task=cfg.task, classes=cfg.classes,
            data=self.data, fraction=cfg.fraction, contrast_config=self.contrast_config,
            audit_path=self.save_dir / "contrast_transform_audit.jsonl")
