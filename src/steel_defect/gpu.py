"""固定输入尺寸CUDA推理，校验模型、配置与TensorRT运行环境。"""
from __future__ import annotations
import math
import statistics
from pathlib import Path
from .data import CLASSES
from .inference import file_sha256, read_image

WEIGHTS_SHA = '09400f91dc6a76bbc101ed3307beb4ca7085cec327cb11aeb50b5c1f8d2948fa'
BACKENDS = ('torch_fp32', 'trt_fp32', 'trt_fp16')


def latency_summary(values):
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError('Latency samples must be finite and positive')
    return dict(count=len(values), p50_ms=statistics.median(values),
                p95_ms=sorted(values)[math.ceil(.95 * len(values)) - 1],
                throughput_images_s=1000 * len(values) / sum(values),
                percentile_method='nearest rank; batch=1; sequential throughput')


def accuracy_gate(reference, candidate):
    """依据预设绝对指标下降上限进行精度验收。"""
    checks = {}
    def check(name, a, b, tolerance):
        checks[name] = bool(a is not None and b is not None and math.isfinite(a)
                            and math.isfinite(b) and b >= a - tolerance - 1e-12)
    for key in ('map50', 'map50_95'):
        check(key, reference['official'][key], candidate['official'][key], .005)
    for key in ('f1', 'recall'):
        check(key, reference['fixed']['overall'][key], candidate['fixed']['overall'][key], .01)
    for a, b in zip(reference['official']['per_class'], candidate['official']['per_class'], strict=True):
        if a['class_id'] != b['class_id']:
            raise ValueError('Class order mismatch')
        if a['gt_count']:
            for key in ('ap50', 'ap50_95'):
                check(f"class_{a['class_id']}_{key}", a[key], b[key], .02)
    return dict(passed=all(checks.values()), checks=checks)


def configure_cuda():
    import torch
    import ultralytics
    if ultralytics.__version__ != '8.3.200':
        raise RuntimeError('Requires frozen ultralytics==8.3.200')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU fallback forbidden')
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return torch


class GPUDetector:
    """每个进程独立加载固定batch=1、320方形输入的模型并使用统一NMS。
    TensorRT FP16采用混合精度策略，输入输出仍为FP32；部署时核验引擎元信息。"""
    def __init__(self, model, backend, profile):
        import json
        if backend not in BACKENDS:
            raise ValueError('Unknown backend')
        self.torch = configure_cuda()
        self.profile_sha256 = validate_profile(profile)
        from ultralytics.nn.autobackend import AutoBackend
        model = Path(model).resolve()
        if backend == 'torch_fp32':
            if model.suffix != '.pt' or file_sha256(model) != WEIGHTS_SHA:
                raise ValueError('Frozen checkpoint mismatch')
        else:
            metadata = json.loads(model.with_suffix('.json').read_text())
            import tensorrt
            if (metadata['tensorrt'] != tensorrt.__version__ or metadata['gpu'] != self.torch.cuda.get_device_name(0)
                    or metadata['compute_capability'] != list(self.torch.cuda.get_device_capability(0))
                    or metadata['profile_sha256'] != self.profile_sha256):
                raise ValueError('Engine runtime/GPU/profile incompatibility; rebuild on target')
            if (metadata['weights_sha256'] != WEIGHTS_SHA or metadata['backend'] != backend
                    or metadata['engine_sha256'] != file_sha256(model)):
                raise ValueError('Engine provenance mismatch')
        self.model = AutoBackend(str(model), device=self.torch.device('cuda:0'), fp16=False, fuse=True, verbose=False)
        if self.model.names != dict(enumerate(CLASSES)) or self.model.fp16:
            raise RuntimeError('Class mapping or FP32 I/O mismatch')
        if backend.startswith('trt') and (not self.model.engine or self.model.dynamic):
            raise RuntimeError('Static TensorRT engine required')
        if backend == 'torch_fp32' and not self.model.pt:
            raise RuntimeError('PyTorch backend required')
        self.backend = backend

    def prepare(self, path):
        import numpy as np
        from ultralytics.data.augment import LetterBox
        rgb = read_image(Path(path))
        bgr = np.asarray(rgb)[:, :, ::-1].copy()
        padded = LetterBox(new_shape=(320, 320), auto=False, scale_fill=False, scaleup=True, stride=32)(image=bgr)
        chw = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1))[None]
        tensor = self.torch.from_numpy(chw).to('cuda:0').float() / 255.0
        return tensor, (rgb.height, rgb.width)

    def forward(self, tensor):
        with self.torch.inference_mode():
            result = self.model(tensor)
        if isinstance(result, (tuple, list)):
            result = result[0]
        return result

    def postprocess(self, raw, shape, confidence=.4):
        from ultralytics.utils.ops import scale_boxes
        from ultralytics.utils.nms import non_max_suppression

        pred = non_max_suppression(raw.clone(), conf_thres=confidence, iou_thres=.7, nc=6,
                                   agnostic=False, max_det=300)[0]
        pred[:, :4] = scale_boxes((320, 320), pred[:, :4], shape)
        return [dict(class_id=int(row[5]), confidence=float(row[4]), xyxy=list(map(float, row[:4])))
                for row in pred.cpu().tolist()]

    def predict(self, path):
        tensor, shape = self.prepare(path)
        return dict(width=shape[1], height=shape[0], detections=self.postprocess(self.forward(tensor), shape))


def validate_profile(path):
    import json
    profile = json.loads(Path(path).read_text())
    expected = dict(weights_sha256=WEIGHTS_SHA, imgsz=320, confidence=.4, nms_iou=.7, preprocessing='none')
    if file_sha256(path) != '2363a45ebb1d9ddabb3b88dee9707a49868fbb36c531585ede482a07001740de' or any(profile.get(k) != v for k, v in expected.items()):
        raise ValueError('V3 requires the frozen final inference profile')
    return file_sha256(path)
