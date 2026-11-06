# 推理与 GPU 部署

本地 CPU 验证环境：Python 3.12、PyTorch 2.8.0、torchvision 0.23.0、Ultralytics 8.3.200。OpenCV 和 NumPy 随依赖安装。原 GPU 实测环境：RTX 4090、PyTorch 2.13.0+cu132、torchvision 0.28.0+cu132、CUDA 13.2、TensorRT 10.16.1.11，另需 ONNX 与 nvidia-ml-py；不要用 CPU requirements 覆盖既有 GPU 环境。

先按目标 GPU/CUDA 环境安装对应 PyTorch，再安装 Ultralytics 8.3.200、PyYAML、Pillow、OpenCV、NumPy、ONNX、TensorRT 10.x 与 nvidia-ml-py。TensorRT 引擎依赖设备与运行库，应在目标机器重新生成。脚本拒绝 CUDA 静默回退，并固定 batch=1、320 方形输入、关闭 TF32。

```bash
python scripts/check_environment.py --require-cuda
python scripts/gpu_v3.py --help
python scripts/gpu_v3.py export --weights models/best.pt --profile models/inference_profile.json --output runs/gpu_export
python scripts/gpu_v3.py run --model runs/gpu_export/trt_fp16.engine --backend trt_fp16 --profile models/inference_profile.json --data data/processed --output runs/trt_fp16
```

导出入口生成 ONNX（opset 17、固定输入、无内置 NMS）与 FP32/FP16 TensorRT 引擎，同时记录模型、profile 与工件哈希。`run` 测量后端，`compare` 比较相同源码与协议的结果，`predict` 执行图像检测。具体参数以各子命令 `--help` 为准。公开仓库不携带机器相关引擎，历史速度不能作为其他 GPU 或公开版代码重新实测的声明。
