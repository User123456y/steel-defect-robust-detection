# 第三方来源与许可

本项目源码及随附 YOLO11n 微调权重按 GNU Affero General Public License v3.0 分发，完整文本见 LICENSE。Ultralytics YOLO，Copyright © Ultralytics；上游源码：https://github.com/ultralytics/ultralytics/tree/v8.3.200 。模型源于官方 YOLO11n 预训练权重；其上游许可说明：https://www.ultralytics.com/license 。

`src/steel_defect/augmentation.py` 参考上游 `ultralytics/data/build.py`、`data/dataset.py`、`models/yolo/detect/train.py` 的接口与训练流程。未将该架构或预训练模型声明为本项目原创。

PyTorch、torchvision、NumPy、OpenCV、Pillow、PyYAML、ONNX、TensorRT 与 CUDA 属于各自项目/权利人，使用时遵守各自许可证。依赖通过包管理器安装，不在仓库中复制其运行库。

NEU-DET 与 GC10-DET 的数据、原始框标签和标注派生图不随仓库分发；项目 AGPL 许可不替代第三方数据授权。来源和文献见 docs/data.md。训练日志、聚合统计与分组清单仅记录实验过程，不授予原数据再分发权。
