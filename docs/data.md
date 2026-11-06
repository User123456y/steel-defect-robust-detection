# 数据来源与准备

NEU-DET：[东北大学表面缺陷数据库](http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm)。请引用 Song 与 Yan, Applied Surface Science 285, 858–864 (2013)，以及发布页要求的相关文献。官方页面未核实到完整明确的商业与再分发许可，镜像与官方字节等价性亦未证明。

GC10-DET：[作者仓库](https://github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Datasets)，Lv 等, Deep Metallic Surface Defect Detection: The New Benchmark and Detection Network, Sensors 2020, 20(6), 1562, [论文](https://doi.org/10.3390/s20061562)。历史压力测试采用 [Dataset Ninja 镜像](https://datasetninja.com/gc10-det) 的 2300 图发布，排除 8 张空标注后评分 2292 图；与论文 3570 图不同。作者欢迎研究使用不等于明示再分发许可，论文开放许可也不自动覆盖全部数据。

## NEU 本地准备

自行取得有权使用的 NEU-DET 图片与 VOC XML，在 `data/raw/NEU-DET/` 下保留 `IMAGES/*.jpg`、`ANNOTATIONS/*.xml` 层级；文件名使用原始类别前缀。六类顺序为 crazing、inclusion、patches、pitted_surface、rolled-in_scale、scratches。准备脚本递归发现图片与 XML，检查尺寸、类别、坐标、缺失配对和源摘要。

```bash
python scripts/prepare_data.py --source data/raw/NEU-DET --output data/processed --curation configs/data_curation.json
```

`configs/data_curation.json` 只包含来源指纹、隔离文件名及保守分组，不含原始框坐标。该规则绑定历史镜像的路径与字节，若来源或目录不同会拒绝应用；请确认源文件身份，勿通过修改哈希跳过检查。准备后核对 `data/split_manifest.json` 中的文件名、图片 SHA256 和划分。历史分组将两个冲突的相同像素样本整体隔离，所有 dHash≤4 候选组保持同一划分，但不声称候选均为真重复或完全排除同源泄漏。

其他合法来源可不传 `--curation` 进行检查；未解决近重复候选会阻止 YOLO 导出，需要按报告复核并形成该来源自己的分组决策，不能将新划分的结果称为原实验复现。

公开验证身份文件仅保存每个验证样本身份及标注结构的 SHA256；运行 GPU 基准时在本地由自备标注生成摘要核验，不分发 GT 文本。原图、标签及本地产物均由 `.gitignore` 排除。
