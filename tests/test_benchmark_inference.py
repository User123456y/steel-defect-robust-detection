"""合成夹具与协议单元测试；测试数值不代表实测模型性能。"""
import copy
import importlib.util
import unittest
from collections import Counter
from pathlib import Path

spec = importlib.util.spec_from_file_location("benchmark", Path(__file__).resolve().parents[1] / "scripts/benchmark_inference.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class BenchmarkProtocolTests(unittest.TestCase):
    def test_pairs_are_balanced_for_every_image(self):
        schedule = benchmark.paired_schedule(30)
        self.assertEqual(Counter((i, order) for i, order in schedule),
                         Counter({(i, order): 5 for i in range(3) for order in (("A", "B"), ("B", "A"))}))
        for count in (0, 29, 31):
            with self.assertRaises(ValueError):
                benchmark.paired_schedule(count)

    def result(self):
        return dict(metadata=dict(model_sha256="a"*64,confidence_threshold=.25,nms_iou=.7,imgsz=320,device="cpu",preprocessing="none"),
                    images=[dict(filename="fixture.jpg",image_sha256="b"*64,width=200,height=200,status="success",
                                 detections=[dict(class_id=1,class_name="inclusion",confidence=.8,xyxy=[1.,2.,30.,40.])])])

    def test_numeric_tolerance_passes_but_wrong_outputs_or_failed_requests_do_not(self):
        left = self.result()
        right = copy.deepcopy(left)
        right["images"][0]["detections"][0]["xyxy"][0] += .0001
        self.assertTrue(benchmark.equivalent(left, right))
        for field, value in [("class_id",2),("class_name","patches"),("confidence",.7),("confidence",float("nan")),("xyxy",[1.,2.,31.,40.])]:
            right = copy.deepcopy(left)
            right["images"][0]["detections"][0][field] = value
            self.assertFalse(benchmark.equivalent(left, right), field)
        right = copy.deepcopy(left)
        right["images"][0]["status"] = "error"
        self.assertFalse(benchmark.equivalent(left, right))
        right = copy.deepcopy(left)
        right["images"][0]["detections"] = []
        self.assertFalse(benchmark.equivalent(left, right))
        right = copy.deepcopy(left)
        right["metadata"]["model_sha256"] = "c"*64
        self.assertFalse(benchmark.equivalent(left, right))


if __name__ == "__main__":
    unittest.main()
