"""合成夹具与协议单元测试；测试数值不代表实测模型性能。"""
import csv
import importlib.util
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from steel_defect.data import CLASSES
from steel_defect.inference import Detector, InferenceError
from steel_defect.export import export_results


class Rows:
    def __init__(self, rows):
        self.rows = rows

    def cpu(self):
        return self

    def tolist(self):
        return self.rows


class MockModel:
    names = dict(enumerate(CLASSES))
    task = "detect"

    def __init__(self, weights):
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        image = kwargs["source"]
        rows = [] if image.getpixel((0, 0))[0] == 0 else [[1, 2, 15, 18, .8, 5]]
        return [SimpleNamespace(orig_shape=(image.height, image.width), boxes=SimpleNamespace(data=Rows(rows)))]


class LoadingTests(unittest.TestCase):
    def test_missing_weights_never_calls_loader(self):
        def forbidden(path):
            self.fail("missing weights must never trigger download/loader")
        with self.assertRaisesRegex(InferenceError, "已训练权重"):
            Detector(None, model_factory=forbidden)

    def test_coco_mapping_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp) / "mock.pt"
            weights.write_bytes(b"fixture, not a trained checkpoint")
            with self.assertRaisesRegex(InferenceError, "COCO"):
                Detector(weights, model_factory=lambda _: SimpleNamespace(names={0: "person"}, task="detect"))


@unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow missing; real image decoding/export tests not run")
class InferenceExportTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.paths = []
        for folder, color in [("left", 255), ("right", 0)]:
            directory = self.root / folder
            directory.mkdir()
            path = directory / "same.png"
            Image.new("L", (30, 20), color).save(path)
            self.paths.append(path)
        broken = self.root / "broken.png"
        broken.write_bytes(b"not an image")
        self.paths.append(broken)
        self.weights = self.root / "mock.pt"
        self.weights.write_bytes(b"test-only mock weights")
        self.loads = 0
        def factory(path):
            self.loads += 1
            return MockModel(path)
        self.detector = Detector(self.weights, model_factory=factory)

    def test_partial_failure_and_zero_boxes_are_distinct_and_model_reused(self):
        result = self.detector.predict(self.paths)
        self.detector.predict(self.paths[:1])
        self.assertEqual(self.loads, 1)
        self.assertEqual([r["status"] for r in result["images"]], ["success", "success", "error"])
        self.assertEqual(result["images"][1]["detections"], [])
        self.assertTrue(result["images"][2]["error"])
        first = result["images"][0]
        self.assertEqual((first["width"], first["height"]), (30, 20))
        self.assertEqual(first["detections"][0], dict(class_id=5, class_name="scratches", confidence=.8, xyxy=[1., 2., 15., 18.]))
        self.assertEqual(self.detector.model.calls[0]["source"].mode, "RGB")

    def test_export_preserves_mapping_errors_metadata_and_original_coordinates(self):
        result = self.detector.predict(self.paths, confidence=.4, imgsz=640)
        bundle = export_results(result, self.paths, self.root / "outputs")
        repeated = export_results(result, self.paths, self.root / "outputs")
        self.assertNotEqual(bundle["folder"], repeated["folder"])
        payload = json.loads(Path(bundle["files"][0]).read_text())
        self.assertEqual(payload["metadata"]["imgsz"], 640)
        self.assertEqual(len(payload["metadata"]["model_sha256"]), 64)
        self.assertEqual(payload["images"][0]["annotated_image"], "0000_annotated.png")
        self.assertEqual(payload["images"][1]["annotated_image"], "0001_annotated.png")
        self.assertIsNone(payload["images"][2]["annotated_image"])
        with Path(bundle["files"][1]).open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3)
        self.assertEqual([r["status"] for r in rows], ["success", "success", "error"])
        self.assertEqual(rows[0]["x2"], "15.0")
        self.assertEqual(rows[0]["confidence_threshold"], "0.4")
        self.assertEqual(rows[1]["detection_count"], "0")
        self.assertEqual(rows[1]["class_id"], "")
        with zipfile.ZipFile(bundle["files"][2]) as archive:
            self.assertEqual(set(archive.namelist()), {"results.json", "detections.csv", "0000_annotated.png", "0001_annotated.png", "0000_original.png", "0001_original.png"})

    def test_exif_orientation_does_not_rotate_only_original_preview(self):
        from PIL import Image
        path = self.root / "phone.jpg"
        image = Image.new("RGB", (30, 20), "black")
        image.paste("red", (20, 0, 30, 20))
        exif = image.getexif()
        exif[274] = 6
        image.save(path, exif=exif)
        result = self.detector.predict([path])
        bundle = export_results(result, [path], self.root / "outputs")
        with Image.open(bundle["originals"][0][0]) as original, Image.open(bundle["gallery"][0][0]) as annotated:
            self.assertEqual(original.size, (30, 20))
            self.assertEqual(annotated.size, original.size)
            self.assertNotIn(274, original.getexif())
            self.assertNotIn(274, annotated.getexif())
            self.assertEqual(original.tobytes(), annotated.tobytes())
            self.assertGreater(original.getpixel((28, 10))[0], 200)

    def test_changed_source_does_not_export_wrong_annotation(self):
        from PIL import Image
        result = self.detector.predict(self.paths[:1])
        Image.new("RGB", (30, 20), "red").save(self.paths[0])
        bundle = export_results(result, self.paths[:1], self.root / "outputs")
        self.assertIn("changed", bundle["result"]["images"][0]["export_error"])
        self.assertEqual(bundle["gallery"], [])

    def test_invalid_parameters_and_mismatched_order_fail(self):
        with self.assertRaises(InferenceError):
            self.detector.predict(self.paths, confidence=float("nan"))
        with self.assertRaises(InferenceError):
            self.detector.predict(self.paths, imgsz=321)
        result = self.detector.predict(self.paths)
        with self.assertRaises(ValueError):
            export_results(result, list(reversed(self.paths)), self.root / "outputs")


if __name__ == "__main__":
    unittest.main()
