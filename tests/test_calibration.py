"""合成夹具与协议单元测试；测试数值不代表实测模型性能。"""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import calibrate_threshold as calibration
from steel_defect.data import CLASSES
from steel_defect.evaluation import count_metrics
from steel_defect.inference import file_sha256


def detection(cls, confidence, xyxy=None):
    return dict(class_id=cls, confidence=confidence, xyxy=xyxy or [0, 0, 10, 10])


def fixture(root):
    for directory in ("images/val", "labels/val"):
        (root / directory).mkdir(parents=True)
    image, label = root / "images/val/mock.jpg", root / "labels/val/mock.txt"
    image.write_bytes(b"SYNTHETIC BYTES; mocked Detector, not an actual image")
    label.write_text("0 0.5 0.5 1 1\n")
    record = dict(split="val", output_image="images/val/mock.jpg", output_label="labels/val/mock.txt",
                  image_sha256=file_sha256(image), width=10, height=10, boxes=[[0, 0, 0, 10, 10]])

    manifest = dict(classes=list(CLASSES), records=[record, dict(split="test", output_image="images/test/DO_NOT_OPEN.jpg", output_label="labels/test/DO_NOT_OPEN.txt")])
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    (root / "validation_report.json").write_text(json.dumps(dict(g1_passed=True, export_complete=True)))
    return path, image, record


class CalibrationTests(unittest.TestCase):
    def test_fixed_grid_and_multiclass_counts(self):
        gt = [dict(class_id=0, xyxy=[0, 0, 10, 10]), dict(class_id=1, xyxy=[20, 20, 30, 30])]
        predictions = [detection(0, .6), detection(0, .3), detection(2, .4, [20, 20, 30, 30])]
        result = calibration.calibrate_predictions([dict(ground_truth=gt, detections=predictions)])
        self.assertEqual(result["grid"], [i / 100 for i in range(5, 76, 5)])
        fixed = result["baseline_confidence_025"]
        self.assertEqual((fixed["tp"], fixed["fp"], fixed["fn"]), (1, 2, 1))
        self.assertEqual(result["selected"]["threshold"], .55)
        self.assertEqual(result["selected_minus_025"]["fp"], -2)
        self.assertEqual(len(result["per_class_curve"]), 15 * 6)

    def test_threshold_strict_comparison(self):
        result = calibration.calibrate_predictions([dict(ground_truth=[], detections=[detection(0, .25)])])
        self.assertEqual(result["curve"][3]["fp"], 1)
        self.assertEqual(result["curve"][4]["fp"], 0)
        self.assertIsNone(result["curve"][4]["precision"])
        self.assertIsNone(result["curve"][4]["f1"])

    def test_exact_ties_precision_then_threshold(self):
        rows = [dict(threshold=.1, **count_metrics(1, 1, 0)),
                dict(threshold=.2, **count_metrics(1, 0, 1)),
                dict(threshold=.3, **count_metrics(2, 0, 2))]
        self.assertEqual(calibration.select_threshold(rows)["threshold"], .3)
        self.assertEqual(calibration.selection_key(rows[1])[0], calibration.selection_key(rows[2])[0])

    def test_null_is_not_fake_perfect_score(self):
        result = calibration.calibrate_predictions([dict(ground_truth=[], detections=[])])
        self.assertIsNone(result["selected"])
        self.assertIsNone(result["selected_minus_025"])
        self.assertTrue(all(row["f1"] is None for row in result["curve"]))
        rows = [dict(threshold=.1, **count_metrics(0, 1, 1)), dict(threshold=.7, **count_metrics(0, 0, 1))]
        self.assertEqual(calibration.select_threshold(rows)["threshold"], .1)

    def test_val_loader_ignores_test_artifacts_and_detects_mutations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, image, record = fixture(root)
            paths, evidence = calibration.load_val_records(manifest)
            self.assertEqual(paths, [image.resolve()])
            self.assertEqual(evidence[0]["ground_truth"][0]["xyxy"], [0, 0, 10, 10])
            (root / record["output_label"]).write_text("1 .5 .5 1 1\n")
            with self.assertRaisesRegex(ValueError, "differs from manifest"):
                calibration.load_val_records(manifest)

    def test_val_record_cannot_point_at_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _, _ = fixture(Path(tmp))
            value = json.loads(manifest.read_text())
            value["records"][0]["output_image"] = "images/test/DO_NOT_OPEN.jpg"
            manifest.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "标定仅允许读取验证集目录"):
                calibration.load_val_records(manifest)

    def test_cli_rejects_test_option(self):
        with patch.object(sys, "argv", ["calibrate", "--weights", "x", "--manifest", "x", "--config", "x", "--output", "x", "--split", "test"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                calibration.parse_args()

    def test_mock_pipeline_predicts_once_and_exports_auditable_curve(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, image, _ = fixture(root)
            weights, config = root / "mock.pt", root / "config.yaml"
            weights.write_bytes(b"MOCK CHECKPOINT")
            config.write_text("mock: true\n")
            args = SimpleNamespace(weights=weights, config=config, manifest=manifest, output=root / "result", device="cpu", imgsz=320)
            calls = []
            class MockDetector:
                device = "cpu"
                def __init__(self, *args):
                    pass
                def predict(self, paths, **kwargs):
                    calls.append((paths, kwargs))
                    return dict(images=[dict(status="success", width=10, height=10, image_sha256=file_sha256(image), detections=[detection(0, .3), detection(1, .2)])])
            with patch.object(calibration, "parse_args", return_value=args), patch.object(calibration, "Detector", MockDetector), patch.object(calibration.importlib.metadata, "version", return_value="8.3.200"), contextlib.redirect_stdout(io.StringIO()):
                calibration.main()
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1], dict(confidence=.001, iou=.7, imgsz=320))
            result = json.loads((args.output / "calibration.json").read_text())
            self.assertEqual(result["metadata"]["status"], "completed")
            self.assertEqual(result["selected"]["threshold"], .25)
            self.assertEqual(result["metadata"]["protocol_sha256"], calibration.canonical_hash(result["metadata"]["protocol"]))
            self.assertEqual(result["metadata"]["prediction_artifact_sha256"], file_sha256(args.output / "val_predictions.json"))
            self.assertEqual(len((args.output / "curve.csv").read_text(encoding="utf-8-sig").splitlines()), 16)


if __name__ == "__main__":
    unittest.main()
