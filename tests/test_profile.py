"""合成夹具与协议单元测试；测试数值不代表实测模型性能。"""
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from steel_defect.inference import InferenceError
from steel_defect.profile import load_profile, resolve_parameters, attach_profile_metadata

spec = importlib.util.spec_from_file_location("predict_cli", ROOT / "scripts/predict.py")
predict_cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict_cli)


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.weights = self.root / "fixture.pt"
        self.weights.write_bytes(b"mock profile binding; not a trained model")
        self.digest = hashlib.sha256(self.weights.read_bytes()).hexdigest()
        self.path = self.root / "profile.json"
        self.values = dict(weights_sha256=self.digest, confidence=.37, nms_iou=.6,
                           imgsz=640, preprocessing="none", model_label="fixture only",
                           source={"split": "val", "report": "fixture"})

    def write_load(self, **updates):
        self.path.write_text(json.dumps(self.values | updates))
        return load_profile(self.path, self.weights)

    def test_invalid_hash_and_changed_weights_fail(self):
        for digest in ("bad", "0" * 64):
            with self.subTest(digest=digest), self.assertRaises(InferenceError):
                self.write_load(weights_sha256=digest)
        self.write_load()
        self.weights.write_bytes(b"changed checkpoint")
        with self.assertRaisesRegex(InferenceError, "SHA256"):
            load_profile(self.path, self.weights)

    def test_protocol_parameters_and_required_fields(self):
        for key, value in [("confidence", True), ("confidence", float("nan")), ("confidence", 0),
                           ("nms_iou", 1.1), ("imgsz", 321), ("imgsz", 640.),
                           ("preprocessing", "clahe"), ("model_label", {})]:
            with self.subTest(key=key, value=value), self.assertRaises(InferenceError):
                self.write_load(**{key: value})
        self.path.write_text("{}")
        with self.assertRaises(InferenceError):
            load_profile(self.path, self.weights)

    def test_cli_defaults_do_not_override_profile_but_explicit_flags_do(self):
        profile = self.write_load()
        args = predict_cli.build_parser().parse_args(["image.jpg", "--weights", str(self.weights), "--profile", str(self.path)])
        actual = resolve_parameters(profile, confidence=args.confidence, iou=args.iou, imgsz=args.imgsz)
        self.assertEqual(actual, dict(confidence=.37, iou=.6, imgsz=640))
        args = predict_cli.build_parser().parse_args(["image.jpg", "--weights", str(self.weights), "--confidence", ".8", "--imgsz", "320"])
        actual = resolve_parameters(profile, confidence=args.confidence, iou=args.iou, imgsz=args.imgsz)
        self.assertEqual(actual, dict(confidence=.8, iou=.6, imgsz=320))
        result = {"metadata": {"model_sha256": self.digest, "confidence_threshold": .8}}
        attach_profile_metadata(result, profile, actual)
        meta = result["metadata"]["inference_profile"]
        self.assertEqual(meta["recommended"]["confidence"], .37)
        self.assertEqual(meta["overridden_parameters"], ["confidence", "imgsz"])
        self.assertEqual(meta["source"], self.values["source"])
        self.assertEqual(result["metadata"]["confidence_threshold"], .8)

    def test_without_profile_preserves_defaults_and_loaded_model_must_match(self):
        self.assertEqual(resolve_parameters(), dict(confidence=.25, iou=.7, imgsz=320))
        profile = self.write_load()
        with self.assertRaises(InferenceError):
            attach_profile_metadata({"metadata": {"model_sha256": "f" * 64}}, profile, resolve_parameters(profile))


if __name__ == "__main__":
    unittest.main()
