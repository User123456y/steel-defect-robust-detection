"""合成夹具与协议单元测试；测试数值不代表实测模型性能。"""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from steel_defect.data import CLASSES
from steel_defect.inference import file_sha256
from steel_defect.evaluation import read_yolo_labels
from steel_defect.robustness import (CONDITIONS, METRICS, digest, labels_text,
    prepare_suite, sample_seed, transform, validate_protocol, val_records)


class RobustnessTests(unittest.TestCase):
    def protocol(self):
        return dict(status="frozen", frozen_at="synthetic", split="val", conditions=CONDITIONS,
                    metrics=METRICS, rng_seed=424242)

    def test_clean_lossless_and_no_mutation(self):
        x = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        original = x.copy()
        actual = transform(x, CONDITIONS[0], 1)
        np.testing.assert_array_equal(actual, x)
        actual[:] = 0
        np.testing.assert_array_equal(x, original)

    def test_contrast_is_mean_centered_not_darkening(self):
        x = np.array([[[0, 20, 100], [200, 220, 100]]], dtype=np.uint8)
        actual = transform(x, CONDITIONS[2], 1)
        np.testing.assert_array_equal(actual, [[[50, 70, 100], [150, 170, 100]]])
        np.testing.assert_array_equal(x.mean((0, 1)), actual.mean((0, 1)))

    def test_noise_repeatability_order_independence_and_clipping(self):
        x = np.full((20, 20, 3), 128, dtype=np.uint8)
        a = sample_seed("a", "gaussian_noise_10", 424242)
        b = sample_seed("b", "gaussian_noise_10", 424242)
        first = transform(x, CONDITIONS[3], a)
        np.testing.assert_array_equal(first[:, :, 0], first[:, :, 1])
        np.testing.assert_array_equal(first[:, :, 1], first[:, :, 2])
        transform(x, CONDITIONS[3], b)
        np.testing.assert_array_equal(first, transform(x, CONDITIONS[3], a))
        self.assertFalse(np.array_equal(first, transform(x, CONDITIONS[3], b)))
        for value in (0, 255):
            result = transform(np.full_like(x, value), CONDITIONS[3], a)
            self.assertEqual(result.dtype, np.uint8)
            self.assertTrue(np.all(result >= 0) and np.all(result <= 255))

    def test_only_frozen_val_protocol(self):
        validate_protocol(self.protocol())
        for delta in (dict(split="test"), dict(status="draft"), dict(rng_seed=0),
                      dict(conditions=CONDITIONS[:1]), dict(metrics={**METRICS, "confidence": .5})):
            with self.assertRaises(ValueError):
                validate_protocol({**self.protocol(), **delta})
        with self.assertRaises(ValueError):
            transform(np.zeros((2, 2, 3), np.uint8), dict(kind="made_up"), 1)

    def test_materialization_only_reads_val_members_and_preserves_gt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rgb = np.full((10, 12, 3), 127, np.uint8)
            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, format="PNG")
            record = dict(split="val", image="IMAGES/fixture.png", width=12, height=10,
                          image_sha256=digest(buf.getvalue()), pixel_sha256=digest(b"12x10:RGB:" + rgb.tobytes()),
                          boxes=[[0, 1, 2, 11, 8]])

            manifest = dict(classes=list(CLASSES), split_counts=dict(val=1, test=1),
                            records=[record, dict(split="test", image="IMAGES/DO_NOT_READ.png")])
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            archive = root / "source.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("NEU-DET/IMAGES/fixture.png", buf.getvalue())
            protocol = {**self.protocol(), "manifest_sha256": file_sha256(manifest_path),
                        "archive_sha256": file_sha256(archive)}
            evidence = prepare_suite(archive, manifest_path, protocol, root / "suite")
            labels = set()
            for name, rows in evidence.items():
                self.assertEqual(len(rows), 1)
                path = Path(rows[0]["image"])
                label = path.parents[2] / "labels/val" / (path.stem + ".txt")
                decoded = read_yolo_labels(label, 12, 10, 6)
                self.assertEqual(decoded[0]["class_id"], rows[0]["ground_truth"][0]["class_id"])
                np.testing.assert_allclose(decoded[0]["xyxy"], rows[0]["ground_truth"][0]["xyxy"], rtol=0, atol=1e-8)
                labels.add(rows[0]["label_sha256"])
            self.assertEqual(len(labels), 1)
            clean = evidence["clean"][0]
            self.assertEqual(clean["source_pixel_sha256"], clean["transformed_pixel_sha256"])
            with self.assertRaises(FileExistsError):
                prepare_suite(archive, manifest_path, protocol, root / "suite")
            with self.assertRaises(ValueError):
                prepare_suite(archive, manifest_path, {**protocol, "manifest_sha256": "bad"}, root / "other")

    def test_val_record_gate(self):
        manifest = dict(classes=list(CLASSES), split_counts=dict(val=1),
                        records=[dict(split="val", image="a"), dict(split="test", image="b")])
        self.assertEqual(len(val_records(manifest)), 1)
        manifest["records"].append(dict(split="val", image="a"))
        with self.assertRaises(ValueError):
            val_records(manifest)


if __name__ == "__main__":
    unittest.main()
