"""合成夹具与协议单元测试；测试数值不代表实测模型性能。"""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from steel_defect.evaluation import (box_iou, check_test_gate, count_metrics, match_image,
                                     official_metrics, read_yolo_labels, summarize)


def box(cls=0, score=None, coordinates=None):
    result = dict(class_id=cls, xyxy=coordinates or [0, 0, 10, 10])
    if score is not None:
        result["confidence"] = score
    return result


class EvaluationTests(unittest.TestCase):
    def test_duplicate_predictions_are_fp(self):
        result = match_image([box()], [box(score=.8), box(score=.9)])
        self.assertEqual(result["matches"][0]["prediction_index"], 1)
        self.assertEqual(result["false_positive_indices"], [0])
        self.assertEqual(result["false_negative_indices"], [])

    def test_multi_class_and_wrong_class_evidence(self):
        gt = [box(0), box(1, coordinates=[20, 20, 30, 30])]
        pred = [box(0, .9), box(2, .8, [20, 20, 30, 30])]
        analysis = match_image(gt, pred)
        self.assertEqual(len(analysis["matches"]), 1)
        self.assertEqual(analysis["false_positive_indices"], [1])
        self.assertEqual(analysis["false_negative_indices"], [1])
        self.assertEqual(analysis["class_confusion_candidates"][0]["ground_truth_index"], 1)
        metric = summarize([dict(ground_truth=gt, detections=pred, analysis=analysis)], ["a", "b", "c"])
        self.assertEqual(metric["overall"]["f1"], .5)
        self.assertEqual(metric["per_class"][2]["fp"], 1)
        self.assertIsNone(metric["per_class"][2]["recall"])

    def test_one_prediction_cannot_claim_two_gt(self):
        result = match_image([box(), box()], [box(score=.9)])
        self.assertEqual(len(result["matches"]), 1)
        self.assertEqual(len(result["false_negative_indices"]), 1)

    def test_empty_sets_and_undefined(self):
        self.assertEqual(match_image([], [box(score=.8)])["false_positive_indices"], [0])
        self.assertEqual(match_image([box()], [])["false_negative_indices"], [0])
        self.assertEqual(match_image([], [])["matches"], [])
        self.assertIsNone(count_metrics(0, 0, 0)["f1"])
        self.assertIsNone(count_metrics(0, 0, 1)["precision"])
        self.assertEqual(count_metrics(0, 0, 1)["f1"], 0)

    def test_iou_and_localization(self):
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)
        result = match_image([box()], [box(score=.9, coordinates=[5, 0, 15, 10])])
        self.assertEqual(len(result["localization_candidates"]), 1)
        self.assertEqual(len(result["matches"]), 0)

    def test_yolo_decode_and_invalid_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            label = Path(tmp) / "label.txt"
            label.write_text("0 0.5 0.5 1 1\n1 0.25 0.25 0.5 0.5\n")
            boxes = read_yolo_labels(label, 200, 100, 2)
            self.assertEqual(boxes[0]["xyxy"], [0, 0, 200, 100])
            self.assertEqual(boxes[1]["xyxy"], [0, 0, 100, 50])
            for invalid in ("0 nan 0.5 1 1", "2 0.5 0.5 1 1", "0 .1 .1 1 1", "0 .5 .5 0 1"):
                label.write_text(invalid)
                with self.assertRaises(ValueError):
                    read_yolo_labels(label, 200, 100, 2)
            label.write_text("")
            self.assertEqual(read_yolo_labels(label, 200, 100, 2), [])

    def test_official_class_index_is_not_class_row(self):
        metric = SimpleNamespace(box=SimpleNamespace(ap_class_index=[1], map50=.7, map=.4,
                                      class_result=lambda row: (.8, .6, .7, .4)))
        result = official_metrics(metric, ["empty", "present"], [0, 2])
        self.assertIsNone(result["per_class"][0]["ap50"])
        self.assertEqual(result["per_class"][1]["ap50"], .7)
        self.assertEqual(result["per_class"][1]["precision_best_f1"], .8)

    def test_official_no_predictions(self):
        metric = SimpleNamespace(box=SimpleNamespace(ap_class_index=[], map50=0, map=0))
        result = official_metrics(metric, ["present", "empty"], [1, 0])
        self.assertEqual(result["per_class"][0]["ap50"], 0)
        self.assertIsNone(result["per_class"][1]["ap50"])

    def test_test_gate_is_explicit_and_hash_bound(self):
        expected = dict(weights_sha256="mock-weight", manifest_sha256="mock-manifest", config_sha256="mock-config", protocol={"confidence": .25})
        frozen = dict(status="frozen", selected_on_val="synthetic selection fixture", frozen_at="fixture", **expected)
        check_test_gate("val", False, None, expected)
        check_test_gate("test", True, frozen, expected)
        for enabled, record in [(False, frozen), (True, None), (True, dict(frozen, weights_sha256="wrong")), (True, dict(frozen, status="draft"))]:
            with self.assertRaises(ValueError):
                check_test_gate("test", enabled, record, expected)
        with self.assertRaises(ValueError):
            check_test_gate("val", True, frozen, expected)


if __name__ == "__main__":
    unittest.main()
