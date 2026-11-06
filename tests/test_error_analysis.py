import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location("error_analysis",Path(__file__).resolve().parents[1]/"scripts/analyze_errors.py")
analysis=importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class AnalysisTests(unittest.TestCase):
    def test_quantile_and_class_strata(self):
        self.assertEqual(analysis.quantile([0,1,2,3]),.75)
        rows=[dict(class_id=0,band="low",output_image="a",missed=True),
              dict(class_id=1,band="low",output_image="b",missed=False)]
        result=analysis.strata(rows,["a","b"])
        self.assertEqual(result[0]["fn_fraction"],1)
        self.assertEqual(result[3]["fn_fraction"],0)
        self.assertIsNone(result[1]["fn_fraction"])

    def test_refuses_test_predictions_and_stale_manifest(self):
        for metadata in (dict(split="test",manifest_sha256="x"),dict(split="val",manifest_sha256="wrong")):
            with self.assertRaises(ValueError):
                analysis.check_predictions(dict(metadata=metadata,images=[]),[],"x")
        with self.assertRaises(ValueError):
            analysis.check_predictions(dict(metadata=dict(split="val",manifest_sha256="x"),images=[]),
                                       [dict(split="val",output_image="a")],"x")

    @unittest.skipUnless(importlib.util.find_spec("PIL"),"Pillow unavailable")
    def test_ring_metric_not_global_rms(self):
        from PIL import Image
        gray=Image.new("L",(20,20),100)
        self.assertEqual(analysis.local_contrast(gray,[5,5,15,15]),0)
        gray.paste(200,(5,5,15,15))
        self.assertAlmostEqual(analysis.local_contrast(gray,[5,5,15,15]),100/255)
        self.assertIsNone(analysis.local_contrast(gray,[0,0,20,20]))
        with self.assertRaises(ValueError):
            analysis.local_contrast(gray,[1,1,1,2])


if __name__=="__main__":
    unittest.main()
