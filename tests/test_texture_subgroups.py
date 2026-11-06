import unittest
import numpy as np
from steel_defect.texture_subgroups import background_texture,fit_thresholds,assign_group,validate_predictions
class TextureTests(unittest.TestCase):
 def test_gradient_constant_and_linear(self):
  a=np.zeros((10,10)); self.assertEqual(background_texture(a,[])["score"],0)
  a=np.tile(np.arange(10),(10,1))
  self.assertAlmostEqual(background_texture(a,[])["score"],1/255)
  self.assertEqual(background_texture(a,[])["valid_pixels"],64)
 def test_union_and_neighbors_excluded(self):
  a=np.zeros((10,10)); a[3:7,3:7]=255
  r=background_texture(a,[[3,3,7,7],[3,3,7,7]])
  self.assertEqual(r["valid_pixels"],32)
  self.assertEqual(r["score"],0)
 def test_coverage_and_full_gt(self):
  self.assertIsNone(background_texture(np.zeros((10,10)),[[0,0,10,10]])["score"])
  self.assertIsNone(background_texture(np.zeros((3,3)),[])["score"])
 def test_train_only_quantile_and_minimum(self):
  rows=[dict(split="train",acquisition_class_id=0,score=float(i)) for i in range(30)]
  rows.append(dict(split="val",acquisition_class_id=0,score=9999))
  t=fit_thresholds(rows,[0,1]); self.assertEqual(t["0"]["threshold"],21.75); self.assertIsNone(t["1"]["threshold"])
  self.assertEqual(assign_group(dict(acquisition_class_id=0,score=21.75),t),"high")
  self.assertEqual(assign_group(dict(acquisition_class_id=0,score=None),t),"undefined")
 def test_refuse_test_or_wrong_hash(self):
  for split,h in (("test","x"),("val","wrong")):
   with self.assertRaises(ValueError): validate_predictions({"metadata":{"split":split,"manifest_sha256":h}},[],"x")
if __name__=="__main__": unittest.main()
