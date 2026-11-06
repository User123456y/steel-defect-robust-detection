"""合成夹具与协议单元测试；测试数值不代表实测模型性能。"""
import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from steel_defect.gpu import latency_summary, accuracy_gate, validate_profile

class GPUTests(unittest.TestCase):
    def test_nearest_rank_and_serial_throughput(self):
        stats=latency_summary(list(range(1,101)))
        self.assertEqual(stats['p95_ms'],95)
        self.assertEqual(stats['p50_ms'],50.5)
        self.assertAlmostEqual(stats['throughput_images_s'],1000/50.5)
        for invalid in ([],[0],[float('nan')],[-1]):
            with self.assertRaises(ValueError): latency_summary(invalid)

    def test_gate_boundary_and_per_class_failure(self):
        baseline=dict(official=dict(map50=.8,map50_95=.5,per_class=[dict(class_id=0,gt_count=3,ap50=.8,ap50_95=.5)]),fixed=dict(overall=dict(f1=.8,recall=.8)))
        candidate=copy.deepcopy(baseline)
        candidate['official']['map50_95']=.495
        self.assertTrue(accuracy_gate(baseline,candidate)['passed'])
        candidate['official']['per_class'][0]['ap50']=.779
        self.assertFalse(accuracy_gate(baseline,candidate)['passed'])
        candidate=copy.deepcopy(baseline);candidate['fixed']['overall']['recall']=None
        self.assertFalse(accuracy_gate(baseline,candidate)['passed'])

    def test_frozen_profile(self):
        profile=Path(__file__).resolve().parents[1]/'models/inference_profile.json'
        self.assertEqual(len(validate_profile(profile)),64)
if __name__=='__main__': unittest.main()
