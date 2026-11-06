"""合成夹具与协议单元测试；测试数值不代表实测模型性能。"""
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import cv2
import numpy as np
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import Format
from ultralytics.data.dataset import YOLODataset
from steel_defect.augmentation import (MeanCenteredContrast, ContrastDataset,
    ContrastTrainer, validate_contrast_config)

CONFIG = dict(probability=.5, alpha_min=.7, alpha_max=1.3)


class ContrastTests(unittest.TestCase):
    def labels(self):
        image = np.repeat(np.array([[0, 60], [120, 240]], dtype=np.uint8)[..., None], 3, axis=2)
        return dict(img=image, cls=np.array([[1]]), instances=object(), shape=image.shape)

    def test_probability_zero_identity_and_rng_unchanged(self):
        labels = self.labels(); state = random.getstate()
        out = MeanCenteredContrast(0, .7, 1.3)(labels)
        self.assertIs(out, labels); self.assertEqual(state, random.getstate())

    def test_alpha_one_identity(self):
        labels = self.labels()
        self.assertIs(MeanCenteredContrast(1, 1, 1)(labels), labels)

    def test_known_alpha_round_clip_shape_labels_and_cache_preserved(self):
        labels = self.labels(); before = labels['img'].copy()
        out = MeanCenteredContrast(1, 1.5, 1.5)(labels)
        expected = np.repeat(np.array([[0, 38], [128, 255]], dtype=np.uint8)[..., None], 3, axis=2)
        np.testing.assert_array_equal(out['img'], expected)
        np.testing.assert_array_equal(labels['img'], before)
        self.assertEqual(out['img'].shape, before.shape)
        self.assertEqual(out['img'].dtype, np.uint8)
        self.assertIs(out['instances'], labels['instances'])
        self.assertIs(out['cls'], labels['cls'])
        self.assertFalse(np.shares_memory(out['img'], labels['img']))

    def test_config_rejects_invalid(self):
        for bad in [None, {}, dict(CONFIG, other=1), dict(CONFIG, probability=float('nan')),
                    dict(CONFIG, probability=True), dict(CONFIG, alpha_min=2), dict(CONFIG, probability=2)]:
            with self.assertRaises(ValueError): validate_contrast_config(bad)

    def test_real_dataset_train_val_and_close_mosaic_rebuild(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); images = root/'images/train'; labels = root/'labels/train'
            images.mkdir(parents=True); labels.mkdir(parents=True)
            image = np.repeat(np.arange(32, dtype=np.uint8)[None, :, None], 32, axis=0)
            image = np.repeat(image, 3, axis=2)
            cv2.imwrite(str(images/'a.png'), image)
            (labels/'a.txt').write_text('0 0.5 0.5 0.5 0.5\n')
            cfg = get_cfg(); cfg.imgsz=64; cfg.workers=0; cfg.cache=False
            trainer = object.__new__(ContrastTrainer)
            trainer.args=cfg; trainer.model=None; trainer.data={'names':{0:'defect'},'nc':1,'channels':3}
            trainer.contrast_config=CONFIG; trainer.save_dir=root
            train = trainer.build_dataset(str(images), mode='train', batch=1)
            val = trainer.build_dataset(str(images), mode='val', batch=1)
            self.assertIs(type(val), YOLODataset); self.assertIsInstance(train, ContrastDataset)
            original = [x['bboxes'].copy() for x in train.labels]
            for rebuild in range(3):
                if rebuild: train.close_mosaic(copy.deepcopy(cfg))
                ts = train.transforms.transforms
                self.assertEqual(sum(isinstance(t,MeanCenteredContrast) for t in ts),1)
                self.assertIsInstance(ts[-1],Format); self.assertIsInstance(ts[-2],MeanCenteredContrast)
                self.assertEqual(sum(isinstance(t,MeanCenteredContrast) for t in val.transforms.transforms),0)
            sample=train[0]; self.assertEqual(tuple(sample['img'].shape),(3,64,64))
            np.testing.assert_array_equal(train.labels[0]['bboxes'],original[0])
            events=[json.loads(s) for s in (root/'contrast_transform_audit.jsonl').read_text().splitlines()]
            self.assertEqual([e['contrast_count'] for e in events],[1,1,1])
            self.assertEqual(events[-1]['mosaic'],0)

    def test_control_changes_only_declared_factor(self):
        import yaml
        root=Path(__file__).resolve().parents[1]
        baseline=yaml.safe_load((root/'configs/baseline.yaml').read_text())
        control=yaml.safe_load((root/'configs/control.yaml').read_text())
        self.assertEqual(control.pop('contrast_augmentation'),CONFIG)
        control['stage']='baseline'
        self.assertEqual(control,baseline)


if __name__=='__main__': unittest.main()
