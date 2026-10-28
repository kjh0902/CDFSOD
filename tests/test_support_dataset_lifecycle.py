"""Exercise the real COCO parsing/serialization lifecycle without MMCV.

MMEngine BaseDataset and pycocotools are real. Only the detection base's image
transforms and package registration are replaced; CocoDataset.load_data_list,
parse_data_info, filter_data and the repository COCO API run unchanged.
"""
import copy
import importlib.util
import json
import os.path as osp
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import warnings

import torch

from visual_test_support import execute, node, sample, source, visual
from test_support_visual import SupportModel


@unittest.skipUnless(importlib.util.find_spec('mmengine') and
                     importlib.util.find_spec('pycocotools'),
                     'MMEngine and pycocotools are required')
class SupportDatasetLifecycleTests(unittest.TestCase):
    def dataset_class(self):
        import pycocotools
        from pycocotools.coco import COCO
        from mmengine.dataset import BaseDataset
        from mmengine.fileio import get_local_path

        class DetectionBase(BaseDataset):
            def __init__(self, backend_args=None, return_classes=True, **kwargs):
                self.backend_args = backend_args
                self.return_classes = return_classes
                self.caption_prompt = None
                super().__init__(**kwargs)

        env = dict(copy=copy, osp=osp, warnings=warnings, pycocotools=pycocotools,
                   _COCO=COCO, BaseDetDataset=DetectionBase, get_local_path=get_local_path)
        execute([node(source('mmdet/datasets/api_wrappers/coco_api.py'), 'COCO')], env)
        cls = node(source('mmdet/datasets/coco.py'), 'CocoDataset')
        cls.decorator_list = []
        execute([cls], env)
        return env['CocoDataset']

    def config(self, folder, categories=('a', 'b')):
        annotation = dict(
            images=[dict(id=11, file_name='unused.jpg', width=64, height=64)],
            categories=[dict(id=17, name=categories[0]), dict(id=3, name=categories[1])],
            annotations=[dict(id=i + 1, image_id=11, category_id=cat_id,
                              bbox=[8, 8, 32, 32], area=1024, iscrowd=0)
                         for i, cat_id in enumerate((17, 3))])
        (Path(folder) / 'support.json').write_text(json.dumps(annotation), encoding='utf8')
        return dict(type='CocoDataset', data_root=folder, ann_file='support.json',
                    data_prefix=dict(img='train/'), metainfo=dict(classes=('a', 'b')),
                    test_mode=True, pipeline=[], filter_cfg=None)

    def registry(self, dataset_class):
        registry = ModuleType('mmdet.registry')
        registry.DATASETS = NS(build=Mock(side_effect=lambda cfg: dataset_class(
            **{key: value for key, value in cfg.items() if key != 'type'})))
        return registry

    def test_actual_coco_lifecycle_loader_and_iteration_recomputation(self):
        from mmengine.fileio import load

        with tempfile.TemporaryDirectory() as folder:
            model = SupportModel()
            del model._support_loader
            model.support_visual_cfg['dataset'] = self.config(folder)
            registry = self.registry(self.dataset_class())
            with patch.dict(sys.modules, {'mmdet.registry': registry}), \
                    patch('mmengine.fileio.load', wraps=load) as read:
                rng = torch.get_rng_state().clone()
                loader = model._get_support_loader()
                self.assertFalse(hasattr(loader.dataset, 'coco'))
                self.assertTrue(loader.dataset._fully_initialized)
                self.assertTrue(loader.dataset.serialize_data)
                self.assertEqual(loader.dataset.cat_ids, [17, 3])
                self.assertEqual(loader.dataset.cat2label, {17: 0, 3: 1})
                read.assert_called_once_with(str(Path(folder) / 'support.json'), backend_args=None)
                self.assertIs(loader, model._get_support_loader())

                parsed = loader.dataset.get_data_info(0)
                # No image decoding/MMCV: preserve the actual parsed GT labels.
                instances = parsed['instances']
                model.data_preprocessor = lambda batch, training: dict(
                    inputs=torch.full((1, 3, 64, 64), 2.), data_samples=[sample(
                        [item['bbox'] for item in instances],
                        [item['bbox_label'] for item in instances])])
                first = model.support_prototypes()
                second = model.support_prototypes()
                torch.testing.assert_close(first, torch.full((2, 256), 2.))
                torch.testing.assert_close(second, first)
                self.assertFalse(first.requires_grad)
                self.assertEqual(model.backbone.calls, 2)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                registry.DATASETS.build.assert_called_once()
                read.assert_called_once()

    def test_category_label_order_validation_survives_coco_deletion(self):
        with tempfile.TemporaryDirectory() as folder:
            model = SupportModel()
            del model._support_loader
            model.support_visual_cfg['dataset'] = self.config(folder, categories=('b', 'a'))
            registry = self.registry(self.dataset_class())
            with patch.dict(sys.modules, {'mmdet.registry': registry}):
                with self.assertRaisesRegex(ValueError, 'category order'):
                    model._get_support_loader()
            self.assertIsNone(getattr(model, '_support_loader', None))


if __name__ == '__main__':
    unittest.main()
