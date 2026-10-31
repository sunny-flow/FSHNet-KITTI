#改进0：工程修复测试——验证三类均值指标和跨epoch最佳checkpoint持久化逻辑。
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]


class DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_train_utils():
    pcdet = types.ModuleType('pcdet')
    pcdet_utils = types.ModuleType('pcdet.utils')
    pcdet_utils.common_utils = types.SimpleNamespace()
    pcdet_utils.commu_utils = types.SimpleNamespace()
    pcdet.utils = pcdet_utils
    sys.modules['pcdet'] = pcdet
    sys.modules['pcdet.utils'] = pcdet_utils

    eval_package = types.ModuleType('eval_utils')
    eval_package.eval_utils = types.SimpleNamespace()
    sys.modules['eval_utils'] = eval_package
    return load_module(
        'epoch_eval_train_utils', ROOT / 'tools' / 'train_utils' / 'train_utils.py'
    )


class TestEpochEvaluation(unittest.TestCase):
    def test_mean_metric(self):
        evaluator = load_module(
            'isolated_kitti_eval_script', ROOT / 'tools' / 'eval_kitti_result.py'
        )
        metrics = {
            'Car_3d/moderate_R40': 70.0,
            'Pedestrian_3d/moderate_R40': 50.0,
            'Cyclist_3d/moderate_R40': 60.0,
        }
        output = evaluator.add_mean_metrics(
            metrics, ['Car', 'Pedestrian', 'Cyclist']
        )
        self.assertEqual(output['mean_3d/moderate_R40'], 60.0)

    def test_cpu_rotate_iou_criterion_conversion(self):
        evaluator = load_module(
            'isolated_kitti_eval_cpu_backend', ROOT / 'tools' / 'eval_kitti_result.py'
        )
        fake_iou = lambda boxes, query: np.array([[0.5]], dtype=np.float32)
        rotate_iou = evaluator.build_cpu_rotate_iou(fake_iou)
        boxes = np.array([[0.0, 0.0, 4.0, 2.0, 0.25]], dtype=np.float32)
        query = np.array([[1.0, 0.0, 2.0, 2.0, -0.25]], dtype=np.float32)

        self.assertAlmostEqual(float(rotate_iou(boxes, query, -1)[0, 0]), 0.5)
        self.assertAlmostEqual(float(rotate_iou(boxes, query, 0)[0, 0]), 0.5)
        self.assertAlmostEqual(float(rotate_iou(boxes, query, 1)[0, 0]), 1.0)
        self.assertAlmostEqual(float(rotate_iou(boxes, query, 2)[0, 0]), 4.0)

    def test_best_checkpoint_updates_only_on_improvement(self):
        train_utils = load_train_utils()
        logger = DummyLogger()
        model = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        with tempfile.TemporaryDirectory() as temporary_dir:
            ckpt_dir = Path(temporary_dir) / 'ckpt'
            ckpt_dir.mkdir()
            summary, is_best = train_utils.update_best_performance(
                model, optimizer, 1, 10,
                {'mean_3d/moderate_R40': 55.0},
                'mean_3d/moderate_R40', ckpt_dir, logger
            )
            self.assertTrue(is_best)
            self.assertEqual(summary['best_epoch'], 1)
            best_path = ckpt_dir / 'best_model.pth'
            self.assertTrue(best_path.exists())
            first_checkpoint = best_path.read_bytes()

            with torch.no_grad():
                model.weight.add_(1.0)
            summary, is_best = train_utils.update_best_performance(
                model, optimizer, 2, 20,
                {'mean_3d/moderate_R40': 54.0},
                'mean_3d/moderate_R40', ckpt_dir, logger
            )
            self.assertFalse(is_best)
            self.assertEqual(summary['best_epoch'], 1)
            self.assertEqual(best_path.read_bytes(), first_checkpoint)

            summary, is_best = train_utils.update_best_performance(
                model, optimizer, 3, 30,
                {'mean_3d/moderate_R40': 56.0},
                'mean_3d/moderate_R40', ckpt_dir, logger
            )
            self.assertTrue(is_best)
            self.assertEqual(summary['best_epoch'], 3)
            saved_checkpoint = torch.load(best_path, map_location='cpu')
            self.assertEqual(saved_checkpoint['epoch'], 3)

            with open(Path(temporary_dir) / 'best_metrics.json', 'r', encoding='utf-8') as f:
                persisted_summary = json.load(f)
            self.assertEqual(persisted_summary['best_value'], 56.0)
            self.assertEqual(persisted_summary['latest_epoch'], 3)

    def test_recover_best_checkpoint_from_existing_metrics(self):
        train_utils = load_train_utils()
        logger = DummyLogger()
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            ckpt_dir = root / 'ckpt'
            metrics_dir = root / 'eval' / 'epoch_1' / 'val'
            ckpt_dir.mkdir()
            metrics_dir.mkdir(parents=True)
            checkpoint = ckpt_dir / 'checkpoint_epoch_1.pth'
            checkpoint.write_bytes(b'epoch-one-checkpoint')
            (metrics_dir / 'kitti_metrics.json').write_text(
                json.dumps({'mean_3d/moderate_R40': 42.5}), encoding='utf-8'
            )

            train_utils.recover_best_performance(
                ckpt_dir, root / 'eval', 'mean_3d/moderate_R40', logger
            )
            self.assertEqual(
                (ckpt_dir / 'best_model.pth').read_bytes(),
                checkpoint.read_bytes()
            )
            summary = json.loads((root / 'best_metrics.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['best_epoch'], 1)
            self.assertEqual(summary['best_value'], 42.5)


if __name__ == '__main__':
    unittest.main()
#end
