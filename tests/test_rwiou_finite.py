#改进0：验证极端尺寸logit和退化框不会在RWIoU前向或反向中产生NaN/Inf。
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_box_utils():
    scipy_module = types.ModuleType('scipy')
    scipy_spatial = types.ModuleType('scipy.spatial')
    scipy_spatial.Delaunay = object
    scipy_module.spatial = scipy_spatial
    sys.modules['scipy'] = scipy_module
    sys.modules['scipy.spatial'] = scipy_spatial

    package_paths = {
        'pcdet': ROOT / 'pcdet',
        'pcdet.utils': ROOT / 'pcdet' / 'utils',
        'pcdet.ops': ROOT / 'pcdet' / 'ops',
        'pcdet.ops.roiaware_pool3d': ROOT / 'pcdet' / 'ops' / 'roiaware_pool3d',
    }
    for name, path in package_paths.items():
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package

    sys.modules['pcdet.utils.common_utils'] = types.ModuleType(
        'pcdet.utils.common_utils'
    )
    sys.modules['pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils'] = (
        types.ModuleType('pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils')
    )

    module_name = 'pcdet.utils.box_utils'
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / 'pcdet/utils/box_utils.py'
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


box_utils = load_box_utils()


class TestFiniteRWIoU(unittest.TestCase):
    def test_extreme_logits_have_finite_gradients(self):
        prediction = torch.tensor([[
            0.0, 0.0, 0.0, 100.0, -100.0, 100.0, 0.0, 0.0
        ]], requires_grad=True)
        target = torch.tensor([[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
        ]])
        distance_penalty, rotated_iou = box_utils.get_rwiou(
            prediction, target
        )
        loss = distance_penalty.sum() + rotated_iou.sum()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(prediction.grad).all())


if __name__ == '__main__':
    unittest.main()
#end
