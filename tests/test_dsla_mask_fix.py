#改进0：为DSLA工程修复提供不依赖新增第三方包的最小单元测试。
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]


class DummyLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return torch.tensor(0.0)


def install_test_stubs():
    package_paths = {
        'pcdet': ROOT / 'pcdet',
        'pcdet.models': ROOT / 'pcdet' / 'models',
        'pcdet.models.model_utils': ROOT / 'pcdet' / 'models' / 'model_utils',
        'pcdet.models.dense_heads': ROOT / 'pcdet' / 'models' / 'dense_heads',
        'pcdet.utils': ROOT / 'pcdet' / 'utils',
        'pcdet.ops': ROOT / 'pcdet' / 'ops',
        'pcdet.ops.iou3d_nms': ROOT / 'pcdet' / 'ops' / 'iou3d_nms',
    }
    for name, path in package_paths.items():
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package

    fake_spconv = types.ModuleType('spconv.pytorch')
    fake_spconv.SubMConv2d = nn.Linear
    fake_spconv.SparseSequential = nn.Sequential
    spconv_root = types.ModuleType('spconv')
    spconv_root.pytorch = fake_spconv
    sys.modules['spconv'] = spconv_root
    sys.modules['spconv.pytorch'] = fake_spconv

    spconv_utils = types.ModuleType('pcdet.utils.spconv_utils')
    spconv_utils.spconv = fake_spconv
    sys.modules[spconv_utils.__name__] = spconv_utils

    for name in ('centernet_utils', 'model_nms_utils'):
        module_name = 'pcdet.models.model_utils.' + name
        sys.modules[module_name] = types.ModuleType(module_name)

    loss_utils = types.ModuleType('pcdet.utils.loss_utils')
    loss_utils.DynamicPositiveMask = DummyLoss
    loss_utils.FocalLossSparse = DummyLoss
    loss_utils.RWIoULoss = DummyLoss
    loss_utils.SlotFormerIoULoss = DummyLoss
    sys.modules[loss_utils.__name__] = loss_utils

    iou_utils = types.ModuleType('pcdet.ops.iou3d_nms.iou3d_nms_utils')
    sys.modules[iou_utils.__name__] = iou_utils


def load_dense_head_module():
    module_name = 'pcdet.models.dense_heads.sparse_dynamic_head'
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / 'pcdet/models/dense_heads/sparse_dynamic_head.py'
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


install_test_stubs()
dense_head_utils = load_dense_head_module()


class TestDSLAMaskFix(unittest.TestCase):
    def test_every_gt_gets_nearest_candidate(self):
        head = dense_head_utils.SparseDynamicHead.__new__(
            dense_head_utils.SparseDynamicHead
        )
        nn.Module.__init__(head)
        head.dynamic_pos_num = 5
        head.point_cloud_range = torch.tensor([0.0, -10.0, -3.0, 20.0, 10.0, 1.0])
        head.voxel_size = torch.tensor([1.0, 1.0, 1.0])
        head.fix_dsla_mask = True
        head.use_shape_normalized_dsla = False

        gt_boxes = torch.tensor([
            [2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            [8.0, 2.0, 0.0, 1.0, 1.0, 1.0, 0.0, 2.0],
            [15.0, -2.0, 0.0, 1.0, 1.0, 1.0, 0.0, 3.0],
        ])
        spatial_indices = torch.tensor([
            [2, 10], [8, 12], [15, 8], [1, 1], [5, 5], [10, 10], [19, 19]
        ])

        _, _, _, mask, _ = head.assign_target_of_single_head(
            3, gt_boxes, (20, 20), 1, spatial_indices, num_max_objs=8
        )
        self.assertTrue(torch.equal(mask[:3, 0], torch.ones(3, dtype=mask.dtype)))
#end


#改进4：验证SN-DSLA在候选不足时保持固定形状、最近候选有效且heatmap不写入填充候选。
class TestShapeNormalizedDSLA(unittest.TestCase):
    @staticmethod
    def build_head():
        head = dense_head_utils.SparseDynamicHead.__new__(
            dense_head_utils.SparseDynamicHead
        )
        nn.Module.__init__(head)
        head.dynamic_pos_num = 5
        head.point_cloud_range = torch.tensor([
            0.0, -10.0, -3.0, 20.0, 10.0, 1.0
        ])
        head.voxel_size = torch.tensor([1.0, 1.0, 1.0])
        head.fix_dsla_mask = True
        head.use_shape_normalized_dsla = True
        head.sn_dsla_min_half_extent = 1.0
        head.sn_dsla_max_distance = 1.0
        return head

    def test_mask_consistent_candidates(self):
        head = self.build_head()

        gt_boxes = torch.tensor([
            [2.0, -8.0, 0.0, 0.6, 0.8, 1.7, 0.0, 1.0],
            [16.0, 6.0, 0.0, 4.0, 1.8, 1.6, 0.0, 2.0],
        ])
        spatial_indices = torch.tensor([
            [2, 2], [16, 16], [10, 10]
        ])

        heatmap, _, inds, mask, distances = head.assign_target_of_single_head(
            3, gt_boxes, (20, 20), 1, spatial_indices, num_max_objs=8
        )
        self.assertEqual(tuple(inds.shape), (8, 5))
        self.assertTrue(torch.equal(mask[:2, 0], torch.ones(2, dtype=mask.dtype)))
        self.assertEqual(int(mask[:2, 3:].sum()), 0)
        self.assertEqual(int(heatmap.sum()), int(mask[:2].sum()))
        self.assertTrue(torch.isfinite(distances).all())

    def test_empty_gt_is_supported(self):
        head = self.build_head()
        gt_boxes = torch.zeros((0, 8))
        spatial_indices = torch.tensor([[1, 1], [2, 2]])
        heatmap, targets, inds, mask, distances = (
            head.assign_target_of_single_head(
                3, gt_boxes, (20, 20), 1, spatial_indices,
                num_max_objs=8
            )
        )
        self.assertEqual(tuple(heatmap.shape), (3, 2))
        self.assertEqual(tuple(targets.shape), (8, 5, 8))
        self.assertEqual(int(mask.sum()), 0)
        self.assertEqual(int(inds.sum()), 0)
        self.assertTrue(torch.isfinite(distances).all())
#end


#改进5：验证BOPR对零范数和极小范数方向输入保持有限且反向梯度有界。
class TestBoundedOrientationProjection(unittest.TestCase):
    def test_projection_is_finite(self):
        orientation = torch.tensor([
            [0.0, 0.0], [0.1, 0.0], [3.0, 4.0]
        ], requires_grad=True)
        projected = dense_head_utils.SparseDynamicHead.bounded_orientation_projection(
            orientation, min_norm=0.25
        )
        projected.square().sum().backward()
        self.assertTrue(torch.isfinite(projected).all())
        self.assertTrue(torch.isfinite(orientation.grad).all())
        self.assertTrue((projected.norm(dim=1) <= 1.0 + 1e-6).all())
#end


#改进0：测试文件直接运行入口。
if __name__ == '__main__':
    unittest.main()
#end
