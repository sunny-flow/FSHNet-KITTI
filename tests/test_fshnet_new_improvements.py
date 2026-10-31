#改进0：为五项创新提供不依赖spconv CUDA扩展的最小测试桩。
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]


class AttrDict(dict):
    __getattr__ = dict.__getitem__


class FakeSparseTensor:
    def __init__(self, features, indices, spatial_shape, batch_size=1):
        self.features = features
        self.indices = indices
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size

    def replace_feature(self, features):
        return FakeSparseTensor(
            features, self.indices, self.spatial_shape, self.batch_size
        )


class NoOpSparseLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, value):
        return value


class VFETemplate(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg


def scatter_mean(value, indices, dim=0):
    output = value.new_zeros((int(indices.max()) + 1,) + tuple(value.shape[1:]))
    output.index_add_(0, indices, value)
    counts = value.new_zeros((output.shape[0],))
    counts.index_add_(0, indices, value.new_ones((value.shape[0],)))
    return output / counts.clamp_min(1).reshape((-1,) + (1,) * (value.ndim - 1))


def install_test_stubs():
    package_paths = {
        'pcdet': ROOT / 'pcdet',
        'pcdet.models': ROOT / 'pcdet' / 'models',
        'pcdet.models.model_utils': ROOT / 'pcdet' / 'models' / 'model_utils',
        'pcdet.models.backbones_3d': ROOT / 'pcdet' / 'models' / 'backbones_3d',
        'pcdet.models.backbones_3d.vfe': ROOT / 'pcdet' / 'models' / 'backbones_3d' / 'vfe',
        'pcdet.utils': ROOT / 'pcdet' / 'utils',
    }
    for name, path in package_paths.items():
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package

    fake_spconv = types.ModuleType('spconv.pytorch')
    fake_spconv.SparseModule = nn.Module
    fake_spconv.SparseSequential = nn.Sequential
    fake_spconv.SubMConv2d = NoOpSparseLayer
    fake_spconv.SubMConv3d = NoOpSparseLayer
    fake_spconv.SparseConv2d = NoOpSparseLayer
    fake_spconv.SparseConv3d = NoOpSparseLayer
    fake_spconv.SparseInverseConv3d = NoOpSparseLayer
    fake_spconv.SparseConvTensor = FakeSparseTensor
    spconv_root = types.ModuleType('spconv')
    spconv_root.pytorch = fake_spconv
    sys.modules['spconv'] = spconv_root
    sys.modules['spconv.pytorch'] = fake_spconv

    spconv_utils = types.ModuleType('pcdet.utils.spconv_utils')
    spconv_utils.spconv = fake_spconv
    spconv_utils.replace_feature = lambda tensor, features: tensor.replace_feature(features)
    sys.modules[spconv_utils.__name__] = spconv_utils

    scatterformer_utils = types.ModuleType(
        'pcdet.models.model_utils.scatterformer_utils'
    )
    scatterformer_utils.rearrange = lambda value, pattern, h: (
        value.reshape(value.shape[0], h, value.shape[1] // h)
        if pattern == 'n (h c) -> n h c'
        else value.reshape(value.shape[0], -1)
    )
    scatterformer_utils.scatter_matmul_kv = lambda key, value, offsets, counts: (
        torch.einsum('nhd,nhe->hde', key, value).unsqueeze(0)
    )
    scatterformer_utils.scatter_matmul_qc = lambda query, kv, offsets, counts: (
        torch.einsum('nhd,hde->nhe', query, kv[0])
    )

    def scatter_add(value, indices, dim=0):
        output = value.new_zeros((int(indices.max()) + 1,) + tuple(value.shape[1:]))
        output.index_add_(0, indices, value)
        return output

    scatterformer_utils.torch_scatter = types.SimpleNamespace(
        scatter_add=scatter_add
    )
    sys.modules[scatterformer_utils.__name__] = scatterformer_utils

    hednet_utils = types.ModuleType('pcdet.models.model_utils.hednet_utils')
    hednet_utils.post_act_block_sparse_3d = lambda *args, **kwargs: NoOpSparseLayer()
    hednet_utils.post_act_block_sparse_2d = lambda *args, **kwargs: NoOpSparseLayer()
    hednet_utils.post_act_block_dense_2d = lambda *args, **kwargs: NoOpSparseLayer()
    sys.modules[hednet_utils.__name__] = hednet_utils

    torch_scatter_module = types.ModuleType('torch_scatter')
    torch_scatter_module.scatter_mean = scatter_mean
    sys.modules['torch_scatter'] = torch_scatter_module

    vfe_template = types.ModuleType(
        'pcdet.models.backbones_3d.vfe.vfe_template'
    )
    vfe_template.VFETemplate = VFETemplate
    sys.modules[vfe_template.__name__] = vfe_template

    dynamic_pillar = types.ModuleType(
        'pcdet.models.backbones_3d.vfe.dynamic_pillar_vfe'
    )
    dynamic_pillar.PFNLayerV2 = NoOpSparseLayer
    sys.modules[dynamic_pillar.__name__] = dynamic_pillar


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


install_test_stubs()
slotformer_utils = load_module(
    'pcdet.models.model_utils.slotformer_utils',
    'pcdet/models/model_utils/slotformer_utils.py'
)
backbone_utils = load_module(
    'pcdet.models.backbones_3d.fshnet_light',
    'pcdet/models/backbones_3d/fshnet_light.py'
)
vfe_utils = load_module(
    'pcdet.models.backbones_3d.vfe.dynamic_voxel_vfe',
    'pcdet/models/backbones_3d/vfe/dynamic_voxel_vfe.py'
)
#end


#改进1：验证LGME几何矩为有限值、保持体素顺序且对体素内点排列不敏感。
class TestLGME(unittest.TestCase):
    def test_local_geometry_moments(self):
        cluster = torch.tensor([
            [-0.5, 0.0, 0.0], [0.5, 0.0, 0.0],
            [0.0, -0.5, -0.5], [0.0, 0.5, 0.5],
        ])
        inverse = torch.tensor([0, 0, 1, 1])
        counts = torch.tensor([2, 2])
        voxel_size = torch.ones(3)
        moments = vfe_utils.DynamicVoxelVFE.compute_local_geometry_moments(
            cluster, inverse, counts, voxel_size
        )
        permuted = vfe_utils.DynamicVoxelVFE.compute_local_geometry_moments(
            cluster[[1, 0, 3, 2]], inverse, counts, voxel_size
        )
        self.assertEqual(tuple(moments.shape), (2, 7))
        self.assertTrue(torch.isfinite(moments).all())
        self.assertTrue(torch.allclose(moments, permuted))

    def test_false_does_not_build_parameters(self):
        base_cfg = AttrDict({
            'USE_NORM': True,
            'WITH_DISTANCE': False,
            'USE_ABSLOTE_XYZ': True,
            'NUM_FILTERS': [64, 64],
        })
        off_cfg = AttrDict(dict(base_cfg))
        off_cfg['USE_LOCAL_GEOMETRY_MOMENT_ENCODING'] = False

        original_cuda = torch.Tensor.cuda
        torch.Tensor.cuda = lambda tensor, *args, **kwargs: tensor
        try:
            baseline = vfe_utils.DynamicVoxelVFE(
                base_cfg, 4, [0.05, 0.05, 0.1],
                [1408, 1600, 40], [0, -40, -3, 70.4, 40, 1]
            )
            switches_off = vfe_utils.DynamicVoxelVFE(
                off_cfg, 4, [0.05, 0.05, 0.1],
                [1408, 1600, 40], [0, -40, -3, 70.4, 40, 1]
            )
        finally:
            torch.Tensor.cuda = original_cuda

        switches_off.load_state_dict(baseline.state_dict(), strict=True)
        self.assertFalse(hasattr(switches_off, 'local_geometry_encoder'))
#end


#改进2：验证DCBC使用占用均值、保持batch-aware BEV坐标且输出有限。
class TestDCBC(unittest.TestCase):
    def test_density_calibrated_bev_compression(self):
        input_features = torch.tensor([
                [2.0, 4.0], [4.0, 8.0], [3.0, 9.0]
            ], requires_grad=True)
        sparse = FakeSparseTensor(
            input_features,
            torch.tensor([
                [0, 0, 2, 3], [0, 1, 2, 3], [1, 0, 2, 3]
            ], dtype=torch.int32),
            [2, 8, 8], batch_size=2
        )
        output = backbone_utils.FSHNet_light.bev_out_density_calibrated(
            sparse, count_power=1.0
        )
        self.assertEqual(tuple(output.features.shape), (2, 2))
        self.assertTrue(torch.isfinite(output.features).all())
        self.assertTrue(torch.equal(
            output.indices,
            torch.tensor([[0, 2, 3], [1, 2, 3]], dtype=torch.int32)
        ))
        self.assertTrue(torch.allclose(
            output.features, torch.tensor([[3.0, 6.0], [3.0, 9.0]])
        ))
        output.features.square().mean().backward()
        self.assertTrue(torch.isfinite(input_features.grad).all())
#end


#改进3：验证ELGM保持128通道并可反向，且全部开关False时严格加载原backbone权重。
class TestELGM(unittest.TestCase):
    def test_output_shape_and_backward(self):
        layer = slotformer_utils.ELGMLayer(
            128, directions=2, win_size=12, shift=False,
            cwi=False, channel_ratio=0.5
        )
        sparse = FakeSparseTensor(
            torch.randn(8, 128, requires_grad=True),
            torch.tensor([
                [0, 0, 0], [0, 1, 1], [0, 2, 2], [0, 3, 3],
                [1, 0, 0], [1, 1, 1], [1, 2, 2], [1, 3, 3],
            ], dtype=torch.int32),
            [12, 12], batch_size=2
        )
        output = layer(sparse)
        self.assertEqual(tuple(output.features.shape), (8, 128))
        output.features.square().mean().backward()
        self.assertTrue(torch.isfinite(output.features).all())
        self.assertTrue(torch.isfinite(sparse.features.grad).all())

    def test_all_switches_false_load_original_state(self):
        baseline = backbone_utils.FSHNet_light({}, 4, [16, 16, 16])
        switches_off = backbone_utils.FSHNet_light({
            'USE_DENSITY_CALIBRATED_BEV_COMPRESSION': False,
            'USE_EFFICIENT_LOCAL_GLOBAL_MIXER': False,
        }, 4, [16, 16, 16])
        switches_off.load_state_dict(baseline.state_dict(), strict=True)
        self.assertTrue(all(
            isinstance(layer, slotformer_utils.SFLayer)
            for layer in switches_off.slotformer[0].sla_layers
        ))
#end


#改进0：测试文件直接运行入口。
if __name__ == '__main__':
    unittest.main()
#end
