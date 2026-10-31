"""CPU smoke check for directly registered parameters; not a detector test.

Run from the project root: python tests/test_optimizer_release_smoke.py
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

import torch
from torch import nn
from tools.train_utils.optimization import build_optimizer


class Config(dict):
    __getattr__ = dict.__getitem__


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rpdi_scale = nn.Parameter(torch.zeros(1))
        self.projection = nn.Linear(2, 2)
        self.norm = nn.BatchNorm1d(2)

    def forward(self, x):
        return self.norm(self.projection(x)) + self.rpdi_scale


class OptimizerCoverageSmoke(unittest.TestCase):
    def test_nonleaf_parameter_is_updated_once(self):
        model = ToyModel()
        cfg = Config(OPTIMIZER='adam_onecycle', LR=0.003, WEIGHT_DECAY=0.0)
        optimizer = build_optimizer(model, cfg)
        params = [p for group in optimizer.param_groups for p in group['params']]
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(len(params), len({id(p) for p in params}))
        self.assertEqual({id(p) for p in model.parameters() if p.requires_grad},
                         {id(p) for p in params})
        self.assertEqual(sum(p is model.rpdi_scale for p in params), 1)
        model(torch.tensor([[1., 2.], [2., 1.]])).sum().backward()
        self.assertNotEqual(float(model.rpdi_scale.grad), 0.)
        optimizer.step()
        self.assertNotEqual(float(model.rpdi_scale.detach()), 0.)


if __name__ == '__main__':
    unittest.main()
