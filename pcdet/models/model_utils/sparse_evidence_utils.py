#改进9：DSEC解耦稀疏证据校准；仅分类分支读取detached特征，保留坐标与原DSLA匹配输入。
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoupledSparseEvidenceCalibration(nn.Module):
    """Class-wise, bounded logit correction on existing [batch, y, x] rows.

    This is an experimental adapter, not a reproduction of a published model.
    Context and input scores are stop-gradient. No dense BEV, new coordinates,
    class softmax, new labels or trainable shared class projection are used.
    """

    def __init__(self, channels, num_classes, voxel_size, point_cloud_range,
                 feature_map_stride, hidden_channels=16, radii=(1, 3),
                 max_logit_delta=0.5, chunk_size=2048):
        super().__init__()
        if min(channels, num_classes, hidden_channels, chunk_size) < 1:
            raise ValueError('DSEC dimensions and chunk_size must be positive')
        if (not radii or any(int(r) != r or r <= 0 for r in radii)
                or len(set(radii)) != len(radii)):
            raise ValueError('DSEC_RADII must be distinct positive integers')
        if not 0.0 < max_logit_delta <= 1.0:
            raise ValueError('DSEC_MAX_LOGIT_DELTA must be in (0, 1]')
        if not feature_map_stride or feature_map_stride <= 0:
            raise ValueError('DSEC requires a positive feature_map_stride')
        self.channels = channels
        self.num_classes = num_classes
        self.radii = tuple(int(r) for r in radii)
        self.chunk_size = chunk_size
        self.max_logit_delta = max_logit_delta
        self.xy_step = tuple(float(v) * feature_map_stride for v in voxel_size[:2])
        self.xy_min = tuple(float(v) for v in point_cloud_range[:2])
        self.range_normalizer = max(1e-6, math.hypot(
            max(abs(point_cloud_range[0]), abs(point_cloud_range[3])),
            max(abs(point_cloud_range[1]), abs(point_cloud_range[4]))))
        directions = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                      if dy != 0 or dx != 0]
        offsets = [(dy * r, dx * r) for r in self.radii for dy, dx in directions]
        self.register_buffer('offsets', torch.tensor(offsets, dtype=torch.long),
                             persistent=False)
        # Each scale contributes C context values, occupancy and score evidence.
        descriptor_dim = channels * (1 + len(self.radii)) + 2 * len(self.radii) + 2
        self.class_adapters = nn.ModuleList([
            nn.Sequential(nn.Linear(descriptor_dim, hidden_channels), nn.ReLU(),
                          nn.Linear(hidden_channels, 1))
            for _ in range(num_classes)
        ])
        for adapter in self.class_adapters:
            nn.init.zeros_(adapter[-1].weight)
            nn.init.zeros_(adapter[-1].bias)

    @staticmethod
    def _keys(indices, height, width):
        return (indices[:, 0] * height + indices[:, 1]) * width + indices[:, 2]

    def _neighbor_rows(self, target, sorted_keys, source_order, height, width):
        """Exact bounded GPU lookup; invalid x/y never wrap into another batch."""
        yx = target[:, None, 1:] + self.offsets[None]
        valid_xy = ((yx[..., 0] >= 0) & (yx[..., 0] < height)
                    & (yx[..., 1] >= 0) & (yx[..., 1] < width))
        keys = ((target[:, None, 0] * height + yx[..., 0]) * width + yx[..., 1])
        positions = torch.searchsorted(sorted_keys, keys.contiguous())
        safe = positions.clamp(max=sorted_keys.numel() - 1)
        valid = valid_xy & (positions < sorted_keys.numel()) & (sorted_keys[safe] == keys)
        return source_order[safe], valid

    def forward(self, features, indices, spatial_shape, base_logits):
        if (features.ndim != 2 or features.shape[1] != self.channels
                or indices.shape != (features.shape[0], 3)
                or base_logits.shape != (features.shape[0], self.num_classes)):
            raise ValueError('DSEC expects aligned features, [batch,y,x] and class logits')
        height, width = int(spatial_shape[0]), int(spatial_shape[1])
        if height < 1 or width < 1:
            raise ValueError('DSEC spatial shape must be positive')
        with torch.cuda.amp.autocast(enabled=False):
            if features.shape[0] == 0:
                # Retain a zero gradient path so empty inputs support backward/DDP.
                zero = sum(p.sum() * 0.0 for p in self.parameters())
                return base_logits.detach().float() + zero
            with torch.no_grad():
                # Bounds precede variance/square computations; no nan_to_num hiding.
                normalized = F.layer_norm(features.detach().float().clamp(-1e4, 1e4),
                                          (self.channels,), eps=1e-4).clamp(-4.0, 4.0)
                unit_features = F.normalize(normalized, dim=1, eps=1e-4)
                probabilities = base_logits.detach().float().sigmoid()
                coordinates = indices.long()
                keys = self._keys(coordinates, height, width)
                sorted_keys, source_order = keys.sort()

            outputs = []
            # Loops are over bounded chunks/classes/scales, never individual voxels.
            for start in range(0, features.shape[0], self.chunk_size):
                stop = min(start + self.chunk_size, features.shape[0])
                with torch.no_grad():
                    rows, valid = self._neighbor_rows(
                        coordinates[start:stop], sorted_keys, source_order, height, width)
                    neighbor_features = normalized[rows]
                    agreement = (unit_features[start:stop, None] * unit_features[rows]).sum(-1)
                    affinity = agreement.clamp(0.0, 1.0).square() * valid.float()
                    radial_xy = (coordinates[start:stop, [2, 1]].float() + 0.5)
                    radial_xy = radial_xy * radial_xy.new_tensor(self.xy_step)
                    radial_xy = radial_xy + radial_xy.new_tensor(self.xy_min)
                    radial = (radial_xy.norm(dim=1, keepdim=True)
                              / self.range_normalizer).clamp(0.0, 1.0)
                    has_context = valid.any(dim=1, keepdim=True).float()
                class_outputs = []
                for class_idx, adapter in enumerate(self.class_adapters):
                    with torch.no_grad():
                        neighbor_scores = probabilities[rows, class_idx]
                        descriptor = [normalized[start:stop],
                                      probabilities[start:stop, class_idx:class_idx + 1], radial]
                        for scale_idx in range(len(self.radii)):
                            sl = slice(scale_idx * 8, (scale_idx + 1) * 8)
                            weights = affinity[:, sl] * (0.05 + neighbor_scores[:, sl])
                            context = (neighbor_features[:, sl] * weights[..., None]).sum(1)
                            context = context / weights.sum(1, keepdim=True).clamp_min(1e-4)
                            occupancy = valid[:, sl].float().mean(1, keepdim=True)
                            evidence = (neighbor_scores[:, sl] * affinity[:, sl]).mean(1, keepdim=True)
                            descriptor.extend([context, occupancy, evidence])
                        descriptor = torch.cat(descriptor, dim=1)
                    correction = self.max_logit_delta * torch.tanh(adapter(descriptor))
                    class_outputs.append(correction * has_context)
                delta = torch.cat(class_outputs, dim=1)
                outputs.append(base_logits[start:stop].detach().float() + delta)
            return torch.cat(outputs, dim=0)


def preserve_primary_classification_gradient(base_loss, calibrated_loss):
    """Report calibrated loss, route original gradient to the primary detector.

    base_loss uses original logits; calibrated_loss uses detached original logits
    plus adapter output. Thus dL/d(primary) = d(base_loss)/d(primary) and
    dL/d(adapter) = d(calibrated_loss)/d(adapter). Existing loss weights still
    apply once. Joint gradient clipping/AMP can still affect training dynamics.
    """
    return calibrated_loss + (base_loss - base_loss.detach())
#end
