#改进10：DGR及共享的稀疏几何证据；FP32、停止梯度、batch隔离，不增加坐标或稠密BEV。
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseGeometryEvidence(nn.Module):
    """Detached local descriptors, NOT measured visibility or uncertainty.

    Bbox layout: offset x/y, absolute z, log l/w/h, raw cos/sin logits.
    Decode neighbor centers to a common metric frame before comparing them.
    Loops are over chunks/offset definitions, never individual GPU voxels.
    """

    def __init__(self, channels, voxel_size, point_cloud_range, stride,
                 radii=(1, 3), chunk_size=2048):
        super().__init__()
        if channels < 1 or chunk_size < 1:
            raise ValueError('Geometry channels/chunk_size must be positive')
        if not radii or any(int(r) != r or r <= 0 for r in radii) or len(set(radii)) != len(radii):
            raise ValueError('Geometry radii must be distinct positive integers')
        self.channels, self.chunk_size = channels, chunk_size
        self.xy_step = tuple(float(v) * stride for v in voxel_size[:2])
        if len(self.xy_step) != 2 or any(not math.isfinite(v) or v <= 0 for v in self.xy_step):
            raise ValueError('Geometry voxel_size * stride must be finite and positive')
        self.xy_min = tuple(float(v) for v in point_cloud_range[:2])
        self.range_normalizer = max(1e-4, math.hypot(
            max(abs(point_cloud_range[0]), abs(point_cloud_range[3])),
            max(abs(point_cloud_range[1]), abs(point_cloud_range[4]))))
        offsets = [(dy * r, dx * r) for r in radii
                   for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
        self.register_buffer('offsets', torch.tensor(offsets, dtype=torch.long), persistent=False)
        self.descriptor_dim = 2 * channels + 24

    @staticmethod
    def keys(indices, height, width):
        return (indices[:, 0] * height + indices[:, 1]) * width + indices[:, 2]

    def neighbor_rows(self, coords, keys, order, height, width):
        yx = coords[:, None, 1:] + self.offsets[None]
        bounded = ((yx[..., 0] >= 0) & (yx[..., 0] < height)
                   & (yx[..., 1] >= 0) & (yx[..., 1] < width))
        queries = (coords[:, None, 0] * height + yx[..., 0]) * width + yx[..., 1]
        positions = torch.searchsorted(keys, queries.contiguous())
        safe = positions.clamp(max=keys.numel() - 1)
        valid = bounded & (positions < keys.numel()) & (keys[safe] == queries)
        return order[safe], valid

    def descriptors(self, features, indices, spatial_shape, boxes, logits, quality):
        n = features.shape[0]
        if (features.shape != (n, self.channels) or indices.shape != (n, 3)
                or boxes.shape != (n, 8) or quality.shape != (n, 1)
                or logits.ndim != 2 or logits.shape[0] != n or logits.shape[1] < 1):
            raise ValueError('Geometry requires aligned NxC / Nx3 / Nx8 / NxK / Nx1 tensors')
        height, width = int(spatial_shape[0]), int(spatial_shape[1])
        if min(height, width) < 1:
            raise ValueError('Geometry spatial shape must be positive')
        # Exit no_grad BEFORE yielding: callers must retain adapter gradients.
        with torch.no_grad():
            norm = F.layer_norm(features.detach().float().clamp(-1e4, 1e4),
                                (self.channels,), eps=1e-4).clamp(-4, 4)
            unit = F.normalize(norm, dim=1, eps=1e-4)
            coords = indices.long()
            keys, order = self.keys(coords, height, width).sort()
            raw = boxes.detach().float()
            step = raw.new_tensor(self.xy_step)
            # Bounds are for descriptors, not silently replacing detector output.
            xy = (coords[:, [2, 1]].float() + raw[:, :2].clamp(-1e4, 1e4)) * step
            xy = xy + raw.new_tensor(self.xy_min)
            centers = torch.cat([xy, raw[:, 2:3].clamp(-1e4, 1e4)], dim=1)
            log_dims = raw[:, 3:6].clamp(-5, 5)
            dims = log_dims.exp().clamp_min(0.1)
            orientation = raw[:, 6:8].sigmoid() * 2 - 1
            own_box = torch.cat([(raw[:, :2].clamp(-100, 100) * step).clamp(-4, 4),
                                 raw[:, 2:3].clamp(-4, 4), log_dims / 5, orientation], dim=1)
            classes = logits.detach().argmax(dim=1)
            iou = ((quality.detach().float() + 1) * 0.5).clamp(0, 1)
        for start in range(0, n, self.chunk_size):
            stop = min(n, start + self.chunk_size)
            with torch.no_grad():
                rows, valid = self.neighbor_rows(coords[start:stop], keys, order, height, width)
                valid = valid & (classes[rows] == classes[start:stop, None])
                center_delta = ((centers[rows] - centers[start:stop, None])
                                / dims[start:stop, None]).clamp(-4, 4)
                size_delta = (log_dims[rows] - log_dims[start:stop, None]).clamp(-4, 4)
                # No inverse variance or inverse point count, including isolated voxels.
                affinity = (unit[start:stop, None] * unit[rows]).sum(-1).clamp(0, 1)
                agreement = torch.exp(-center_delta.square().mean(-1)
                                      - size_delta.square().mean(-1))
                weights = valid.float() * affinity * agreement
                total = weights.sum(1, keepdim=True)
                normalized_weights = weights / total.clamp_min(1e-4)
                mean = lambda values: (values * normalized_weights[..., None]).sum(1)
                radial = (centers[start:stop, :2].norm(dim=1, keepdim=True)
                          / self.range_normalizer).clamp(0, 1)
                descriptor = torch.cat([
                    norm[start:stop], mean(norm[rows]), own_box[start:stop],
                    mean(center_delta), mean(center_delta.abs()),
                    mean(size_delta), mean(size_delta.abs()),
                    valid.float().mean(1, keepdim=True), total / self.offsets.shape[0],
                    radial, iou[start:stop],
                ], dim=1)
                supported = (total > 1e-4).float()
            yield start, stop, descriptor, supported


def route_primary_gradient(base_loss, adapted_loss):
    """Explicit surrogate: report adapted loss, keep baseline primary gradients.

    Requires detached adapter inputs/targets. Not a conventional shared loss;
    global clipping and AMP can still couple otherwise separated gradients.
    """
    return adapted_loss + (base_loss - base_loss.detach())


class DecoupledGeometryRefinement(nn.Module):
    def __init__(self, channels, voxel_size, point_cloud_range, stride,
                 hidden_channels=32, max_center_delta=0.1, max_log_size_delta=0.1,
                 radii=(1, 3), chunk_size=2048):
        super().__init__()
        if (hidden_channels < 1 or not 0 < max_center_delta <= 0.5
                or not 0 < max_log_size_delta <= 0.25):
            raise ValueError('Invalid DGR hidden size or bounded residual limits')
        self.evidence = SparseGeometryEvidence(channels, voxel_size, point_cloud_range,
                                               stride, radii, chunk_size)
        self.adapter = nn.Sequential(nn.Linear(self.evidence.descriptor_dim, hidden_channels),
                                     nn.ReLU(), nn.Linear(hidden_channels, 6))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        # xy offsets are encoded in BEV cells, z in metres, sizes in log metres.
        self.delta_limits = (max_center_delta / self.evidence.xy_step[0],
                             max_center_delta / self.evidence.xy_step[1], max_center_delta,
                             max_log_size_delta, max_log_size_delta, max_log_size_delta)

    def forward(self, features, indices, spatial_shape, boxes, logits, quality):
        with torch.cuda.amp.autocast(enabled=False):
            outputs = []
            for start, stop, descriptor, supported in self.evidence.descriptors(
                    features, indices, spatial_shape, boxes, logits, quality):
                delta = self.adapter(descriptor).tanh() * descriptor.new_tensor(self.delta_limits)
                base = boxes[start:stop].detach().float()
                outputs.append(torch.cat([base[:, :6] + delta * supported, base[:, 6:]], dim=1))
            if not outputs:
                return boxes.detach().float() + sum(p.sum() * 0 for p in self.parameters())
            return torch.cat(outputs, dim=0)
#end


#改进11：GCQC仅校准已有IoU分支；不改变分类、候选坐标、NMS阈值或IoU分数融合权重。
class GeometryConsistencyQualityCalibration(nn.Module):
    def __init__(self, channels, voxel_size, point_cloud_range, stride,
                 hidden_channels=16, max_quality_delta=0.2, radii=(1, 3), chunk_size=2048):
        super().__init__()
        if hidden_channels < 1 or not 0 < max_quality_delta <= 0.5:
            raise ValueError('Invalid GCQC hidden size or bounded quality limit')
        self.max_quality_delta = max_quality_delta
        self.evidence = SparseGeometryEvidence(channels, voxel_size, point_cloud_range,
                                               stride, radii, chunk_size)
        self.adapter = nn.Sequential(nn.Linear(self.evidence.descriptor_dim, hidden_channels),
                                     nn.ReLU(), nn.Linear(hidden_channels, 1))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, features, indices, spatial_shape, boxes, logits, quality):
        with torch.cuda.amp.autocast(enabled=False):
            outputs = []
            for start, stop, descriptor, supported in self.evidence.descriptors(
                    features, indices, spatial_shape, boxes, logits, quality):
                delta = self.max_quality_delta * self.adapter(descriptor).tanh() * supported
                outputs.append(quality[start:stop].detach().float() + delta)
            if not outputs:
                return quality.detach().float() + sum(p.sum() * 0 for p in self.parameters())
            return torch.cat(outputs, dim=0)
#end
