from functools import partial
import torch
import torch.nn as nn

from ...utils.spconv_utils import replace_feature, spconv
from pcdet.models.model_utils.hednet_utils import post_act_block_sparse_3d, post_act_block_sparse_2d, post_act_block_dense_2d
from pcdet.models.model_utils.slotformer_utils import SlotFormer

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class FSHNet_light(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        spconv_kernel_sizes = model_cfg.get('SPCONV_KERNEL_SIZES', [3, 3, 3, 3])
        channels = model_cfg.get('CHANNELS', [16, 32, 64, 128, 128])
        out_channel = model_cfg.get('OUT_CHANNEL', 128)

        #改进14：主BEV高度聚合的FP32数值修复开关；默认关闭以保留原模型输出。
        self.stable_bev_fp32 = bool(self.model_cfg.get('STABLE_BEV_FP32', False))
        #end

        #改进2：DCBC仅对3D到BEV的高度聚合做FP32密度校准；默认False严格保留原求和路径。
        self.use_density_calibrated_bev_compression = self.model_cfg.get(
            'USE_DENSITY_CALIBRATED_BEV_COMPRESSION', False
        )
        if self.use_density_calibrated_bev_compression:
            self.dcbc_count_power = float(self.model_cfg.get(
                'DCBC_COUNT_POWER', 0.5
            ))
            if not 0.0 <= self.dcbc_count_power <= 1.0:
                raise ValueError('DCBC_COUNT_POWER must be in [0, 1]')
        #end

        #改进6：RDSC距离-稀疏度校准默认关闭；仅开启时构建有界、零初始化的通道校准器。
        self.use_range_density_sparse_calibration = self.model_cfg.get(
            'USE_RANGE_DENSITY_SPARSE_CALIBRATION', False
        )
        if self.use_range_density_sparse_calibration:
            self.rdsc_hidden_channels = int(self.model_cfg.get(
                'RDSC_HIDDEN_CHANNELS', 16
            ))
            self.rdsc_density_cap = float(self.model_cfg.get(
                'RDSC_DENSITY_CAP', 16.0
            ))
            self.rdsc_max_residual = float(self.model_cfg.get(
                'RDSC_MAX_RESIDUAL', 0.1
            ))
            if self.rdsc_hidden_channels <= 0:
                raise ValueError('RDSC_HIDDEN_CHANNELS must be positive')
            if self.rdsc_density_cap <= 1.0:
                raise ValueError('RDSC_DENSITY_CAP must exceed 1')
            if not 0.0 < self.rdsc_max_residual <= 0.5:
                raise ValueError('RDSC_MAX_RESIDUAL must be in (0, 0.5]')
            self.range_density_calibrator = nn.Sequential(
                nn.Linear(2, self.rdsc_hidden_channels),
                nn.ReLU(),
                nn.Linear(self.rdsc_hidden_channels, channels[3]),
            )
            nn.init.zeros_(self.range_density_calibrator[-1].weight)
            nn.init.zeros_(self.range_density_calibrator[-1].bias)
        #end

        #改进7：RPDI保坐标细节注入默认关闭；零尺度初始化保证初始输出与原路径完全一致。
        self.use_range_preserving_detail_injection = self.model_cfg.get(
            'USE_RANGE_PRESERVING_DETAIL_INJECTION', False
        )
        if self.use_range_preserving_detail_injection:
            self.rpdi_gate_reduction = int(self.model_cfg.get(
                'RPDI_GATE_REDUCTION', 8
            ))
            self.rpdi_max_residual = float(self.model_cfg.get(
                'RPDI_MAX_RESIDUAL', 0.1
            ))
            if self.rpdi_gate_reduction <= 0:
                raise ValueError('RPDI_GATE_REDUCTION must be positive')
            if not 0.0 < self.rpdi_max_residual <= 0.5:
                raise ValueError('RPDI_MAX_RESIDUAL must be in (0, 0.5]')
            rpdi_hidden = max(1, out_channel // self.rpdi_gate_reduction)
            self.rpdi_projection = nn.Linear(
                channels[2], out_channel, bias=False
            )
            self.rpdi_norm = nn.LayerNorm(out_channel, eps=1e-4)
            self.rpdi_activation = nn.ReLU()
            self.rpdi_gate = nn.Sequential(
                nn.Linear(out_channel + 1, rpdi_hidden),
                nn.ReLU(),
                nn.Linear(rpdi_hidden, 1),
                nn.Sigmoid(),
            )
            self.rpdi_scale = nn.Parameter(torch.zeros(1))
            nn.init.zeros_(self.rpdi_gate[-2].weight)
            nn.init.zeros_(self.rpdi_gate[-2].bias)
        #end

        #改进8：RAVC射线可见性上下文默认关闭；仅聚合已有稀疏坐标并使用零初始化有界残差。
        self.use_ray_aligned_visibility_context = self.model_cfg.get(
            'USE_RAY_ALIGNED_VISIBILITY_CONTEXT', False
        )
        if self.use_ray_aligned_visibility_context:
            self.ravc_angular_bins = int(self.model_cfg.get(
                'RAVC_ANGULAR_BINS', 48
            ))
            self.ravc_radial_bins = int(self.model_cfg.get(
                'RAVC_RADIAL_BINS', 16
            ))
            self.ravc_hidden_channels = int(self.model_cfg.get(
                'RAVC_HIDDEN_CHANNELS', 32
            ))
            self.ravc_count_cap = float(self.model_cfg.get(
                'RAVC_COUNT_CAP', 32.0
            ))
            self.ravc_max_residual = float(self.model_cfg.get(
                'RAVC_MAX_RESIDUAL', 0.1
            ))
            if min(self.ravc_angular_bins, self.ravc_radial_bins,
                   self.ravc_hidden_channels) <= 0:
                raise ValueError('RAVC bins and hidden channels must be positive')
            if self.ravc_count_cap <= 1.0:
                raise ValueError('RAVC_COUNT_CAP must exceed 1')
            if not 0.0 < self.ravc_max_residual <= 0.5:
                raise ValueError('RAVC_MAX_RESIDUAL must be in (0, 0.5]')
            self.ravc_norm = nn.LayerNorm(channels[3], eps=1e-4)
            self.ravc_context_projection = nn.Sequential(
                nn.Linear(channels[3] * 2 + 1,
                          self.ravc_hidden_channels),
                nn.ReLU(),
                nn.Linear(self.ravc_hidden_channels, channels[3]),
            )
            ravc_gate_hidden = max(4, self.ravc_hidden_channels // 4)
            self.ravc_gate = nn.Sequential(
                nn.Linear(2, ravc_gate_hidden),
                nn.ReLU(),
                nn.Linear(ravc_gate_hidden, 1),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.ravc_context_projection[-1].weight)
            nn.init.zeros_(self.ravc_context_projection[-1].bias)
            nn.init.zeros_(self.ravc_gate[-2].weight)
            nn.init.zeros_(self.ravc_gate[-2].bias)
        #end

        #改进3：ELGM默认关闭；只读取新增模块自身配置，不改变原slot宽度、层数、方向数和head数。
        self.use_efficient_local_global_mixer = self.model_cfg.get(
            'USE_EFFICIENT_LOCAL_GLOBAL_MIXER', False
        )
        if self.use_efficient_local_global_mixer:
            self.elgm_channel_ratio = float(self.model_cfg.get(
                'ELGM_CHANNEL_RATIO', 0.5
            ))
            self.elgm_replace_layer = int(self.model_cfg.get(
                'ELGM_REPLACE_LAYER', 1
            ))
            self.elgm_max_residual = float(self.model_cfg.get(
                'ELGM_MAX_RESIDUAL', 0.1
            ))
            if self.elgm_max_residual <= 0:
                raise ValueError('ELGM_MAX_RESIDUAL must be positive')
        #end

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, channels[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(channels[0]),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(channels[0], channels[1], spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[0]//2), indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(channels[1], channels[2], spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[1]//2), indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 6]
            block(channels[2], channels[3], spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[2]//2), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
        )

        self.conv5 = spconv.SparseSequential(
            # [200, 176, 6] <- [100, 88, 3]
            block(channels[3], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv5', conv_type='spconv'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5'),
        )
        
        self.conv6 = spconv.SparseSequential(
            # [200, 176, 6] <- [100, 88, 3]
            block(channels[4], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv6', conv_type='spconv'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6'),
        )


        #改进3：False分支沿用原SlotFormer构造；True分支仅替换指定层为ELGM。
        if self.use_efficient_local_global_mixer:
            slotformer_block = SlotFormer(
                channels[3], 4, num_layers=2, win_size=12, shift=False, cwi=True,
                use_efficient_local_global_mixer=True,
                elgm_channel_ratio=self.elgm_channel_ratio,
                elgm_replace_layer=self.elgm_replace_layer,
                elgm_max_residual=self.elgm_max_residual
            )
        else:
            slotformer_block = SlotFormer(
                channels[3], 4, num_layers=2, win_size=12, shift=False, cwi=True
            )
        self.slotformer = spconv.SparseSequential(
            slotformer_block,
            nn.BatchNorm1d(channels[3], eps=1e-3, momentum=0.01),
        )
        #end

        self.up_block = spconv.SparseSequential(
            post_act_block_sparse_2d(channels[3], channels[3], kernel_size=3, stride=1, \
                              padding=1, conv_type='spconv', indice_key='up_block'),
        )

        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv2d(channels[3], out_channel, 3, stride=1, padding=1, bias=False, indice_key='spconv_out'),
            norm_fn(out_channel),
            nn.ReLU(),
        )

        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(True),
        )

        self.forward_ret_dict = {}
        self.num_point_features = out_channel
        self.backbone_channels = {
            'x_conv1': channels[0],
            'x_conv2': channels[1],
            'x_conv3': channels[2],
            'x_conv4': channels[3]
        }

    def bev_out(self, x_conv):
        features_cat = x_conv.features
        #改进14：仅在公共工程开关开启时先转FP32再求和，避免半精度index_add溢出。
        if self.stable_bev_fp32:
            features_cat = features_cat.float()
        #end
        indices_cat = x_conv.indices[:, [0, 2, 3]]
        spatial_shape = x_conv.spatial_shape[1:]

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)

        x_out = spconv.SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=spatial_shape,
            batch_size=x_conv.batch_size
        )
        return x_out

    #改进6：共享的BEV径向距离与高度占用统计完全在GPU上计算，不建立稠密特征图。
    @staticmethod
    def normalized_bev_range(indices, spatial_shape):
        with torch.cuda.amp.autocast(enabled=False):
            coordinates = indices[:, 1:3].float()
            height = coordinates.new_tensor(float(spatial_shape[0])).clamp_min(1.0)
            width = coordinates.new_tensor(float(spatial_shape[1])).clamp_min(1.0)
            lateral = (
                coordinates[:, 0:1] + 0.5 - height * 0.5
            ) / (height * 0.5).clamp_min(1.0)
            forward = (coordinates[:, 1:2] + 0.5) / width
            return torch.sqrt(
                forward.square() + lateral.square()
            ).mul(0.7071067811865476).clamp(0.0, 1.0)

    @staticmethod
    def bev_height_occupancy(x_conv):
        bev_indices = x_conv.indices[:, [0, 2, 3]]
        unique_indices, counts = torch.unique(
            bev_indices, dim=0, return_counts=True
        )
        return unique_indices, counts[:, None]

    def apply_range_density_sparse_calibration(self, x, height_counts):
        output_dtype = x.features.dtype
        with torch.cuda.amp.autocast(enabled=False):
            radial_range = self.normalized_bev_range(
                x.indices, x.spatial_shape
            )
            density = torch.log1p(height_counts.float()) / torch.log1p(
                height_counts.new_tensor(self.rdsc_density_cap).float()
            ).clamp_min(1e-6)
            calibration_signal = torch.cat([
                radial_range, density.clamp(0.0, 1.0)
            ], dim=1)
            channel_delta = self.rdsc_max_residual * torch.tanh(
                self.range_density_calibrator(calibration_signal)
            )
            calibrated = x.features.float() * (1.0 + channel_delta)
        return replace_feature(x, calibrated.to(output_dtype))
    #end

    #改进7：RPDI使用batch-aware整数键精确匹配源/目标稀疏坐标，只返回目标坐标上的特征。
    @staticmethod
    def match_sparse_features_by_coordinates(
            source_indices, source_features, target_indices, spatial_shape):
        matched = source_features.new_zeros(
            (target_indices.shape[0], source_features.shape[1])
        )
        valid = torch.zeros(
            target_indices.shape[0], dtype=torch.bool,
            device=target_indices.device
        )
        if source_indices.shape[0] == 0 or target_indices.shape[0] == 0:
            return matched, valid

        height = int(spatial_shape[0])
        width = int(spatial_shape[1])
        source_long = source_indices.long()
        target_long = target_indices.long()
        source_keys = (
            (source_long[:, 0] * height + source_long[:, 1]) * width
            + source_long[:, 2]
        )
        target_keys = (
            (target_long[:, 0] * height + target_long[:, 1]) * width
            + target_long[:, 2]
        )
        sorted_keys, source_order = torch.sort(source_keys)
        positions = torch.searchsorted(sorted_keys, target_keys)
        safe_positions = positions.clamp(max=sorted_keys.shape[0] - 1)
        valid = (
            (positions < sorted_keys.shape[0])
            & (sorted_keys[safe_positions] == target_keys)
        )
        target_rows = torch.nonzero(valid, as_tuple=False).squeeze(1)
        source_rows = source_order[safe_positions[valid]]
        matched = matched.index_copy(
            0, target_rows, source_features[source_rows]
        )
        return matched, valid

    def apply_range_preserving_detail_injection(self, x, x_conv3):
        output_dtype = x.features.dtype
        #改进13：RPDI在高度聚合前转换为FP32，避免AMP半精度index_add溢出后污染残差分支。
        with torch.cuda.amp.autocast(enabled=False):
            detail_bev = self.bev_out(replace_feature(
                x_conv3, x_conv3.features.float()
            ))
        #end
        with torch.cuda.amp.autocast(enabled=False):
            detail_features = self.rpdi_activation(self.rpdi_norm(
                self.rpdi_projection(detail_bev.features.float())
            ))
            matched_detail, valid = self.match_sparse_features_by_coordinates(
                detail_bev.indices, detail_features,
                x.indices, x.spatial_shape
            )
            radial_range = self.normalized_bev_range(
                x.indices, x.spatial_shape
            )
            gate = self.rpdi_gate(torch.cat([
                matched_detail, radial_range
            ], dim=1))
            gate = gate * valid[:, None].to(gate.dtype)
            residual_scale = self.rpdi_max_residual * torch.tanh(
                self.rpdi_scale.float()
            )
            fused = (
                x.features.float()
                + residual_scale * gate * torch.tanh(matched_detail)
            )
        return replace_feature(x, fused.to(output_dtype))
    #end

    #改进8：RAVC按batch、前向半平面角度和距离环分组，聚合射线邻域上下文且不扩散坐标。
    @staticmethod
    def ray_aligned_group_context(
            features, indices, spatial_shape, angular_bins, radial_bins):
        if indices.shape[0] == 0:
            return (
                torch.zeros_like(features),
                features.new_zeros((0, 1)),
                features.new_zeros((0, 1)),
            )
        with torch.cuda.amp.autocast(enabled=False):
            coordinates = indices[:, 1:3].float()
            height = coordinates.new_tensor(float(spatial_shape[0])).clamp_min(1.0)
            width = coordinates.new_tensor(float(spatial_shape[1])).clamp_min(1.0)
            lateral = (
                coordinates[:, 0] + 0.5 - height * 0.5
            ) / (height * 0.5).clamp_min(1.0)
            forward = (coordinates[:, 1] + 0.5) / width
            radial_range = torch.sqrt(
                forward.square() + lateral.square()
            ).mul(0.7071067811865476).clamp(0.0, 1.0)
            angle = torch.atan2(lateral, forward.clamp_min(1e-6))
            angle = angle.clamp(
                -1.5707963267948966, 1.5707963267948966
            )
            angular_index = torch.floor(
                (angle + 1.5707963267948966)
                / 3.141592653589793 * angular_bins
            ).long().clamp(0, angular_bins - 1)
            radial_index = torch.floor(
                radial_range * radial_bins
            ).long().clamp(0, radial_bins - 1)
            group_coordinates = torch.stack([
                indices[:, 0].long(), angular_index, radial_index
            ], dim=1)
            unique_groups, inverse, counts = torch.unique(
                group_coordinates, dim=0, return_inverse=True,
                return_counts=True
            )
            group_sums = features.float().new_zeros(
                (unique_groups.shape[0], features.shape[1])
            )
            group_sums.index_add_(0, inverse, features.float())
            group_means = group_sums / counts[:, None].float().clamp_min(1.0)
            return (
                group_means[inverse], radial_range[:, None],
                counts[inverse, None].float()
            )

    def apply_ray_aligned_visibility_context(self, x):
        output_dtype = x.features.dtype
        with torch.cuda.amp.autocast(enabled=False):
            normalized = self.ravc_norm(x.features.float())
            context, radial_range, group_counts = (
                self.ray_aligned_group_context(
                    normalized, x.indices, x.spatial_shape,
                    self.ravc_angular_bins, self.ravc_radial_bins
                )
            )
            context_delta = self.ravc_context_projection(torch.cat([
                normalized, context, radial_range
            ], dim=1))
            group_density = torch.log1p(group_counts) / torch.log1p(
                group_counts.new_tensor(self.ravc_count_cap)
            ).clamp_min(1e-6)
            gate = self.ravc_gate(torch.cat([
                radial_range, 1.0 - group_density.clamp(0.0, 1.0)
            ], dim=1))
            residual = (
                self.ravc_max_residual * gate * torch.tanh(context_delta)
            )
            enhanced = x.features.float() + residual
        return replace_feature(x, enhanced.to(output_dtype))
    #end

    #改进2：DCBC使用FP32求和并按同一BEV坐标的高度占用数做幂归一化，不增加坐标或参数。
    @staticmethod
    def bev_out_density_calibrated(x_conv, count_power):
        with torch.cuda.amp.autocast(enabled=False):
            features_cat = x_conv.features.float()
            indices_cat = x_conv.indices[:, [0, 2, 3]]
            spatial_shape = x_conv.spatial_shape[1:]

            indices_unique, inverse = torch.unique(
                indices_cat, dim=0, return_inverse=True
            )
            features_unique = features_cat.new_zeros(
                (indices_unique.shape[0], features_cat.shape[1])
            )
            features_unique.index_add_(0, inverse, features_cat)
            occupancy = features_cat.new_zeros((indices_unique.shape[0], 1))
            occupancy.index_add_(
                0, inverse, features_cat.new_ones((features_cat.shape[0], 1))
            )
            features_unique = features_unique / occupancy.clamp_min(1.0).pow(
                count_power
            )
            features_unique = features_unique.to(x_conv.features.dtype)

            return spconv.SparseConvTensor(
                features=features_unique,
                indices=indices_unique,
                spatial_shape=spatial_shape,
                batch_size=x_conv.batch_size
            )
    #end

    def upsampling(self, x):
        
        x.indices[:, 1:] *= 2
        x.spatial_shape = [s * 2 for s in x.spatial_shape]
        
        # if is_difussion and x.indices.shape[0] > 0:
        x = self.up_block(x)

        return x
    
    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)
        x_conv6 = self.conv6(x_conv5)

        x_conv5.indices[:, 1:] *= 2
        x_conv6.indices[:, 1:] *= 4
        x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
        x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices, x_conv6.indices])

        #改进2：True进入密度校准压缩，False逐行复用原始bev_out求和逻辑。
        if self.use_density_calibrated_bev_compression:
            x = self.bev_out_density_calibrated(
                x_conv4, self.dcbc_count_power
            )
        else:
            x = self.bev_out(x_conv4)
        #end

        #改进6：RDSC只校准已有BEV体素；False时不计算占用统计且逐行保持原前向。
        if self.use_range_density_sparse_calibration:
            count_indices, height_counts = self.bev_height_occupancy(x_conv4)
            matched_counts, _ = self.match_sparse_features_by_coordinates(
                count_indices, height_counts.float(),
                x.indices, x.spatial_shape
            )
            x = self.apply_range_density_sparse_calibration(
                x, matched_counts.clamp_min(1.0)
            )
        #end
        x = self.slotformer(x)

        #改进8：RAVC位于语义混合后、上采样前，增强部分可见目标但不改变候选稀疏坐标。
        if self.use_ray_aligned_visibility_context:
            x = self.apply_ray_aligned_visibility_context(x)
        #end

        x = self.upsampling(x)
        x = self.conv_out(x)

        #改进7：RPDI在检测共享卷积前注入x_conv3细节，输出active voxel数量和坐标严格不变。
        if self.use_range_preserving_detail_injection:
            x = self.apply_range_preserving_detail_injection(x, x_conv3)
        #end
        out = self.shared_conv(x)

        batch_dict.update({'spatial_features_2d': out})
        
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        
        return batch_dict
