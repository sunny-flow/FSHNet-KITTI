import copy
import numpy as np

import torch
import torch.nn as nn
import spconv.pytorch as spconv
from torch.nn.init import kaiming_normal_
from ..model_utils import centernet_utils
from ..model_utils import model_nms_utils
from ...utils import loss_utils
from ...utils.spconv_utils import spconv
from ...ops.iou3d_nms import iou3d_nms_utils

#改进9：DSEC只依赖已有PyTorch；独立分类证据分支与梯度隔离函数。
from ..model_utils.sparse_evidence_utils import (
    DecoupledSparseEvidenceCalibration, preserve_primary_classification_gradient,
)
#end
#改进10：DGR独立几何分支和显式主网络梯度路由。
from ..model_utils.sparse_geometry_utils import DecoupledGeometryRefinement, route_primary_gradient
#end
#改进11：GCQC独立IoU质量分支。
from ..model_utils.sparse_geometry_utils import GeometryConsistencyQualityCalibration
#end

#改进12：按最终分数对合并后的多head预测施加nuScenes每帧总框数上限。
def keep_top_predictions(prediction, max_boxes):
    max_boxes = int(max_boxes)
    if max_boxes < 1:
        raise ValueError('MAX_OBJ_PER_SAMPLE must be positive')
    scores = prediction['pred_scores']
    if scores.shape[0] <= max_boxes:
        return prediction
    selected = scores.topk(max_boxes, sorted=True).indices
    return {key: value.index_select(0, selected) for key, value in prediction.items()}
#end

class SeparateHead(nn.Module):

    def __init__(self, input_channels, sep_head_dict, kernel_size, init_bias=-2.19, use_bias=False):
        super().__init__()

        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(spconv.SparseSequential(
                    spconv.SubMConv2d(input_channels, input_channels, kernel_size, padding=int(kernel_size//2), bias=use_bias, indice_key=cur_name),
                    nn.BatchNorm1d(input_channels),
                    nn.ReLU()
                ))
            fc_list.append(spconv.SubMConv2d(input_channels, output_channels, 1, bias=True, indice_key=cur_name+'out'))
            fc = nn.Sequential(*fc_list)
            if 'heatmap' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, spconv.SubMConv2d):
                        nn.init.normal_(m.weight, mean=0, std=0.001)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict


class SparseDynamicHead(nn.Module):

    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=False):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = torch.Tensor(point_cloud_range).cuda()
        self.voxel_size = torch.Tensor(voxel_size).cuda()
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)

        self.bn_momentum = model_cfg.get('BN_MOM', 0.1)
        self.bn_eps = model_cfg.get('BN_EPS', 1e-5)

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []

        self.up_strides = self.model_cfg.get('UP_STRIDES', [1, 1, 1])
        self.dynamic_pos_num = self.model_cfg.get('DYNAMIC_POS_NUM', [5])
        self.candidate_num = self.model_cfg.get('CANDIDATE_NUM', [5])
        self.r_factor = self.model_cfg.get('R_FACTOR', 0.5)

        #改进0：DSLA掩码工程修复开关默认关闭，关闭时保持原索引行为。
        self.fix_dsla_mask = self.model_cfg.get('FIX_DSLA_MASK', False)
        #end

        #改进4：SN-DSLA仅在开启时读取形状归一化候选距离参数。
        self.use_shape_normalized_dsla = self.model_cfg.get(
            'USE_SHAPE_NORMALIZED_DSLA', False
        )
        if self.use_shape_normalized_dsla:
            self.sn_dsla_min_half_extent = float(self.model_cfg.get(
                'SN_DSLA_MIN_HALF_EXTENT', 1.0
            ))
            self.sn_dsla_max_distance = float(self.model_cfg.get(
                'SN_DSLA_MAX_DISTANCE', 1.0
            ))
            if self.sn_dsla_min_half_extent <= 0:
                raise ValueError('SN_DSLA_MIN_HALF_EXTENT must be positive')
            if self.sn_dsla_max_distance <= 0:
                raise ValueError('SN_DSLA_MAX_DISTANCE must be positive')
        #end

        #改进5：BOPR默认关闭；开启时只增加有界方向投影，不构建额外可学习参数。
        self.use_bounded_orientation_projection = self.model_cfg.get(
            'USE_BOUNDED_ORIENTATION_PROJECTION', False
        )
        if self.use_bounded_orientation_projection:
            self.orientation_min_norm = float(self.model_cfg.get(
                'ORIENTATION_MIN_NORM', 0.25
            ))
            if not 0.0 < self.orientation_min_norm <= 1.0:
                raise ValueError('ORIENTATION_MIN_NORM must be in (0, 1]')
        #end

        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        kernel_size_head = self.model_cfg.get('KERNEL_SIZE_HEAD', 3)

        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = model_cfg.SEPARATE_HEAD_CFG
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            cur_head_dict['heatmap'] = dict(out_channels=len(cur_class_names), num_conv=model_cfg.NUM_HM_CONV)
            self.heads_list.append(
                SeparateHead(
                    input_channels=input_channels,
                    sep_head_dict=cur_head_dict,
                    kernel_size=kernel_size_head,
                    init_bias=-np.log((1 - 0.01) / 0.01),
                    use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False),
                )
            )
        
        self.get_dynamic_masks = loss_utils.DynamicPositiveMask(1, self.model_cfg.get('DCLA_REG_WEIGHT', 3), 
                                                                self.voxel_size[:2] * self.feature_map_stride)
        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}
        self.init_weights()
        self.build_losses()

        #改进9：False不构建参数；恢复CPU随机状态以保持原主网络初始化序列。
        self.use_decoupled_sparse_evidence = self.model_cfg.get(
            'USE_DECOUPLED_SPARSE_EVIDENCE', False
        )
        if self.use_decoupled_sparse_evidence:
            with torch.random.fork_rng(devices=[]):
                self.evidence_calibrators = nn.ModuleList([
                    DecoupledSparseEvidenceCalibration(
                        input_channels, len(names), voxel_size, point_cloud_range,
                        self.feature_map_stride,
                        hidden_channels=int(model_cfg.get('DSEC_HIDDEN_CHANNELS', 16)),
                        radii=model_cfg.get('DSEC_RADII', [1, 3]),
                        max_logit_delta=float(model_cfg.get('DSEC_MAX_LOGIT_DELTA', 0.5)),
                        chunk_size=int(model_cfg.get('DSEC_CHUNK_SIZE', 2048)),
                    ) for names in self.class_names_each_head
                ])
        #end

        #改进10：DGR关闭不构建参数；不消耗主网络初始化随机序列，仅支持本项目8通道bbox布局。
        self.use_decoupled_geometry_refinement = model_cfg.get('USE_DECOUPLED_GEOMETRY_REFINEMENT', False)
        if self.use_decoupled_geometry_refinement:
            self.validate_geometry_head()
            with torch.random.fork_rng(devices=[]):
                self.geometry_refiners = nn.ModuleList([
                    DecoupledGeometryRefinement(
                        input_channels, voxel_size, point_cloud_range, self.feature_map_stride,
                        hidden_channels=int(model_cfg.get('DGR_HIDDEN_CHANNELS', 32)),
                        max_center_delta=float(model_cfg.get('DGR_MAX_CENTER_DELTA', 0.1)),
                        max_log_size_delta=float(model_cfg.get('DGR_MAX_LOG_SIZE_DELTA', 0.1)),
                        radii=model_cfg.get('DGR_RADII', [1, 3]),
                        chunk_size=int(model_cfg.get('DGR_CHUNK_SIZE', 2048)),
                    ) for _ in self.class_names_each_head
                ])
        #end
        #改进11：GCQC可单独开启；False不增加参数，零初始化保持原IoU输出。
        self.use_geometry_consistency_quality = model_cfg.get('USE_GEOMETRY_CONSISTENCY_QUALITY', False)
        if self.use_geometry_consistency_quality:
            self.validate_geometry_head()
            with torch.random.fork_rng(devices=[]):
                self.quality_calibrators = nn.ModuleList([
                    GeometryConsistencyQualityCalibration(
                        input_channels, voxel_size, point_cloud_range, self.feature_map_stride,
                        hidden_channels=int(model_cfg.get('GCQC_HIDDEN_CHANNELS', 16)),
                        max_quality_delta=float(model_cfg.get('GCQC_MAX_QUALITY_DELTA', 0.2)),
                        radii=model_cfg.get('GCQC_RADII', [1, 3]),
                        chunk_size=int(model_cfg.get('GCQC_CHUNK_SIZE', 2048)),
                    ) for _ in self.class_names_each_head
                ])
        #end

    #改进10：校验布局，保留原bbox与IoU用于原DSLA匹配及原主干监督。
    def validate_geometry_head(self):
        heads = self.separate_head_cfg.HEAD_DICT
        #改进12：GCQC兼容nuScenes的10通道框；DGR仍限定原8通道，避免无意改变其训练路径。
        bbox_channels = heads.get('bbox', {}).get('out_channels')
        if (list(self.separate_head_cfg.HEAD_ORDER) != ['bbox']
                or bbox_channels not in (8, 10)
                or heads.get('iou', {}).get('out_channels') != 1):
            raise ValueError('DGR/GCQC require HEAD_ORDER=[bbox], bbox=8 or 10 and iou=1')
        if self.use_decoupled_geometry_refinement and bbox_channels != 8:
            raise ValueError('DGR currently requires an 8-channel bbox; disable DGR for nuScenes velocity boxes')
        #end

    def apply_geometry_adapters(self, x, pred_dicts):
        self.forward_ret_dict['geometry_base_predictions'] = []
        for idx, pred_dict in enumerate(pred_dicts):
            base = {'bbox': pred_dict['bbox'], 'iou': pred_dict['iou']}
            self.forward_ret_dict['geometry_base_predictions'].append(base)
            if self.use_decoupled_geometry_refinement:
                pred_dict['bbox'] = self.geometry_refiners[idx](
                    x.features, x.indices, x.spatial_shape, base['bbox'],
                    pred_dict['heatmap'], base['iou'])
            #改进11：GCQC读取最终几何的detached证据，但不向DGR或RPDI反传梯度。
            if self.use_geometry_consistency_quality:
                #改进12：质量证据仅使用前8个几何通道，速度仍由独立回归监督。
                pred_dict['iou'] = self.quality_calibrators[idx](
                    x.features, x.indices, x.spatial_shape, pred_dict['bbox'][:, :8],
                    pred_dict['heatmap'], base['iou'])
                #end
            #end

    def refined_training_boxes(self, raw_boxes, batch_indices, inds, batch_size):
        orientation = raw_boxes[:, -2:].float().sigmoid() * 2 - 1
        if self.use_bounded_orientation_projection:
            orientation = self.bounded_orientation_projection(orientation, self.orientation_min_norm)
        encoded = torch.cat([raw_boxes[:, :-2].float(), orientation], dim=1)
        return torch.stack([encoded[batch_indices == i][inds[i]] for i in range(batch_size)])
    #end

    #改进9：只替换分类logit，保留基础分类图供DSLA和主网络梯度使用。
    def apply_evidence_calibration(self, x, pred_dicts):
        base_heatmaps = []
        for pred_dict, calibrator in zip(pred_dicts, self.evidence_calibrators):
            base_heatmaps.append(pred_dict['heatmap'])
            pred_dict['heatmap'] = calibrator(
                x.features, x.indices, x.spatial_shape, pred_dict['heatmap']
            )
        self.forward_ret_dict['dsec_base_heatmaps'] = base_heatmaps
    #end

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.momentum = self.bn_momentum
                m.eps = self.bn_eps

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossSparse())
        self.add_module('reg_loss_func', loss_utils.RWIoULoss(self.voxel_size[:2] * self.feature_map_stride))
        self.add_module('crit_iou', loss_utils.SlotFormerIoULoss())

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    #改进5：BOPR在FP32中对cos/sin向量做有界投影，零范数与极小范数输入均保持有限。
    @staticmethod
    def bounded_orientation_projection(orientation, min_norm=0.25):
        with torch.cuda.amp.autocast(enabled=False):
            orientation_fp32 = orientation.float()
            orientation_norm = torch.sqrt(
                orientation_fp32.square().sum(dim=-1, keepdim=True) + 1e-12
            )
            projected = orientation_fp32 / orientation_norm.clamp_min(
                min_norm
            )
        return projected.to(orientation.dtype)
    #end

    def distance(self, voxel_indices, center):
        if len(center.shape) == 1:
            center = center.unsqueeze(0)
        distances = ((voxel_indices - center)**2).sum(-1)
        return distances

    def manhattan_distance(self, voxel_indices, centers):
        """
        voxel_indices: [N, 2]
        centers: [M, 2]
        """
        voxel_indices = voxel_indices[None, :, :]
        centers = centers[:, None, :]
        distances = (torch.abs(voxel_indices - centers)).sum(-1)
        return distances

    def assign_target_of_single_head(self, num_classes, gt_boxes, feature_map_size, 
                                     feature_map_stride, spatial_indices, num_max_objs=500,):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, len(spatial_indices))

        ret_boxes = gt_boxes.new_zeros((num_max_objs, self.dynamic_pos_num, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs, self.dynamic_pos_num).long()
        mask = gt_boxes.new_zeros(num_max_objs, self.dynamic_pos_num).long()
        center_distances = gt_boxes.new_zeros(num_max_objs, self.dynamic_pos_num)
        #voxel_occ_mask = gt_boxes.new_zeros(num_voxels).long()

        box_masks = ((gt_boxes[:, 3] > 0) & (gt_boxes[:, 4] > 0) & (gt_boxes[:, 5] > 0) & (gt_boxes[:, 0] >= self.point_cloud_range[0]) \
                     & (gt_boxes[:, 1] >= self.point_cloud_range[1]) & (gt_boxes[:, 0] < self.point_cloud_range[3]) \
                     & (gt_boxes[:, 1] < self.point_cloud_range[4]))
        
        box_num = box_masks.sum()
        gt_boxes = gt_boxes[box_masks]

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride

        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #

        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        
        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride
        
        box_radius = torch.sqrt((dx / 2)**2 + (dy / 2)**2)
        #radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        #radius = torch.clamp_min(radius.int(), min=min_radius)

        #改进4：SN-DSLA以目标长宽归一化候选偏移，并仅写入真实存在的候选。
        if self.use_shape_normalized_dsla:
            candidate_count = min(
                int(self.dynamic_pos_num), int(spatial_indices.shape[0])
            )
            if candidate_count > 0:
                candidate_offsets = (
                    spatial_indices.float()[None, :, :] + 0.5
                    - center[:, None, :]
                )
                half_extent = torch.stack([dx, dy], dim=-1) * 0.5
                half_extent = half_extent.clamp_min(
                    self.sn_dsla_min_half_extent
                )
                distances = (
                    candidate_offsets / half_extent[:, None, :]
                ).square().sum(dim=-1)
                sort_distance, sort_inds = torch.sort(distances, dim=-1)
                sort_distance = sort_distance[:, :candidate_count]
                sort_inds = sort_inds[:, :candidate_count]

                inds[:box_num, :candidate_count] = sort_inds
                mask[:box_num, :candidate_count] = (
                    sort_distance <= self.sn_dsla_max_distance
                )
                mask[:box_num, 0] = True
                center_distances[:box_num, :candidate_count] = sort_distance
        #end

        #改进0：非SN-DSLA路径保留原候选计算；True修正逐GT最近候选掩码，False保留旧链式索引行为。
        if not self.use_shape_normalized_dsla:
            distances = self.manhattan_distance(
                spatial_indices.float() + 0.5, center
            )
            sort_res = torch.sort(distances, dim=-1)
            #改进16：候选少于5个（包括零个）时只写实际存在的列，避免空索引和形状不匹配。
            candidate_count = min(int(self.dynamic_pos_num), int(spatial_indices.shape[0]))
            sort_distance = sort_res[0][:, :candidate_count]
            sort_inds = sort_res[1][:, :candidate_count]

            inds[:box_num, :candidate_count] = sort_inds
            mask[:box_num, :candidate_count] = sort_distance <= box_radius[:, None]
            if self.fix_dsla_mask and candidate_count > 0:
                mask[:box_num, 0] = True
            elif not self.fix_dsla_mask and candidate_count > 0 and gt_boxes.shape[0] > 0:
                mask[:box_num][0] = True
            center_distances[:box_num, :candidate_count] = sort_distance
            #end
        #end

        #改进4：SN-DSLA无候选时返回固定形状空目标。
        #改进16：无稀疏候选时两条路径都保持空监督，不索引不存在的体素。
        if spatial_indices.shape[0] > 0:
            ret_boxes[:box_num, :, 0:2] = center[:, None, :] - spatial_indices[inds[:box_num]][..., :2] - 0.5
            ret_boxes[:box_num, :, 2:3] = z[:, None, None]
            ret_boxes[:box_num, :, 3:6] = gt_boxes[:, None, 3:6].log()
            ret_boxes[:box_num, :, 6:7] = torch.cos(gt_boxes[:, None, 6:7])
            #改进12：方向正弦只写第7通道，保留nuScenes第8、9通道给速度。
            ret_boxes[:box_num, :, 7:8] = torch.sin(gt_boxes[:, None, 6:7])
            #end
            if gt_boxes.shape[1] > 8:
                    ret_boxes[:box_num, :, 8:] = gt_boxes[:, None, 7:-1]
        #end

        cur_class_id = (gt_boxes[:, -1] - 1).long()

        #改进4：SN-DSLA只把mask标记的真实候选写入heatmap，绝不激活填充槽。
        if self.use_shape_normalized_dsla:
            valid_rows, valid_columns = torch.nonzero(
                mask[:box_num].bool(), as_tuple=True
            )
            heatmap[
                cur_class_id[valid_rows],
                inds[:box_num][valid_rows, valid_columns]
            ] = 1
        else:
            heatmap[cur_class_id[:, None], sort_inds] = 1
        #end

        # centernet_utils.draw_gaussian_to_sparse_heatmap
        """for k in range(gt_boxes.shape[0]):
            cur_class_id = (gt_boxes[k, -1] - 1).long()
            center_distances = (torch.abs(spatial_indices - center_int[k][None])).sum(-1)
            centernet_utils.draw_gaussian_to_sparse_heatmap(
                heatmap[cur_class_id], center_distances, self.cross_area_r, normalize=True)"""

            #inds[k] = center_distances.argmin()
            #mask[k] = True

        return heatmap, ret_boxes, inds, mask, center_distances
    
    def assign_targets(self, gt_boxes, feature_map_size, spatial_indices, **kwargs):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:
        """
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'target_boxes': [],
            'inds': [],
            'masks': [],
            'distances': [],
            'gt_boxes': []
        }

        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list, gt_boxes_list, distances_list = [], [], [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []

                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                heatmap, ret_boxes, inds, mask, distances = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head,
                    feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    spatial_indices=spatial_indices[spatial_indices[:, 0] == bs_idx][:, [2, 1]],
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                )
                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                target_boxes_list.append(ret_boxes.to(gt_boxes_single_head.device))
                inds_list.append(inds.to(gt_boxes_single_head.device))
                masks_list.append(mask.to(gt_boxes_single_head.device))
                gt_boxes_list.append(gt_boxes_single_head)
                distances_list.append(distances)

            """v_hm = spconv.SparseConvTensor(
                features=torch.cat(heatmap_list, dim=1).transpose(1, 0).contiguous(),
                indices=spatial_indices.int(),
                spatial_shape=feature_map_size,
                batch_size=batch_size
            )
            v_hm = v_hm.dense().permute(2, 3, 0, 1)
            from matplotlib import pyplot as plt
            data = v_hm.view(*feature_map_size, -1).clone()
            data = (torch.sigmoid(data)).contiguous().detach().cpu().numpy()
            plt.imshow(data, interpolation='nearest')
            plt.show()"""
            
            ret_dict['heatmaps'].append(torch.cat(heatmap_list, dim=1).permute(1, 0))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
            ret_dict['gt_boxes'].append(gt_boxes_list)
            ret_dict['distances'].append(torch.stack(distances_list, dim=0))

        return ret_dict

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']
        
        batch_size = self.forward_ret_dict['batch_size']
        
        tb_dict = {}
        loss = 0
        batch_indices = self.forward_ret_dict['spatial_indices'][:, 0]
        spatial_indices = self.forward_ret_dict['spatial_indices'][:, 1:]

        for idx, pred_dict in enumerate(pred_dicts):
            #改进0：修复模式在FP32中执行稀疏分类与回归匹配，降低AMP下cost/loss溢出风险；关闭时保持原dtype路径。
            if self.fix_dsla_mask:
                pred_hm = self.sigmoid(pred_dict['heatmap'].float())
                pred_reg = torch.cat([
                    pred_dict[head_name].float()
                    for head_name in self.separate_head_cfg.HEAD_ORDER
                ], dim=1)
            else:
                pred_hm = self.sigmoid(pred_dict['heatmap'])
                pred_reg = torch.cat([
                    pred_dict[head_name]
                    for head_name in self.separate_head_cfg.HEAD_ORDER
                ], dim=1)
            #end
            #改进10：DGR精修不进入DSLA回归cost；原回归预测仍接受原损失梯度。
            if self.use_decoupled_geometry_refinement:
                raw_reg = self.forward_ret_dict['geometry_base_predictions'][idx]['bbox']
                pred_reg = raw_reg.float().clone() if self.fix_dsla_mask else raw_reg.clone()
            #end
            #改进12：10通道布局的方向位于6:8，末两位是vx/vy；8通道旧路径不变。
            if pred_reg.shape[-1] == 10:
                pred_reg[..., 6:8] = pred_reg[..., 6:8].sigmoid() * 2 - 1
            else:
                pred_reg[..., -2:] = pred_reg[..., -2:].sigmoid() * 2 - 1
            #end

            #改进5：训练匹配和回归损失均使用同一有界方向表示。
            if self.use_bounded_orientation_projection:
                projected_orientation = self.bounded_orientation_projection(
                    pred_reg[..., -2:], self.orientation_min_norm
                )
                pred_reg = torch.cat([
                    pred_reg[..., :-2], projected_orientation
                ], dim=-1)
            #end

            target_boxes = target_dicts['target_boxes'][idx]
            target_hm = target_dicts['heatmaps'][idx]

            masks = target_dicts['masks'][idx]
            inds = target_dicts['inds'][idx]
            distances = target_dicts['distances'][idx]
            
            pred_boxes = []
            pred_cls = []
            target_cls = []
            batch_spatial_indices = []
            #改进9：匹配始终使用原heatmap；新分数不得反馈改变DSLA正样本选择。
            if self.use_decoupled_sparse_evidence:
                base_logits = self.forward_ret_dict['dsec_base_heatmaps'][idx]
                matching_hm = self.sigmoid(base_logits.float()) if self.fix_dsla_mask else self.sigmoid(base_logits)
            #end
            for bs_idx in range(batch_size):
                batch_inds = batch_indices==bs_idx
                pred_boxes.append(pred_reg[batch_inds][inds[bs_idx]])
                #改进9：False沿用原分类cost输入和计算精度。
                if self.use_decoupled_sparse_evidence:
                    pred_cls.append(matching_hm[batch_inds][inds[bs_idx]])
                else:
                    pred_cls.append(pred_hm[batch_inds][inds[bs_idx]])
                #end
                target_cls.append(target_hm[batch_inds][inds[bs_idx]])
                batch_spatial_indices.append(spatial_indices[batch_inds][inds[bs_idx]])

            batch_spatial_indices = torch.stack(batch_spatial_indices)
            target_cls = torch.stack(target_cls)
            pred_boxes = torch.stack(pred_boxes)
            pred_cls = torch.stack(pred_cls)

            #  calculate IoU    
            iou_targets = self.get_iou_targets(pred_boxes.float(), target_boxes, masks, batch_spatial_indices)
 
            # get dynamic positive masks 
            pos_masks = self.get_dynamic_masks(pred_cls, target_cls, pred_boxes, target_boxes, masks, iou_targets)
            #pos_masks = masks

            # calculate cls loss
            for bs_idx in range(batch_size):
                batch_inds = (batch_indices==bs_idx).nonzero().squeeze()
                cur_target_hm = target_hm[batch_inds]
                cur_target_hm[inds[bs_idx]] *= pos_masks[bs_idx][..., None]
                target_hm[batch_inds] = cur_target_hm

            hm_loss = self.hm_loss_func(pred_hm, target_hm)
            #改进9：原Focal监督和权重不变；主网络获得原分类梯度，DSEC获得校准梯度。
            if self.use_decoupled_sparse_evidence:
                base_hm_loss = self.hm_loss_func(matching_hm, target_hm)
                tb_dict['dsec_base_hm_loss_head_%d' % idx] = base_hm_loss.item()
                tb_dict['dsec_calibrated_hm_loss_head_%d' % idx] = hm_loss.item()
                tb_dict['dsec_abs_logit_delta_head_%d' % idx] = (
                    pred_dict['heatmap'].detach() - base_logits.detach().float()
                ).abs().mean().item()
                hm_loss = preserve_primary_classification_gradient(base_hm_loss, hm_loss)
            #end
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

            # calculate reg loss
            reg_loss = self.reg_loss_func(
                pred_boxes.view(-1, pred_boxes.shape[-1]), target_boxes.view(-1, target_boxes.shape[-1]), 
                pos_masks.eq(1).view(-1), self.r_factor
            )
            #改进10：相同RWIoU、相同正样本与loc_weight；主网络梯度取原损失，DGR梯度取精修损失。
            if self.use_decoupled_geometry_refinement:
                refined_boxes = self.refined_training_boxes(pred_dict['bbox'], batch_indices, inds, batch_size)
                refined_loss = self.reg_loss_func(
                    refined_boxes.reshape(-1, 8), target_boxes.reshape(-1, 8),
                    pos_masks.eq(1).reshape(-1), self.r_factor)
                tb_dict['dgr_base_reg_loss_head_%d' % idx] = reg_loss.item()
                tb_dict['dgr_refined_reg_loss_head_%d' % idx] = refined_loss.item()
                reg_loss = route_primary_gradient(reg_loss, refined_loss)
            #end
            #loc_loss = (reg_loss * reg_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
            #改进12：RWIoU只监督几何；10通道时额外以原code_weights末两项监督速度，8通道路径不变。
            if pred_boxes.shape[-1] == 10:
                code_weights = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights']
                if len(code_weights) != 10:
                    raise ValueError('10-channel bbox requires 10 code_weights')
                valid_velocity = (pos_masks.eq(1) & masks.bool()
                                  & torch.isfinite(target_boxes[..., 8:10]).all(dim=-1))
                target_velocity = target_boxes[..., 8:10].float()
                target_velocity = torch.where(torch.isfinite(target_velocity),
                                              target_velocity, torch.zeros_like(target_velocity))
                velocity_difference = (
                    pred_boxes[..., 8:10].float() - target_velocity
                ).abs()
                velocity_difference = torch.where(
                    valid_velocity[..., None], velocity_difference,
                    torch.zeros_like(velocity_difference))
                velocity_weights = velocity_difference.new_tensor(code_weights[8:10])
                velocity_loss = (velocity_difference * velocity_weights).sum() / valid_velocity.sum().clamp_min(1)
                loc_loss = (reg_loss + velocity_loss) * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
                tb_dict['velocity_loss_head_%d' % idx] = velocity_loss.item()
            else:
                loc_loss = reg_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
            #end

            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()

            if 'iou' in pred_dict:
                pred_ious = pred_dict['iou']
                #改进11：原IoU分支仍针对原框监督，保持主网络训练路径；新输出使用独立质量监督。
                if self.use_geometry_consistency_quality:
                    pred_ious = self.forward_ret_dict['geometry_base_predictions'][idx]['iou']
                #end
                #改进0：工程修复开启时让IoU监督保持FP32，避免半精度回归分支溢出。
                if self.fix_dsla_mask:
                    pred_ious = pred_ious.float()
                #end
                batch_pred_ious = []
                for bs_idx in range(batch_size):
                    batch_inds = batch_indices==bs_idx
                    batch_pred_ious.append(pred_ious[batch_inds][inds[bs_idx]])
                batch_pred_ious = torch.stack(batch_pred_ious)
                iou_loss = self.crit_iou(batch_pred_ious.view(-1), iou_targets.view(-1) * 2 - 1, pos_masks.eq(1).view(-1))
                #改进11：质量目标对应最终框且停止梯度；复用原IoU loss及权重，不回传DGR或匹配。
                if self.use_geometry_consistency_quality:
                    calibrated_ious = torch.stack([
                        pred_dict['iou'].float()[batch_indices == i][inds[i]] for i in range(batch_size)
                    ])
                    quality_targets = iou_targets
                    if self.use_decoupled_geometry_refinement:
                        quality_targets = self.get_iou_targets(
                            refined_boxes.detach().float(), target_boxes, masks, batch_spatial_indices)
                    calibrated_loss = self.crit_iou(calibrated_ious.reshape(-1),
                                                   quality_targets.detach().reshape(-1) * 2 - 1,
                                                   pos_masks.eq(1).reshape(-1))
                    tb_dict['gcqc_base_iou_loss_head_%d' % idx] = iou_loss.item()
                    tb_dict['gcqc_calibrated_iou_loss_head_%d' % idx] = calibrated_loss.item()
                    iou_loss = route_primary_gradient(iou_loss, calibrated_loss)
                #end
                loss += (hm_loss + loc_loss + iou_loss)
                #loss += (hm_loss + loc_loss)
                tb_dict['iou_loss_head_%d' % idx] = iou_loss.item()
            else:
                loss += (hm_loss + loc_loss)

        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict
    
    def get_iou_targets(self, box_preds, box_targets, masks, spatial_indices):
        iou_targets = torch.zeros_like(masks).float()
        input_shape = box_preds.shape
        box_preds = box_preds.reshape(-1, input_shape[-1])
        box_targets = box_targets.reshape(-1, input_shape[-1])
        spatial_indices = spatial_indices.reshape(-1, spatial_indices.shape[-1])
        box_inds = masks.view(-1).nonzero().squeeze(-1)

        qboxes = self._get_predicted_boxes(box_preds[box_inds], spatial_indices[box_inds])
        gboxes = self._get_predicted_boxes(box_targets[box_inds], spatial_indices[[box_inds]])

        iou_pos_targets = iou3d_nms_utils.boxes_aligned_iou3d_gpu(qboxes, gboxes).detach()

        iou_targets.view(-1)[box_inds] = iou_pos_targets.squeeze(-1)
        iou_targets = torch.clamp(iou_targets, 0, 1)

        return iou_targets
    
    def _get_predicted_boxes(self, pred_boxes, spatial_indices):
        center, center_z, dim, rot_cos, rot_sin = pred_boxes[..., :2], pred_boxes[..., 2:3], pred_boxes[..., 3:6], \
                                                  pred_boxes[..., 6:7], pred_boxes[..., 7:8]
        dim = torch.exp(torch.clamp(dim, min=-5, max=5))
        angle = torch.atan2(rot_sin, rot_cos)
        xs = (spatial_indices[:, 1:2] + center[:, 0:1]) * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        ys = (spatial_indices[:, 0:1] + center[:, 1:2]) * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
        box_part_list = [xs, ys, center_z, dim, angle]
        pred_box = torch.cat((box_part_list), dim=-1)
        return pred_box
    
    def generate_predicted_boxes(self, batch_size, pred_dicts, sparse_indices, post_process_cfg):
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

        ret_dict = [{'pred_boxes': [], 'pred_scores': [], 'pred_labels': []} for _ in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['heatmap'].sigmoid()
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            #改进0：修复模式在解码前约束尺寸logit，避免exp产生Inf；关闭时保持原解码。
            if self.fix_dsla_mask:
                batch_dim = pred_dict['dim'].float().clamp(-5, 5).exp()
            else:
                batch_dim = pred_dict['dim'].exp()
            #end
            batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1).sigmoid() * 2 - 1
            batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1).sigmoid() * 2 - 1
            #改进5：推理解码前投影方向向量，保持atan2输入有限且范数不超过1。
            if self.use_bounded_orientation_projection:
                batch_orientation = self.bounded_orientation_projection(
                    torch.cat([batch_rot_cos, batch_rot_sin], dim=1),
                    self.orientation_min_norm
                )
                batch_rot_cos = batch_orientation[:, 0:1]
                batch_rot_sin = batch_orientation[:, 1:2]
            #end
            batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None
            batch_iou = (pred_dict['iou'] + 1) * 0.5 if 'iou' in pred_dict else None
            #改进0：修复模式约束解码IoU分数到有效概率区间。
            if self.fix_dsla_mask and batch_iou is not None:
                batch_iou = batch_iou.float().clamp(1e-4, 1.0)
            #end

            final_pred_dicts = centernet_utils.decode_bbox_from_sparse_heatmap(
                heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel, iou=batch_iou,
                point_cloud_range=self.point_cloud_range,
                voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                sparse_indices=sparse_indices,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range
            )

            for bidx, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]

                if post_process_cfg.get('USE_IOU_TO_RECTIFY_SCORE', False) and 'pred_ious' in final_dict:
                    pred_scores, pred_labels, pred_ious = final_dict['pred_scores'], final_dict['pred_labels'], final_dict['pred_ious']
                    #改进0：修复模式在幂运算前约束底数，防止负IoU或零分数产生NaN。
                    if self.fix_dsla_mask:
                        pred_scores = pred_scores.float().clamp(1e-4, 1.0)
                        pred_ious = pred_ious.float().clamp(1e-4, 1.0)
                    #end
                    IOU_RECTIFIER = pred_scores.new_tensor(post_process_cfg.IOU_RECTIFIER)
                    final_dict['pred_scores'] = torch.pow(pred_scores, 1 - IOU_RECTIFIER[pred_labels]) * torch.pow(pred_ious, IOU_RECTIFIER[pred_labels])

                if post_process_cfg.NMS_CONFIG.NMS_TYPE == 'class_specific_nms':
                    selected, selected_scores = model_nms_utils.class_specific_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        box_labels=final_dict['pred_labels'], nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.NMS_CONFIG.get('SCORE_THRESH', None)
                    )
                else:
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=post_process_cfg.NMS_CONFIG, score_thresh=None,
                    )

                ret_dict[bidx]['pred_boxes'].append(final_dict['pred_boxes'][selected])
                ret_dict[bidx]['pred_scores'].append(selected_scores)
                ret_dict[bidx]['pred_labels'].append(final_dict['pred_labels'][selected])

        for bidx in range(batch_size):
            ret_dict[bidx]['pred_boxes'] = torch.cat(ret_dict[bidx]['pred_boxes'], dim=0)
            ret_dict[bidx]['pred_scores'] = torch.cat(ret_dict[bidx]['pred_scores'], dim=0)
            ret_dict[bidx]['pred_labels'] = torch.cat(ret_dict[bidx]['pred_labels'], dim=0) + 1
            #改进12：nuScenes要求多个head合计至多500框；默认关闭，KITTI原输出不变。
            if post_process_cfg.get('ENFORCE_GLOBAL_MAX_OBJ_PER_SAMPLE', False):
                ret_dict[bidx] = keep_top_predictions(
                    ret_dict[bidx], post_process_cfg.MAX_OBJ_PER_SAMPLE
                )
            #end

        return ret_dict

    def generate_predicted_boxes_v2(self, batch_size, pred_dicts, sparse_indices):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

        ret_dict = [{'pred_boxes': [], 'pred_scores': [], 'pred_labels': []} for _ in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['heatmap'].sigmoid()
            batch_bbox = pred_dict['bbox']
            batch_center = batch_bbox[:, :2]
            batch_center_z = batch_bbox[:, 2:3]
            #改进0：修复模式在统一bbox解码前约束尺寸logit，避免exp溢出。
            if self.fix_dsla_mask:
                batch_dim = batch_bbox[:, 3:6].float().clamp(-5, 5).exp()
            else:
                batch_dim = batch_bbox[:, 3:6].exp()
            #end
            batch_rot_cos = batch_bbox[:, 6:7].sigmoid() * 2 - 1
            batch_rot_sin = batch_bbox[:, 7:8].sigmoid() * 2 - 1
            #改进5：统一bbox解码路径使用与训练一致的有界方向投影。
            if self.use_bounded_orientation_projection:
                batch_orientation = self.bounded_orientation_projection(
                    torch.cat([batch_rot_cos, batch_rot_sin], dim=1),
                    self.orientation_min_norm
                )
                batch_rot_cos = batch_orientation[:, 0:1]
                batch_rot_sin = batch_orientation[:, 1:2]
            #end
            #改进12：统一bbox头的nuScenes速度来自末两通道；KITTI 8通道路径不变。
            batch_vel = (batch_bbox[:, 8:10] if batch_bbox.shape[1] == 10 else
                         pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None)
            #end
            batch_iou = (pred_dict['iou'] + 1) * 0.5 if 'iou' in pred_dict else None
            #改进0：修复模式约束统一bbox路径中的IoU分数。
            if self.fix_dsla_mask and batch_iou is not None:
                batch_iou = batch_iou.float().clamp(1e-4, 1.0)
            #end

            final_pred_dicts = centernet_utils.decode_bbox_from_sparse_heatmap(
                heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel, iou=batch_iou,
                point_cloud_range=self.point_cloud_range,
                voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                sparse_indices=sparse_indices,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range
            )

            for bidx, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]

                if post_process_cfg.get('USE_IOU_TO_RECTIFY_SCORE', False) and 'pred_ious' in final_dict:
                    pred_scores, pred_labels, pred_ious = final_dict['pred_scores'], final_dict['pred_labels'], final_dict['pred_ious']
                    #改进0：修复模式在统一bbox分数校准前约束幂运算底数。
                    if self.fix_dsla_mask:
                        pred_scores = pred_scores.float().clamp(1e-4, 1.0)
                        pred_ious = pred_ious.float().clamp(1e-4, 1.0)
                    #end
                    IOU_RECTIFIER = pred_scores.new_tensor(post_process_cfg.IOU_RECTIFIER)
                    final_dict['pred_scores'] = torch.pow(pred_scores, 1 - IOU_RECTIFIER[pred_labels]) * torch.pow(pred_ious, IOU_RECTIFIER[pred_labels])

                if post_process_cfg.NMS_CONFIG.NMS_TYPE == 'class_specific_nms':
                    selected, selected_scores = model_nms_utils.class_specific_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        box_labels=final_dict['pred_labels'], nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.NMS_CONFIG.get('SCORE_THRESH', None)
                    )
                else:
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=post_process_cfg.NMS_CONFIG, score_thresh=None,
                    )

                ret_dict[bidx]['pred_boxes'].append(final_dict['pred_boxes'][selected])
                ret_dict[bidx]['pred_scores'].append(selected_scores)
                ret_dict[bidx]['pred_labels'].append(final_dict['pred_labels'][selected])

        for bidx in range(batch_size):
            ret_dict[bidx]['pred_boxes'] = torch.cat(ret_dict[bidx]['pred_boxes'], dim=0)
            ret_dict[bidx]['pred_scores'] = torch.cat(ret_dict[bidx]['pred_scores'], dim=0)
            ret_dict[bidx]['pred_labels'] = torch.cat(ret_dict[bidx]['pred_labels'], dim=0) + 1
            #改进12：推理实际使用v2路径，合并六个head后按最终分数限制全局框数。
            if post_process_cfg.get('ENFORCE_GLOBAL_MAX_OBJ_PER_SAMPLE', False):
                ret_dict[bidx] = keep_top_predictions(
                    ret_dict[bidx], post_process_cfg.MAX_OBJ_PER_SAMPLE
                )
            #end

        return ret_dict
    
    def forward(self, data_dict):
        x = data_dict['spatial_features_2d']
        spatial_indices = None

        pred_dicts = []
        for head in self.heads_list:
            pred_dict = head(x)
            for k, v in pred_dict.items():
                spatial_indices = v.indices
                pred_dict[k] = v.features
            pred_dicts.append(pred_dict)
        #改进10：几何分支处理原始头输出；DSEC在其后运行，不改变几何分支读取的原分类输入。
        if self.use_decoupled_geometry_refinement or self.use_geometry_consistency_quality:
            self.apply_geometry_adapters(x, pred_dicts)
        #end
        #改进9：各头输出完成后独立校准分类分数；bbox/IoU张量保持原引用。
        if self.use_decoupled_sparse_evidence:
            self.apply_evidence_calibration(x, pred_dicts)
        #end
        self.forward_ret_dict['pred_dicts'] = pred_dicts
        self.forward_ret_dict['spatial_indices'] = spatial_indices
        self.forward_ret_dict['batch_size'] = data_dict['batch_size']
        self.forward_ret_dict['feature_map_size'] = x.spatial_shape

        if self.training:
            target_dicts = self.assign_targets(data_dict['gt_boxes'], x.spatial_shape[-2:], spatial_indices)
            self.forward_ret_dict['target_dicts'] = target_dicts
        
        else:
            data_dict['final_box_dicts'] = self.generate_predicted_boxes_v2(data_dict['batch_size'], pred_dicts, \
                                                                         spatial_indices)

        return data_dict
