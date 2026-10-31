#改进0：工程修复——独立KITTI AP进程，不加载训练模型或创建PyTorch CUDA上下文。
import argparse
import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a saved KITTI result.pkl in isolation')
    parser.add_argument('--result_pkl', required=True, type=Path)
    parser.add_argument('--gt_info', required=True, type=Path)
    parser.add_argument('--output_json', required=True, type=Path)
    parser.add_argument('--output_txt', required=True, type=Path)
    parser.add_argument('--class_names', nargs='+', required=True)
    return parser.parse_args()


def add_mean_metrics(metrics, class_names):
    for metric_name in ('3d', 'bev', 'image'):
        for difficulty in ('easy', 'moderate', 'hard'):
            class_keys = [
                '%s_%s/%s_R40' % (class_name, metric_name, difficulty)
                for class_name in class_names
            ]
            if all(key in metrics for key in class_keys):
                metrics['mean_%s/%s_R40' % (metric_name, difficulty)] = (
                    sum(float(metrics[key]) for key in class_keys) / len(class_keys)
                )
    return metrics


def build_cpu_rotate_iou(boxes_iou_bev_cpu):
    def rotate_iou_cpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
        del device_id
        output_dtype = boxes.dtype
        boxes = np.asarray(boxes, dtype=np.float32)
        query_boxes = np.asarray(query_boxes, dtype=np.float32)
        if boxes.shape[0] == 0 or query_boxes.shape[0] == 0:
            return np.zeros((boxes.shape[0], query_boxes.shape[0]), dtype=output_dtype)

        boxes_7d = np.zeros((boxes.shape[0], 7), dtype=np.float32)
        query_boxes_7d = np.zeros((query_boxes.shape[0], 7), dtype=np.float32)
        boxes_7d[:, 0:2] = boxes[:, 0:2]
        query_boxes_7d[:, 0:2] = query_boxes[:, 0:2]
        boxes_7d[:, 3:5] = boxes[:, 2:4]
        query_boxes_7d[:, 3:5] = query_boxes[:, 2:4]
        boxes_7d[:, 5] = 1.0
        query_boxes_7d[:, 5] = 1.0
        boxes_7d[:, 6] = -boxes[:, 4]
        query_boxes_7d[:, 6] = -query_boxes[:, 4]

        bev_iou = np.asarray(
            boxes_iou_bev_cpu(boxes_7d, query_boxes_7d), dtype=np.float32
        )
        area_boxes = (boxes[:, 2] * boxes[:, 3])[:, None]
        area_query = (query_boxes[:, 2] * query_boxes[:, 3])[None, :]
        intersection = bev_iou * (area_boxes + area_query) / np.maximum(1.0 + bev_iou, 1e-8)
        intersection = np.maximum(intersection, 0.0)

        if criterion == -1:
            output = bev_iou
        elif criterion == 0:
            output = np.divide(
                intersection, area_boxes,
                out=np.zeros_like(intersection), where=area_boxes > 0
            )
        elif criterion == 1:
            output = np.divide(
                intersection, area_query,
                out=np.zeros_like(intersection), where=area_query > 0
            )
        else:
            output = intersection
        return output.astype(output_dtype, copy=False)

    return rotate_iou_cpu_eval


def atomic_write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    temporary_path.write_text(content, encoding='utf-8')
    temporary_path.replace(path)


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu

    kitti_eval_parent = project_root / 'pcdet' / 'datasets' / 'kitti'
    sys.path.insert(0, str(kitti_eval_parent))
    rotate_iou_stub = types.ModuleType('kitti_object_eval_python.rotate_iou')
    rotate_iou_stub.rotate_iou_gpu_eval = build_cpu_rotate_iou(boxes_bev_iou_cpu)
    sys.modules['kitti_object_eval_python.rotate_iou'] = rotate_iou_stub
    from kitti_object_eval_python.eval import get_official_eval_result

    with open(args.gt_info, 'rb') as f:
        gt_infos = pickle.load(f)
    with open(args.result_pkl, 'rb') as f:
        det_annos = pickle.load(f)

    gt_annos = [info['annos'] for info in gt_infos]
    if len(gt_annos) != len(det_annos):
        raise ValueError(
            'KITTI ground-truth/prediction count mismatch: %d != %d'
            % (len(gt_annos), len(det_annos))
        )

    result_str, result_dict = get_official_eval_result(
        gt_annos, det_annos, args.class_names
    )
    result_dict = add_mean_metrics(
        {key: float(value) for key, value in result_dict.items()},
        args.class_names
    )

    atomic_write_text(args.output_txt, result_str)
    atomic_write_text(
        args.output_json,
        json.dumps(result_dict, indent=2, sort_keys=True, ensure_ascii=False) + '\n'
    )
    # KITTI完整指标由父进程读取official_kitti_result.txt并统一记录，避免同一结果打印两次。
    print('KITTI metrics JSON: %s' % args.output_json)


if __name__ == '__main__':
    main()
#end
