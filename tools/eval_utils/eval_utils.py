import pickle
import time

#改进0：工程修复——KITTI官方AP改由独立进程计算，隔离PyTorch/spconv与Numba CUDA上下文。
import json
import os
import subprocess
import sys
from pathlib import Path
#end

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
# from visual_utils import draw_scenes

#改进0：工程修复——在仅加载Numba评估器的子进程中计算KITTI AP，避免主进程段错误。
def evaluate_kitti_result_in_subprocess(cfg, dataset, class_names, result_dir, logger):
    result_pkl = Path(result_dir) / 'result.pkl'
    metrics_json = Path(result_dir) / 'kitti_metrics.json'
    result_txt = Path(result_dir) / 'official_kitti_result.txt'
    eval_script = Path(cfg.ROOT_DIR) / 'tools' / 'eval_kitti_result.py'
    gt_info_path = Path(dataset.root_path) / cfg.DATA_CONFIG.INFO_PATH['test'][0]

    if not result_pkl.exists():
        raise FileNotFoundError('KITTI prediction file not found: %s' % result_pkl)
    if not gt_info_path.exists():
        raise FileNotFoundError('KITTI validation info file not found: %s' % gt_info_path)

    command = [
        sys.executable, str(eval_script),
        '--result_pkl', str(result_pkl),
        '--gt_info', str(gt_info_path),
        '--output_json', str(metrics_json),
        '--output_txt', str(result_txt),
        '--class_names', *list(class_names),
    ]
    child_env = os.environ.copy()
    numba_cache_dir = Path(result_dir) / '.numba_cache'
    numba_cache_dir.mkdir(parents=True, exist_ok=True)
    child_env['NUMBA_CACHE_DIR'] = str(numba_cache_dir)
    child_env['NUMBA_DISABLE_CUDA'] = '1'
    child_env['PYTHONUNBUFFERED'] = '1'

    logger.info('Run KITTI official AP in isolated process: %s' % ' '.join(command))
    completed = subprocess.run(
        command, env=child_env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, universal_newlines=True
    )
    if completed.stdout:
        logger.info('\n%s' % completed.stdout.rstrip())
    if completed.returncode != 0:
        raise RuntimeError(
            'Isolated KITTI evaluation failed with return code %d. '
            'Predictions remain available at %s' % (completed.returncode, result_pkl)
        )

    with open(metrics_json, 'r', encoding='utf-8') as f:
        result_dict = json.load(f)
    with open(result_txt, 'r', encoding='utf-8') as f:
        result_str = f.read()
    return result_str, result_dict
#end

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)

        disp_dict = {}

        # if cfg.get('VISUALIZE', False):
        #     gt_boxes = batch_dict['gt_boxes'][0, :, :-1]
        #     draw_scenes(points=batch_dict['points'][:, 1:], gt_boxes=gt_boxes, ref_boxes=pred_dicts[0]['pred_boxes'], \
        #                 ref_labels=pred_dicts[0]['pred_labels'], ref_scores=pred_dicts[0]['pred_scores'])
            
        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    #改进0：工程修复——KITTI使用独立子进程；其他数据集完整保留原评估逻辑。
    if cfg.MODEL.POST_PROCESSING.EVAL_METRIC == 'kitti':
        result_str, result_dict = evaluate_kitti_result_in_subprocess(
            cfg, dataset, class_names, result_dir, logger
        )
    else:
        result_str, result_dict = dataset.evaluation(
            det_annos, class_names,
            eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
            output_path=final_output_dir
        )
    #end

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
