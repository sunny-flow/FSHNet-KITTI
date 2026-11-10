import os

import torch
import tqdm
import time
import glob
from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils, commu_utils

#改进13：仅用标准库检查训练标量是否为有限值，不改变优化器、AMP或损失权重。
import math
#end

#改进0：逐epoch评估、指标表格和最佳checkpoint持久化所需的标准库与现有评估模块。
import json
import re
import shutil
from pathlib import Path
from eval_utils import eval_utils
#end


#改进0：将KITTI R40核心指标格式化为固定列宽表格，便于跨epoch和消融实验直接比较。
def format_epoch_performance_table(epoch, train_loss, metrics, class_names):
    #改进12：nuScenes只报告真实mAP/NDS，不套用KITTI的R40类别表。
    if 'mAP' in metrics and 'NDS' in metrics:
        return ('Epoch %d performance | train_loss=%s\n'
                '+----------+----------+\n'
                '| mAP      | NDS      |\n'
                '+----------+----------+\n'
                '| %8.4f | %8.4f |\n'
                '+----------+----------+' % (
                    epoch,
                    '--' if train_loss is None else '%.6f' % float(train_loss),
                    float(metrics['mAP']), float(metrics['NDS'])))
    #end
    metric_columns = [
        ('BBox-M', 'image', 'moderate'),
        ('BEV-M', 'bev', 'moderate'),
        ('3D-E', '3d', 'easy'),
        ('3D-M', '3d', 'moderate'),
        ('3D-H', '3d', 'hard'),
        ('AOS-M', 'aos', 'moderate'),
    ]

    def metric_value(class_name, metric_name, difficulty):
        key = '%s_%s/%s_R40' % (
            class_name, metric_name, difficulty
        )
        value = metrics.get(key, None)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    rows = []
    for class_name in class_names:
        rows.append([
            class_name,
            *[
                metric_value(class_name, metric_name, difficulty)
                for _, metric_name, difficulty in metric_columns
            ]
        ])

    mean_row = ['Mean']
    for _, metric_name, difficulty in metric_columns:
        values = [
            metric_value(class_name, metric_name, difficulty)
            for class_name in class_names
        ]
        values = [value for value in values if value is not None]
        mean_row.append(sum(values) / len(values) if values else None)
    rows.append(mean_row)

    headers = ['Class'] + [column[0] for column in metric_columns]
    formatted_rows = []
    for row in rows:
        formatted_rows.append([
            str(row[0]),
            *[
                '--' if value is None else '%.4f' % value
                for value in row[1:]
            ]
        ])
    widths = [
        max(len(headers[column]), *[
            len(row[column]) for row in formatted_rows
        ])
        for column in range(len(headers))
    ]

    def render(row):
        return '| ' + ' | '.join(
            value.ljust(widths[index])
            for index, value in enumerate(row)
        ) + ' |'

    separator = '+-' + '-+-'.join('-' * width for width in widths) + '-+'
    lines = [
        'Epoch %d performance (AP_R40, %%) | train_loss=%s' % (
            epoch,
            '--' if train_loss is None else '%.6f' % float(train_loss)
        ),
        separator,
        render(headers),
        separator,
    ]
    lines.extend(render(row) for row in formatted_rows)
    lines.append(separator)
    return '\n'.join(lines)


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    with temporary_path.open('w', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write('\n')
    temporary_path.replace(path)


def update_best_performance(
        model, optimizer, epoch, accumulated_iter, metrics,
        metric_name, ckpt_save_dir, logger):
    ckpt_save_dir = Path(ckpt_save_dir)
    summary_path = ckpt_save_dir.parent / 'best_metrics.json'
    previous = {}
    if summary_path.is_file():
        try:
            with summary_path.open('r', encoding='utf-8') as stream:
                previous = json.load(stream)
        except (OSError, ValueError):
            logger.warning(
                'Ignore unreadable best metric summary: %s' % summary_path
            )

    if metric_name not in metrics:
        raise KeyError(
            'Best metric %s is absent from epoch %d results. Available: %s' % (
                metric_name, epoch, sorted(metrics.keys())
            )
        )
    latest_value = float(metrics[metric_name])
    best_value = float(previous.get('best_value', float('-inf')))
    is_best = latest_value > best_value
    if is_best:
        best_value = latest_value
        best_epoch = int(epoch)
        save_checkpoint(
            checkpoint_state(model, optimizer, epoch, accumulated_iter),
            filename=ckpt_save_dir / 'best_model'
        )
    else:
        best_epoch = int(previous.get('best_epoch', epoch))

    summary = {
        'metric_name': metric_name,
        'best_value': best_value,
        'best_epoch': best_epoch,
        'latest_value': latest_value,
        'latest_epoch': int(epoch),
    }
    _write_json_atomic(summary_path, summary)
    if is_best:
        logger.info(
            'New best %s: %.6f at epoch %d; saved to %s' % (
                metric_name, best_value, best_epoch,
                ckpt_save_dir / 'best_model.pth'
            )
        )
    else:
        logger.info(
            'Best %s remains %.6f at epoch %d (current %.6f)' % (
                metric_name, best_value, best_epoch, latest_value
            )
        )
    return summary, is_best


def recover_best_performance(
        ckpt_save_dir, eval_output_dir, metric_name, logger):
    ckpt_save_dir = Path(ckpt_save_dir)
    eval_output_dir = Path(eval_output_dir)
    candidates = []
    summary_path = ckpt_save_dir.parent / 'best_metrics.json'
    existing_best_path = ckpt_save_dir / 'best_model.pth'
    if summary_path.is_file() and existing_best_path.is_file():
        try:
            with summary_path.open('r', encoding='utf-8') as stream:
                existing_summary = json.load(stream)
            if existing_summary.get('metric_name') == metric_name:
                candidates.append((
                    float(existing_summary['best_value']),
                    int(existing_summary['best_epoch']),
                    existing_best_path,
                ))
        except (KeyError, OSError, ValueError, TypeError):
            logger.warning(
                'Ignore invalid existing best summary: %s' % summary_path
            )
    #改进12：恢复最佳模型时同时识别nuScenes与KITTI已完成的真实评估指标。
    metrics_files = list(eval_output_dir.glob('epoch_*/val/kitti_metrics.json'))
    metrics_files += list(eval_output_dir.glob('epoch_*/val/nuscenes_metrics.json'))
    for metrics_path in metrics_files:
        match = re.match(r'^epoch_(\d+)$', metrics_path.parents[1].name)
        if match is None:
            continue
        epoch = int(match.group(1))
        checkpoint_path = ckpt_save_dir / (
            'checkpoint_epoch_%d.pth' % epoch
        )
        if not checkpoint_path.is_file():
            continue
        try:
            with metrics_path.open('r', encoding='utf-8') as stream:
                metrics = json.load(stream)
            if metric_name in metrics:
                candidates.append((
                    float(metrics[metric_name]), epoch, checkpoint_path
                ))
        except (OSError, ValueError, TypeError):
            logger.warning('Ignore invalid metrics file: %s' % metrics_path)
    #end

    if not candidates:
        return None
    best_value, best_epoch, checkpoint_path = max(
        candidates, key=lambda item: (item[0], item[1])
    )
    best_path = ckpt_save_dir / 'best_model.pth'
    if checkpoint_path.resolve() != best_path.resolve():
        shutil.copy2(str(checkpoint_path), str(best_path))
    summary = {
        'metric_name': metric_name,
        'best_value': best_value,
        'best_epoch': best_epoch,
        'latest_value': best_value,
        'latest_epoch': best_epoch,
    }
    _write_json_atomic(summary_path, summary)
    logger.info(
        'Recovered best %s %.6f at epoch %d from %s' % (
            metric_name, best_value, best_epoch, checkpoint_path
        )
    )
    return summary


def cleanup_completed_eval_artifacts(
        eval_output_dir, logger, keep_prediction_epochs=0):
    eval_output_dir = Path(eval_output_dir)
    if keep_prediction_epochs < 0 or not eval_output_dir.exists():
        return {'removed_files': 0, 'removed_bytes': 0}

    completed_epochs = []
    for result_path in eval_output_dir.glob('epoch_*/val/result.pkl'):
        #改进12：仅在对应数据集指标文件存在时清理预测缓存，保留失败评估现场。
        metrics_ready = any((result_path.parent / name).is_file() for name in
                            ('kitti_metrics.json', 'nuscenes_metrics.json'))
        match = re.match(r'^epoch_(\d+)$', result_path.parents[1].name)
        if metrics_ready and match is not None:
            completed_epochs.append((int(match.group(1)), result_path))
        #end
    completed_epochs.sort(key=lambda item: item[0])

    if keep_prediction_epochs > 0:
        removable = completed_epochs[:-keep_prediction_epochs]
    else:
        removable = completed_epochs

    removed_files = 0
    removed_bytes = 0
    for _, result_path in removable:
        if result_path.is_file():
            removed_bytes += result_path.stat().st_size
            result_path.unlink()
            removed_files += 1

        final_result_dir = result_path.parent / 'final_result'
        #改进12：nuScenes保留官方metrics_summary与结果JSON，避免清理时丢失分类别证据。
        if final_result_dir.is_dir() and not (result_path.parent / 'nuscenes_metrics.json').is_file():
            for file_path in final_result_dir.rglob('*'):
                if file_path.is_file():
                    removed_bytes += file_path.stat().st_size
                    removed_files += 1
            shutil.rmtree(str(final_result_dir))
        #end

    if removed_files > 0:
        logger.info(
            'Cleaned %d completed-evaluation prediction files (%.3f GiB). '
            'Metrics tables and failed-evaluation predictions were preserved.' % (
                removed_files, removed_bytes / float(1024 ** 3)
            )
        )
    return {
        'removed_files': removed_files,
        'removed_bytes': removed_bytes,
    }
#end

#改进14：仅在训练损失异常时检查已有输入、稀疏特征、预测和目标，定位首个非有限阶段。
def describe_nonfinite_training_tensors(model, batch):
    checked = []
    invalid = []

    def inspect(label, value, row_indices=None):
        tensor = value.features if hasattr(value, 'features') else value
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            return
        checked.append(label)
        with torch.no_grad():
            bad = ~torch.isfinite(tensor.detach())
            bad_count = int(bad.sum().item())
            if bad_count:
                first = torch.nonzero(bad, as_tuple=False)[0].cpu().tolist()
                sparse_indices = row_indices if row_indices is not None else getattr(value, 'indices', None)
                batch_index = None
                if (isinstance(sparse_indices, torch.Tensor) and first
                        and first[0] < sparse_indices.shape[0]):
                    batch_index = int(sparse_indices[first[0], 0].item())
                invalid.append('%s: bad=%d/%d, first_index=%s, batch_index=%s' % (
                    label, bad_count, tensor.numel(), first, batch_index
                ))

    for key in ('points', 'voxel_features', 'gt_boxes'):
        inspect('input.' + key, batch.get(key))
    for key, value in batch.get('multi_scale_3d_features', {}).items():
        inspect('backbone.' + key, value)
    inspect('backbone.shared_output', batch.get('spatial_features_2d'))

    network = model.module if hasattr(model, 'module') else model
    head = getattr(network, 'dense_head', None)
    if head is not None:
        saved = getattr(head, 'forward_ret_dict', {})
        prediction_indices = saved.get('spatial_indices')
        for head_index, prediction in enumerate(saved.get('pred_dicts', [])):
            for key, value in prediction.items():
                inspect('prediction.%d.%s' % (head_index, key), value,
                        prediction_indices)
        for key, values in saved.get('target_dicts', {}).items():
            if isinstance(values, (list, tuple)):
                for head_index, value in enumerate(values):
                    inspect('target.%d.%s' % (head_index, key), value)

    bad_parameters = []
    for name, parameter in network.named_parameters():
        if not torch.isfinite(parameter.detach()).all().item():
            bad_parameters.append(name)
            if len(bad_parameters) == 8:
                break
    return {'checked': checked, 'invalid': invalid,
            'nonfinite_parameters_first_8': bad_parameters}
#end

def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False, 
                    use_logger_to_record=False, logger=None, logger_iter_interval=50, cur_epoch=None, 
                    total_epochs=None, ckpt_save_dir=None, ckpt_save_time_interval=300, show_gpu_stat=False, use_amp=False):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    ckpt_save_cnt = 1
    start_it = accumulated_iter % total_it_each_epoch

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=optim_cfg.get('LOSS_SCALE_FP16', 2.0**16))

    #改进15：AMP仅检查优化器参数；训练前确认所有可训练参数都在其中，并以同一集合裁剪梯度。
    optimizer_parameters = [
        param for group in optimizer.param_groups for param in group['params']
        if param.requires_grad
    ]
    optimizer_parameter_ids = {id(param) for param in optimizer_parameters}
    untracked_parameters = [
        name for name, param in model.named_parameters()
        if param.requires_grad and id(param) not in optimizer_parameter_ids
    ]
    if untracked_parameters:
        raise RuntimeError(
            'Trainable parameters are absent from the optimizer/AMP checks: %s' %
            untracked_parameters
        )
    #end
    
    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
        data_time = common_utils.AverageMeter()
        batch_time = common_utils.AverageMeter()
        forward_time = common_utils.AverageMeter()
        batch_loss = common_utils.AverageMeter()

    end = time.time()
    for cur_it in range(start_it, total_it_each_epoch):
        try:

            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        data_timer = time.time()
        cur_data_time = data_timer - end

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()

        batch['cur_epoch'] = cur_epoch
        batch['total_epoch'] = total_epochs
        with torch.cuda.amp.autocast(enabled=use_amp): 
            loss, tb_dict, disp_dict = model_func(model, batch)

        #改进13：首次非有限loss即停止，记录分项与样本，防止继续训练或保存无效权重。
        nonfinite_terms = {}
        if not torch.isfinite(loss.detach()).all().item():
            nonfinite_terms['loss'] = float(loss.detach().float().mean().item())
        for term_name, term_value in tb_dict.items():
            if isinstance(term_value, (int, float)):
                scalar = float(term_value)
            elif isinstance(term_value, torch.Tensor) and term_value.numel() == 1:
                scalar = float(term_value.detach().item())
            else:
                continue
            if not math.isfinite(scalar):
                nonfinite_terms[term_name] = scalar
        if nonfinite_terms:
            frame_ids = repr(batch.get('frame_id', None))
            metadata = batch.get('metadata', None)
            sample_tokens = None
            if isinstance(metadata, dict):
                metadata = [metadata]
            if metadata is not None:
                sample_tokens = [
                    item.get('token', item.get('sample_token'))
                    for item in metadata if isinstance(item, dict)
                ]
            #改进14：同时报告输入与模型各阶段的有限性；只在失败时同步GPU张量。
            tensor_report = describe_nonfinite_training_tensors(model, batch)
            #end
            raise FloatingPointError(
                'Non-finite training loss before backward at epoch %s, '
                'iteration %d, global iteration %d; frame_id=%s, '
                'sample_tokens=%s, amp_scale=%s, terms=%s, tensor_report=%s' % (
                    str(cur_epoch + 1 if cur_epoch is not None else None),
                    cur_it + 1, accumulated_iter, frame_ids,
                    repr(sample_tokens),
                    str(scaler.get_scale()) if use_amp else 'disabled',
                    nonfinite_terms, tensor_report,
                )
            )
        #end
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        #改进15：裁剪范围与GradScaler反缩放范围一致，避免遗漏参数的梯度污染全部更新。
        clip_grad_norm_(optimizer_parameters, optim_cfg.GRAD_NORM_CLIP)
        #end
        scaler.step(optimizer)
        scaler.update()

        accumulated_iter += 1
 
        cur_forward_time = time.time() - data_timer
        cur_batch_time = time.time() - end
        end = time.time()

        # average reduce
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)
        avg_batch_loss = commu_utils.average_reduce_value(loss.item())

        # log to console and tensorboard
        if rank == 0:
            data_time.update(avg_data_time)
            forward_time.update(avg_forward_time)
            batch_time.update(avg_batch_time)
            batch_loss.update(avg_batch_loss)
            
            disp_dict.update({
                'loss': f'{batch_loss.val:.3f}({batch_loss.avg:.3f})', 'lr': cur_lr, 'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})', 'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})'
            })
            
            if use_logger_to_record:
                if accumulated_iter % logger_iter_interval == 0 or cur_it == start_it or cur_it + 1 == total_it_each_epoch:
                    trained_time_past_all = tbar.format_dict['elapsed']
                    second_each_iter = pbar.format_dict['elapsed'] / max(cur_it - start_it + 1, 1.0)

                    trained_time_each_epoch = pbar.format_dict['elapsed']
                    remaining_second_each_epoch = second_each_iter * (total_it_each_epoch - cur_it)
                    remaining_second_all = second_each_iter * ((total_epochs - cur_epoch) * total_it_each_epoch - cur_it)
                    
                    logger.info(
                        'Train: {:>4d}/{} ({:>3.0f}%) [{:>4d}/{} ({:>3.0f}%)]  '
                        'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                        'LR: {lr:.3e}  '
                        f'Time cost: {tbar.format_interval(trained_time_each_epoch)}/{tbar.format_interval(remaining_second_each_epoch)} ' 
                        f'[{tbar.format_interval(trained_time_past_all)}/{tbar.format_interval(remaining_second_all)}]  '
                        'Acc_iter {acc_iter:<10d}  '
                        'Data time: {data_time.val:.2f}({data_time.avg:.2f})  '
                        'Forward time: {forward_time.val:.2f}({forward_time.avg:.2f})  '
                        'Batch time: {batch_time.val:.2f}({batch_time.avg:.2f})'.format(
                            cur_epoch+1,total_epochs, 100. * (cur_epoch+1) / total_epochs,
                            cur_it,total_it_each_epoch, 100. * cur_it / total_it_each_epoch,
                            loss=batch_loss,
                            lr=cur_lr,
                            acc_iter=accumulated_iter,
                            data_time=data_time,
                            forward_time=forward_time,
                            batch_time=batch_time
                            )
                    )
                    
                    if show_gpu_stat and accumulated_iter % (3 * logger_iter_interval) == 0:
                        # To show the GPU utilization, please install gpustat through "pip install gpustat"
                        gpu_info = os.popen('gpustat').read()
                        logger.info(gpu_info)
            else:                
                pbar.update()
                pbar.set_postfix(dict(total_it=accumulated_iter))
                tbar.set_postfix(disp_dict)
                # tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
            
            # save intermediate ckpt every {ckpt_save_time_interval} seconds         
            time_past_this_epoch = pbar.format_dict['elapsed']
            if time_past_this_epoch // ckpt_save_time_interval >= ckpt_save_cnt:
                ckpt_name = ckpt_save_dir / 'latest_model'
                save_checkpoint(
                    checkpoint_state(model, optimizer, cur_epoch, accumulated_iter), filename=ckpt_name,
                )
                logger.info(f'Save latest model to {ckpt_name}')
                ckpt_save_cnt += 1
                
    if rank == 0:
        pbar.close()
        return accumulated_iter, batch_loss.avg
    else:
        return accumulated_iter, None


#改进0：训练循环接收唯一的逐epoch验证DataLoader、输出目录和最佳指标设置。
def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False, use_amp=False,
                use_logger_to_record=False, logger=None, logger_iter_interval=None, ckpt_save_time_interval=None,
                show_gpu_stat=False, cfg=None, eval_loader=None, eval_args=None,
                eval_output_dir=None, eval_interval=1,
                best_metric='mean_3d/moderate_R40',
                max_eval_prediction_epochs=0):
#end
    accumulated_iter = start_iter

    #改进0：断点续训先回收已完成评估的大型预测，再恢复best_model；指标表格与失败现场不删除。
    if rank == 0 and eval_loader is not None and eval_output_dir is not None:
        cleanup_completed_eval_artifacts(
            eval_output_dir, logger,
            keep_prediction_epochs=max_eval_prediction_epochs
        )
        recover_best_performance(
            ckpt_save_dir, eval_output_dir, best_metric, logger
        )
    #end

    #改进0：checkpoint已保存但评估中断时，续训先补齐该epoch且仅补一次，再进入下一轮训练。
    if eval_loader is not None and eval_args is not None \
            and eval_output_dir is not None and eval_interval > 0 \
            and start_epoch > 0 and start_epoch % eval_interval == 0:
        resume_result_dir = (
            Path(eval_output_dir)
            / ('epoch_%d' % start_epoch)
            / cfg.DATA_CONFIG.DATA_SPLIT['test']
        )
        #改进12：nuScenes续训按nuScenes指标文件判断是否需要补评，避免每次重复验证。
        metrics_name = ('nuscenes_metrics.json' if cfg.MODEL.POST_PROCESSING.EVAL_METRIC == 'nuscenes'
                        else 'kitti_metrics.json')
        resume_metrics_path = resume_result_dir / metrics_name
        #end
        if not resume_metrics_path.is_file():
            is_distributed = isinstance(
                model, torch.nn.parallel.DistributedDataParallel
            )
            if is_distributed:
                torch.distributed.barrier()
            eval_model = model.module if is_distributed else model
            logger.info(
                'Checkpoint epoch %d has no completed metrics; run one catch-up evaluation.' %
                start_epoch
            )
            resume_metrics = eval_utils.eval_one_epoch(
                cfg, eval_args, eval_model, eval_loader,
                start_epoch, logger,
                dist_test=is_distributed,
                result_dir=resume_result_dir,
            )
            if rank == 0:
                performance_table = format_epoch_performance_table(
                    start_epoch, None, resume_metrics, cfg.CLASS_NAMES
                )
                logger.info('\n%s' % performance_table)
                (resume_result_dir / 'performance_table.txt').write_text(
                    performance_table + '\n', encoding='utf-8'
                )
                update_best_performance(
                    model, optimizer, start_epoch, accumulated_iter,
                    resume_metrics, best_metric, ckpt_save_dir, logger
                )
                cleanup_completed_eval_artifacts(
                    eval_output_dir, logger,
                    keep_prediction_epochs=max_eval_prediction_epochs
                )
            model.train()
            torch.cuda.empty_cache()
    #end

    # use for disable data augmentation hook
    hook_config = cfg.get('HOOK', None) 
    augment_disable_flag = False

    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        batch_losses = []
        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            
            augment_disable_flag = disable_augmentation_hook(hook_config, dataloader_iter, total_epochs, cur_epoch, cfg, augment_disable_flag, logger)
            accumulated_iter, batch_loss = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter, 
                
                cur_epoch=cur_epoch, total_epochs=total_epochs,
                use_logger_to_record=use_logger_to_record, 
                logger=logger, logger_iter_interval=logger_iter_interval,
                ckpt_save_dir=ckpt_save_dir, ckpt_save_time_interval=ckpt_save_time_interval, 
                show_gpu_stat=show_gpu_stat,
                use_amp=use_amp
            )

            torch.cuda.empty_cache()
            if rank == 0:
                #改进0：日志使用与checkpoint和性能表一致的1-based epoch编号，避免看似重复一轮。
                logger.info("epoch{}_losses: {}".format(cur_epoch + 1, batch_loss))
                #end
                batch_losses.append(batch_loss)

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )

            #改进0：每个指定epoch只在此处验证一次，打印表格并按mean 3D Moderate R40保存最佳模型。
            should_evaluate = (
                eval_loader is not None
                and eval_args is not None
                and eval_output_dir is not None
                and eval_interval > 0
                and trained_epoch % eval_interval == 0
            )
            if should_evaluate:
                is_distributed = isinstance(
                    model, torch.nn.parallel.DistributedDataParallel
                )
                eval_model = model.module if is_distributed else model
                result_dir = (
                    Path(eval_output_dir)
                    / ('epoch_%d' % trained_epoch)
                    / cfg.DATA_CONFIG.DATA_SPLIT['test']
                )
                logger.info(
                    '**********************Start epoch %d evaluation**********************' %
                    trained_epoch
                )
                epoch_metrics = eval_utils.eval_one_epoch(
                    cfg, eval_args, eval_model, eval_loader,
                    trained_epoch, logger,
                    dist_test=is_distributed,
                    result_dir=result_dir,
                )
                if rank == 0:
                    performance_table = format_epoch_performance_table(
                        trained_epoch, batch_loss, epoch_metrics,
                        cfg.CLASS_NAMES
                    )
                    logger.info('\n%s' % performance_table)
                    result_dir.mkdir(parents=True, exist_ok=True)
                    (result_dir / 'performance_table.txt').write_text(
                        performance_table + '\n', encoding='utf-8'
                    )
                    for key, value in epoch_metrics.items():
                        if tb_log is not None:
                            tb_log.add_scalar(
                                'eval/' + key, float(value), trained_epoch
                            )
                    update_best_performance(
                        model, optimizer, trained_epoch,
                        accumulated_iter, epoch_metrics,
                        best_metric, ckpt_save_dir, logger
                    )
                    cleanup_completed_eval_artifacts(
                        eval_output_dir, logger,
                        keep_prediction_epochs=max_eval_prediction_epochs
                    )
                model.train()
                torch.cuda.empty_cache()
            #end
                 
        logger.info("batch_losses: {}".format(batch_losses))


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        if torch.__version__ >= '1.4':
            torch.save({'optimizer_state': optimizer_state}, optimizer_filename, _use_new_zipfile_serialization=False)
        else:
            torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    if torch.__version__ >= '1.4':
        torch.save(state, filename, _use_new_zipfile_serialization=False)
    else:
        torch.save(state, filename)


def disable_augmentation_hook(hook_config, dataloader, total_epochs, cur_epoch, cfg, flag, logger):
    """
    This hook turns off the data augmentation during training.
    """
    if hook_config is not None:
        DisableAugmentationHook = hook_config.get('DisableAugmentationHook', None)
        if DisableAugmentationHook is not None:
            num_last_epochs = DisableAugmentationHook.NUM_LAST_EPOCHS
            if (total_epochs - num_last_epochs) <= cur_epoch and not flag:
                DISABLE_AUG_LIST = DisableAugmentationHook.DISABLE_AUG_LIST
                dataset_cfg=cfg.DATA_CONFIG
                logger.info(f'Disable augmentations: {DISABLE_AUG_LIST}')
                dataset_cfg.DATA_AUGMENTOR.DISABLE_AUG_LIST = DISABLE_AUG_LIST
                dataloader._dataset.data_augmentor.disable_augmentation(dataset_cfg.DATA_AUGMENTOR)
                flag = True
    return flag
