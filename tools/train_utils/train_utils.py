import os

import torch
import tqdm
import time
import glob
#改进0：工程修复——支持逐epoch验证、最佳指标持久化及best checkpoint保存。
import json
import math
import shutil
from pathlib import Path
#end
from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils, commu_utils
#改进0：工程修复——复用统一验证入口；KITTI AP会在该入口中自动切换到独立进程。
from eval_utils import eval_utils
#end

#改进0：工程修复——原子记录最佳指标，并独立保存best_model.pth避免被常规checkpoint轮转删除。
def update_best_performance(model, optimizer, epoch, accumulated_iter, metrics,
                            best_metric, ckpt_save_dir, logger):
    serializable_metrics = {}
    for key, value in metrics.items():
        try:
            serializable_metrics[key] = float(value)
        except (TypeError, ValueError):
            continue

    if best_metric not in serializable_metrics:
        raise KeyError(
            'Best metric %s is absent. Available metrics: %s'
            % (best_metric, sorted(serializable_metrics.keys()))
        )
    current_value = serializable_metrics[best_metric]
    if not math.isfinite(current_value):
        raise ValueError('Best metric %s is not finite: %s' % (best_metric, current_value))

    summary_path = Path(ckpt_save_dir).parent / 'best_metrics.json'
    previous_summary = {}
    if summary_path.exists():
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                previous_summary = json.load(f)
        except (OSError, ValueError):
            logger.warning('Ignore unreadable best metric summary: %s' % summary_path)

    same_metric = previous_summary.get('best_metric') == best_metric
    previous_best = float(previous_summary.get('best_value', -math.inf)) if same_metric else -math.inf
    is_best = current_value > previous_best

    if is_best:
        best_epoch = int(epoch)
        best_value = current_value
        best_checkpoint = Path(ckpt_save_dir) / 'best_model.pth'
        save_checkpoint(
            checkpoint_state(model, optimizer, epoch, accumulated_iter),
            filename=Path(ckpt_save_dir) / 'best_model'
        )
        logger.info(
            'New best %s: %.6f at epoch %d; saved to %s'
            % (best_metric, best_value, best_epoch, best_checkpoint)
        )
    else:
        best_epoch = int(previous_summary['best_epoch'])
        best_value = previous_best
        best_checkpoint = Path(previous_summary['best_checkpoint'])
        logger.info(
            'Current %s %.6f did not exceed best %.6f at epoch %d'
            % (best_metric, current_value, best_value, best_epoch)
        )

    summary = {
        'best_metric': best_metric,
        'best_value': best_value,
        'best_epoch': best_epoch,
        'best_checkpoint': str(best_checkpoint),
        'latest_epoch': int(epoch),
        'latest_value': current_value,
        'latest_metrics': serializable_metrics,
    }
    temporary_path = summary_path.with_suffix(summary_path.suffix + '.tmp')
    with open(temporary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write('\n')
    temporary_path.replace(summary_path)
    return summary, is_best
#end

#改进0：工程修复——断点续训时可从已有逐epoch指标恢复最佳模型，覆盖评估后异常退出场景。
def recover_best_performance(ckpt_save_dir, eval_output_dir, best_metric, logger):
    summary_path = Path(ckpt_save_dir).parent / 'best_metrics.json'
    if summary_path.exists() or eval_output_dir is None:
        return

    candidates = []
    for metrics_path in Path(eval_output_dir).glob('epoch_*/**/kitti_metrics.json'):
        try:
            epoch = int(metrics_path.parents[1].name.split('_')[-1])
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            metric_value = float(metrics[best_metric])
            checkpoint_path = Path(ckpt_save_dir) / ('checkpoint_epoch_%d.pth' % epoch)
            if math.isfinite(metric_value) and checkpoint_path.exists():
                candidates.append((metric_value, epoch, metrics, checkpoint_path))
        except (OSError, ValueError, KeyError, IndexError):
            continue

    if not candidates:
        return
    best_value, best_epoch, best_metrics, checkpoint_path = max(
        candidates, key=lambda item: item[0]
    )
    best_checkpoint = Path(ckpt_save_dir) / 'best_model.pth'
    temporary_checkpoint = best_checkpoint.with_suffix('.pth.tmp')
    shutil.copy2(checkpoint_path, temporary_checkpoint)
    temporary_checkpoint.replace(best_checkpoint)
    summary = {
        'best_metric': best_metric,
        'best_value': best_value,
        'best_epoch': best_epoch,
        'best_checkpoint': str(best_checkpoint),
        'latest_epoch': best_epoch,
        'latest_value': best_value,
        'latest_metrics': {key: float(value) for key, value in best_metrics.items()},
    }
    temporary_summary = summary_path.with_suffix(summary_path.suffix + '.tmp')
    with open(temporary_summary, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write('\n')
    temporary_summary.replace(summary_path)
    logger.info(
        'Recovered best %s %.6f from epoch %d'
        % (best_metric, best_value, best_epoch)
    )
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

        #改进0：工程修复——在非有限loss污染优化器和checkpoint前立即停止，并报告具体loss分项。
        if not bool(torch.isfinite(loss.detach()).all().item()):
            non_finite_terms = {}
            for key, value in tb_dict.items():
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(numeric_value):
                    non_finite_terms[key] = numeric_value
            optimizer.zero_grad()
            raise FloatingPointError(
                'Non-finite training loss at epoch %d iteration %d; terms=%s'
                % (cur_epoch + 1, cur_it + 1, non_finite_terms)
            )
        #end

        #改进0：工程修复——在KITTI训练进度中显示分类、定位和IoU loss，便于定位数值来源。
        for term_name, term_value in tb_dict.items():
            if term_name.startswith(('hm_loss_', 'loc_loss_', 'iou_loss_')):
                try:
                    disp_dict[term_name] = '%.3f' % float(term_value)
                except (TypeError, ValueError):
                    continue
        #end
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
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


#改进0：工程修复——训练循环接收逐epoch验证所需对象和最佳指标名称，默认关闭以兼容其他调用方。
def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False, use_amp=False,
                use_logger_to_record=False, logger=None, logger_iter_interval=None, ckpt_save_time_interval=None,
                show_gpu_stat=False, cfg=None, eval_loader=None, eval_args=None, eval_output_dir=None,
                eval_interval=0, best_metric='mean_3d/moderate_R40'):
#end
    accumulated_iter = start_iter

    # use for disable data augmentation hook
    hook_config = cfg.get('HOOK', None) 
    augment_disable_flag = False

    #改进0：工程修复——进入续训循环前恢复此前已成功计算但尚未登记的最佳指标。
    if rank == 0 and eval_loader is not None:
        recover_best_performance(
            ckpt_save_dir, eval_output_dir, best_metric, logger
        )
    #end

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
                logger.info("epoch{}_losses: {}".format(cur_epoch, batch_loss))
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

            #改进0：工程修复——按配置在epoch结束后执行一次验证、记录指标并保留最佳checkpoint。
            should_evaluate = (
                rank == 0 and eval_loader is not None and eval_interval > 0
                and trained_epoch % eval_interval == 0
            )
            if should_evaluate:
                epoch_result_dir = (
                    Path(eval_output_dir) / ('epoch_%d' % trained_epoch)
                    / cfg.DATA_CONFIG.DATA_SPLIT['test']
                )
                logger.info(
                    '**********************Start epoch %d isolated evaluation**********************'
                    % trained_epoch
                )
                torch.cuda.empty_cache()
                eval_model = model.module if isinstance(
                    model, torch.nn.parallel.DistributedDataParallel
                ) else model
                try:
                    epoch_metrics = eval_utils.eval_one_epoch(
                        cfg, eval_args, eval_model, eval_loader, trained_epoch,
                        logger, dist_test=False, result_dir=epoch_result_dir
                    )
                    summary, _ = update_best_performance(
                        model, optimizer, trained_epoch, accumulated_iter,
                        epoch_metrics, best_metric, ckpt_save_dir, logger
                    )
                    if tb_log is not None:
                        for metric_name, metric_value in summary['latest_metrics'].items():
                            tb_log.add_scalar('eval/' + metric_name, metric_value, trained_epoch)
                        tb_log.add_scalar('eval/best_value', summary['best_value'], trained_epoch)
                except Exception as eval_error:
                    error_path = epoch_result_dir / 'evaluation_error.json'
                    with open(error_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            'epoch': trained_epoch,
                            'error_type': type(eval_error).__name__,
                            'error': str(eval_error),
                            'result_pkl': str(epoch_result_dir / 'result.pkl'),
                        }, f, indent=2, ensure_ascii=False)
                        f.write('\n')
                    logger.exception(
                        'Epoch %d evaluation failed; training will continue. Details: %s'
                        % (trained_epoch, error_path)
                    )
                model.train()
                torch.cuda.empty_cache()
                logger.info(
                    '**********************End epoch %d isolated evaluation**********************'
                    % trained_epoch
                )
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
