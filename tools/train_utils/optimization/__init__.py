from functools import partial

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from .fastai_optim import OptimWrapper
from .learning_schedules_fastai import CosineWarmupLR, OneCycle, CosineAnnealing


def build_optimizer(model, optim_cfg):
    if optim_cfg.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY)
    elif optim_cfg.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY,
            momentum=optim_cfg.MOMENTUM
        )
    elif optim_cfg.OPTIMIZER in ['adam_onecycle','adam_cosineanneal']:
        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]
        betas = optim_cfg.get('BETAS', (0.9, 0.99))
        betas = tuple(betas)
        optimizer_func = partial(optim.Adam, betas=betas)
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(model), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
        )
    elif optim_cfg.OPTIMIZER == 'adamw_onecycle':
        def children(m: nn.Module):
            res = m.children()
            res = list(res)
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

        optimizer_func = partial(optim.AdamW, betas=(0.9, 0.99))
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(model), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
        )
    else:
        raise NotImplementedError

    #改进15：fastai叶模块分组会遗漏父模块直接注册的Parameter（如RPDI的rpdi_scale）。
    #把遗漏参数纳入已有非BN组，使AMP反缩放、溢出检测、梯度裁剪和Adam更新覆盖同一集合。
    optimizer_param_ids = {
        id(param) for group in optimizer.param_groups for param in group['params']
    }
    missing = [
        (name, param) for name, param in model.named_parameters()
        if param.requires_grad and id(param) not in optimizer_param_ids
    ]
    if missing:
        optimizer.param_groups[0]['params'].extend(param for _, param in missing)
        print('Optimizer included previously omitted trainable parameters: %s' %
              [name for name, _ in missing])
    locations = {}
    for group_index, group in enumerate(optimizer.param_groups):
        for parameter_index, param in enumerate(group['params']):
            locations.setdefault(id(param), []).append((group_index, parameter_index))
    remaining = [
        name for name, param in model.named_parameters()
        if param.requires_grad and id(param) not in locations
    ]
    parameter_names = {id(param): name for name, param in model.named_parameters()}
    duplicates = {
        parameter_names.get(identity, '<unnamed>'): positions
        for identity, positions in locations.items() if len(positions) > 1
    }
    if remaining or duplicates:
        raise RuntimeError(
            'Optimizer parameter coverage failed: missing=%s; duplicate (group, index)=%s'
            % (remaining, duplicates)
        )
    #end
    return optimizer


def build_scheduler(optimizer, total_iters_each_epoch, total_epochs, last_epoch, optim_cfg):
    decay_steps = [x * total_iters_each_epoch for x in optim_cfg.DECAY_STEP_LIST]
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in decay_steps:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * optim_cfg.LR_DECAY
        return max(cur_decay, optim_cfg.LR_CLIP / optim_cfg.LR)

    lr_warmup_scheduler = None
    total_steps = total_iters_each_epoch * total_epochs
    if optim_cfg.OPTIMIZER == 'adam_onecycle' or optim_cfg.OPTIMIZER == 'adamw_onecycle':
        lr_scheduler = OneCycle(
            optimizer, total_steps, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
        )
    elif optim_cfg.OPTIMIZER == 'adam_cosineanneal':
        lr_scheduler = CosineAnnealing(
            optimizer, total_steps, total_epochs, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.PCT_START, optim_cfg.WARMUP_ITER
        )
    else:
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

        if optim_cfg.LR_WARMUP:
            lr_warmup_scheduler = CosineWarmupLR(
                optimizer, T_max=optim_cfg.WARMUP_EPOCH * len(total_iters_each_epoch),
                eta_min=optim_cfg.LR / optim_cfg.DIV_FACTOR
            )

    return lr_scheduler, lr_warmup_scheduler
