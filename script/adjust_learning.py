import math


def adjust_learning_rate(optimizer, current_epoch, max_epoch, lr_min, lr_max, warmup=True):
    # warmup_epoch: Number of model warm-ups
    warmup_epoch = 5 if warmup else 0
    # Model warm-up phase
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    # Formal training phase of the model
    elif current_epoch < max_epoch:
        lr = lr_min + (lr_max - lr_min) * (
                1 + math.cos(math.pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    else:
        lr = lr_min + (lr_max - lr_min) * (
                1 + math.cos(math.pi * (current_epoch - max_epoch) / max_epoch)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr