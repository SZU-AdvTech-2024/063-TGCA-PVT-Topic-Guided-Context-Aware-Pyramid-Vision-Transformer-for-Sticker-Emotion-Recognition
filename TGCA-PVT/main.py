# Code based on the Pyramid Vision Transformer
# https://github.com/whai362/PVT
# Licensed under the Apache License, Version 2.0

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import torch.nn as nn   # cx
import torch.nn.functional as F   # cx
from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from dataset import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler
from model import TGCAPVT
import utils
import collections
import os


def get_args_parser():
    parser = argparse.ArgumentParser('TGCAPVT training and evaluation script', add_help=False)
    parser.add_argument('--fp32-resume', action='store_true', default=False)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--config', default='configs/pvt/pvt_small.py', type=str, help='config')

    # Vision Model parameters
    parser.add_argument('--model', default='pvt_small', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=448, type=int, help='images input size')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    # Language Model parameters
    parser.add_argument('--tunebert', default=False, type=bool)
    parser.add_argument('--bert-dim', default=768, type=int)
    parser.add_argument('--bert-model', default='bert-base-uncased', type=str)
    # Fusion Model parameters
    parser.add_argument('--hidden-dim', default=768, type=int)
    parser.add_argument('--fuse-mlp-dim', default=3072, type=int)
    parser.add_argument('--fuse-dropout-rate', default=0.1, type=float)
    parser.add_argument('--fuse-num-heads', default=12, type=int)
    parser.add_argument('--fuse-attention-dropout-rate', default=0.0, type=float)
    
    # parser.add_argument('--model-ema', action='store_true')
    # parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    # parser.set_defaults(model_ema=True)
    # parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    # parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-5)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    '''
    rand-m9-mstd0.5-inc1 是一个 RandAugment 策略的预定义配置，具体解释如下：

    rand:
    表示 RandAugment 策略，它是一种自动化的数据增强方法。
    相较于传统 AutoAugment，RandAugment 通过少量超参数（如增强强度和操作数量）生成多种增强组合，而不需要手动定义策略表。
    m9:
    m 表示增强的强度（magnitude），数值范围通常为 1-10。
    m9 表示增强强度设定为 9，数值越高，增强的变化幅度越大。
    强度可能影响增强操作的程度，如旋转角度、裁剪区域大小、颜色抖动范围等。
    mstd0.5:
    mstd 表示增强强度的标准差（magnitude standard deviation）。
    mstd0.5 表示增强强度会围绕 m9 波动，标准差为 0.5。
    引入随机性，使增强策略在训练过程中更具多样性。
    inc1:
    inc 表示操作数量（increase step count）。
    inc1 表示每次增强应用一个操作。
    如果设置更高的值（如 inc2 或 inc3），则每个样本可能依次应用多个增强操作。
    总结：
    
    rand-m9-mstd0.5-inc1 的含义是：使用 RandAugment 策略，增强强度为 9，强度随机波动范围为 0.5，每次增强应用 1 个操作。
    '''
    parser.add_argument('--color-jitter', type=float, default=0.1, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.2, help='Label smoothing (default: 0.1)')
    '''
    2. interpolation=args.train_interpolation 的作用
        interpolation 控制在图像缩放或裁剪过程中使用的插值算法。
        参数 args.train_interpolation 指定了插值的类型，用于决定如何在图像缩放时填充新增像素。
    默认值 bicubic 的含义
    bicubic 是双三次插值方法（Bicubic Interpolation）。
        它是一种基于邻域的插值方法，考虑周围 16 个像素（4×4 网格）来计算新像素的值。
        双三次插值可以生成更平滑的图像，适用于训练图像输入，因为平滑图像可能提高模型的泛化性能。
    其他插值方式：
    bilinear：双线性插值，计算新像素值时仅考虑最近的 4 个像素（2×2 网格）。计算效率较高，但生成的图像可能不如双三次插值平滑。
    nearest：最近邻插值，仅复制最近像素的值，不进行加权平均，生成的图像可能有明显的块状效果。
    random：随机选择一种插值方法。
    总结：
    interpolation 是一种与数据增强相关的参数，用于处理图像尺寸变化时的像素填充方法。
    默认值 bicubic 平滑且效果较优，是训练常用的选项
    '''
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    '''
    re_prob（随机擦除概率）：
        表示随机擦除的应用概率。
        范围：0 到 1。
        例如，re_prob=0.5 表示有 50% 的概率对图像应用随机擦除操作。
    re_mode（擦除模式）：
        定义擦除区域的像素填充方式。
        常见值：
            'pixel'：被遮挡区域填充为随机像素值。
            'constant'：被遮挡区域填充为固定值（如 0，表示黑色）。
            'mean'：被遮挡区域填充为图像均值。
        默认值 pixel 使遮挡区域更具随机性。
    re_count（擦除次数）：
        指定每张图像上随机擦除的次数。
        默认值为 1，表示每张图像仅擦除一个随机区域。
        较高的值（如 re_count=3）可以对图像进行多次遮挡，从而增加复杂性。
    总结：
        随机擦除是一种模拟信息丢失的增强方式，参数的选择可以平衡增强的强度和复杂性：
        re_prob 控制应用的频率。
        re_mode 控制被遮挡区域的填充方式。
        re_count 决定遮挡的数量。
    '''
    parser.add_argument('--reprob', type=float, default=0, metavar='PCT',
                        help='Random erase prob (default: 0)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0)')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    # parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
    #                     help='Name of teacher model to train (default: "regnety_160"')
    # parser.add_argument('--teacher-path', type=str, default='')
    # parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    # parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    # parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument('--visfinetune', default='', help='finetune from checkpoint')
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--dataset', default='SER',
                        choices=['SER', 'SER_V', 'FI', 'EmotionROI'], type=str)
    parser.add_argument('--data-path', default='/root/autodl-tmp/cx/ser30k', type=str,
                        help='dataset path')
    parser.add_argument('--use-mcloader', action='store_true', default=False, help='Use mcloader')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # hyper-parameter
    parser.add_argument('--alpha', default=8, type=int, help='alpha')
    parser.add_argument('--locals', default=[1, 1, 1, 0], nargs='+', type=int, help='locals')
    return parser

# Loss function
class CustomLoss2(nn.Module):
    def __init__(self):
        super(CustomLoss2, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, predicted_probabilities, targets):
        # 计算整体的交叉熵损失
        log_probs = F.log_softmax(predicted_probabilities, dim=1)
        loss1 = F.nll_loss(log_probs, targets)

        # 提取积极、消极和中立情绪的概率
        predicted_pro = F.softmax(predicted_probabilities, dim=1)

        # 积极情绪对应类别 0 和 1 (labels 0)
        positive_pro = torch.sum(predicted_pro[:, :2], dim=1)
        # 消极情绪对应类别 2-5 (labels 1)
        negative_pro = torch.sum(predicted_pro[:, 2:6], dim=1)
        # 中立情绪对应类别 6 (labels 2)
        neutral_pro = predicted_pro[:, 6]

        # 创建新的目标张量，分别对应积极、消极和中立情绪
        new_targets = torch.zeros(targets.size(0), dtype=torch.long)

        # 这里创建三类目标标签
        new_targets[targets < 2] = 0  # 积极情绪对应类别标签为0
        new_targets[(targets >= 2) & (targets <= 5)] = 1  # 消极情绪对应类别标签为1
        new_targets[targets == 6] = 2  # 中立情绪对应类别标签为2

        # 将 new_targets 移动到与预测概率相同的设备上
        new_targets = new_targets.to(predicted_pro.device)

        # 将积极、消极和中立情绪的概率堆叠为一个新的张量
        log_probs_2 = torch.log(torch.stack([positive_pro, negative_pro, neutral_pro], dim=1))

        # 计算三类交叉熵损失
        loss2 = F.nll_loss(log_probs_2, new_targets)

        # 最终损失为整体损失和三类情绪损失之和
        final_loss = loss1 + loss2

        return final_loss

def main(args):
    utils.init_distributed_mode(args)
    print(args)
    # if args.distillation_type != 'none' and args.finetune and not args.eval:
    #     raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train,
                # num_replicas=num_tasks,
                num_replicas=0,
                rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val,
                # num_replicas=num_tasks,
                num_replicas=0,
                rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    print(f"Creating model...")
    model = TGCAPVT(args)
    if args.visfinetune:
        checkpoint = torch.load(args.visfinetune, map_location='cpu')
        if 'model' in checkpoint:
            checkpoint_model = checkpoint['model']
        else:
            checkpoint_model = checkpoint
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model:
                del checkpoint_model[k]
        print("Load model.vision_model weights...")
        model.vision_model.load_state_dict(checkpoint_model, strict=False)
    
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        if 'model' in checkpoint:
            checkpoint_model = checkpoint['model']
        else:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:  # , 'pos_embed'
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        print(f"=========>Load from {args.finetune}...")
        model.load_state_dict(checkpoint_model, strict=False)
        print("=========>Load successfully")

    model.to(device)

    model_ema = None
    # if args.model_ema:
    #     # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
    #     model_ema = ModelEma(
    #         model,
    #         decay=args.model_ema_decay,
    #         device='cpu' if args.model_ema_force_cpu else '',
    #         resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
        print("==========use 1=============")
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        # criterion = CustomLoss2()  # cx
    else:
        criterion = torch.nn.CrossEntropyLoss()
        print("===========use 2==========")

    # teacher_model = None
    # if args.distillation_type != 'none':
    #     assert args.teacher_path, 'need to specify teacher-path when using distillation'
    #     print(f"Creating teacher model: {args.teacher_model}")
    #     teacher_model = create_model(
    #         args.teacher_model,
    #         pretrained=False,
    #         num_classes=args.nb_classes,
    #         global_pool='avg',
    #     )
    #     if args.teacher_path.startswith('https'):
    #         checkpoint = torch.hub.load_state_dict_from_url(
    #             args.teacher_path, map_location='cpu', check_hash=True)
    #     else:
    #         checkpoint = torch.load(args.teacher_path, map_location='cpu')
    #     teacher_model.load_state_dict(checkpoint['model'])
    #     teacher_model.to(device)
    #     teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    # criterion = DistillationLoss(
    #     criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    # )
    criterion = DistillationLoss(
        criterion, None, 'none', 0, 0
    )

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        if 'model' in checkpoint:
            msg = model_without_ddp.load_state_dict(checkpoint['model'])
        else:
            msg = model_without_ddp.load_state_dict(checkpoint)
        print(msg)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            # if args.model_ema:
            #     utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.fp32_resume and epoch > args.start_epoch + 1:
            args.fp32_resume = False
        loss_scaler._scaler = torch.cuda.amp.GradScaler(enabled=not args.fp32_resume)

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=True,   # args.finetune == '',  # keep in eval mode during finetuning
            fp32=args.fp32_resume
        )

        # 清理缓存
        torch.cuda.empty_cache()  # 在每个 epoch 结束后释放缓存
        lr_scheduler.step(epoch)


        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        if test_stats["acc1"] >= max_accuracy:
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint_best5.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        # 'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)
        
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # 检查保存文件是否存在
    output_dir = "checkpoints/SER"
    checkpoint_file = os.path.join(output_dir, "checkpoint_best5.pth")
    if os.path.exists(checkpoint_file):
        print("Checkpoint saved successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args = utils.update_from_config(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
