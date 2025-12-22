# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    Junde Wu
"""

import argparse
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from skimage import io
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from tensorboardX import SummaryWriter
#from dataset import *
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

import cfg
import function
from conf import settings
#from models.discriminatorlayer import discriminator
from dataset import *
from utils import *
import swanlab


def main():

    args = cfg.parse_args()

    seed = args.seed
    set_seed(seed)

    GPUdevice = torch.device('cuda', args.gpu_device)

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
    if args.pretrain:
        weights = torch.load(args.pretrain)
        net.load_state_dict(weights,strict=False)
    
    # 初始化SwanLab
    # run = swanlab.init(
    #     # 设置项目
    #     project=args.project_name,
    #     experiment_name=args.exp_name,
    #     # 跟踪超参数与实验元数据
    #     config={
    #         "learning_rate": args.lr,
    #         "epochs": settings.EPOCH,
    #         "batch_size": args.b,
    #         "model": args.net,
    #         "dataset": args.dataset,
    #         "seed": args.seed,
    #     },
    # )

    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay

    start_epoch = 0
    best_dice = 0.0 # 初始化最佳指标

    # 1. 设置路径和 logger
    if args.weights == 0:
        # 只有在不加载检查点时才创建新的路径
        args.path_helper = set_log_dir('logs', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])
        logger.info(args)

    '''load pretrained model'''
    if args.weights != 0:
        print(f'=> resuming from {args.weights}')
        assert os.path.exists(args.weights)
        checkpoint_file = os.path.join(args.weights)
        assert os.path.exists(checkpoint_file)
        loc = 'cuda:{}'.format(args.gpu_device)
        checkpoint = torch.load(checkpoint_file, map_location=loc)
        start_epoch = checkpoint['epoch']
        best_dice = checkpoint['best_tol']

        net.load_state_dict(checkpoint['state_dict'],strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'], strict=False)
        # ⚠️ 修复：加载 scheduler 状态 (可选，但推荐)
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        logger.info(args) # 重新打印参数，确保日志连贯性
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

    nice_train_loader, nice_test_loader = get_dataloader(args)

    '''checkpoint path and tensorboard'''
    # iter_per_epoch = len(Glaucoma_training_loader)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    # input_tensor = torch.Tensor(args.b, 3, 256, 256).cuda(device = GPUdevice)
    # writer.add_graph(net, Variable(input_tensor, requires_grad=True))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    '''begain training'''
    best_acc = 0.0
    best_tol = 1e4
    # best_dice = 0.0

    for epoch in range(start_epoch, settings.EPOCH):

        if epoch < 1:
            if args.dataset != 'REFUGE':
                tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
                logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
            else:
                tol, (eiou_cup, eiou_disc, edice_cup, edice_disc) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
                logger.info(f'Total score: {tol}, IOU_CUP: {eiou_cup}, IOU_DISC: {eiou_disc}, DICE_CUP: {edice_cup}, DICE_DISC: {edice_disc} || @ epoch {epoch}.')

            # swanlab.log(
            #         {
            #             "epoch": epoch,
            #             "val/total_score": tol,
            #             "val/iou": eiou,
            #             "val/dice": edice,
            #         }
            #     )
            
        net.train()
        time_start = time.time()
        loss = function.train_sam(args, net, optimizer, nice_train_loader, epoch, writer, vis = args.vis)
        logger.info(f'Train loss: {loss} || @ epoch {epoch}.')

        # swanlab.log({"train/loss": loss, "epoch": epoch})

        time_end = time.time()
        print('time_for_training ', time_end - time_start)

        net.eval()
        if epoch and epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:
            if args.dataset != 'REFUGE':
                tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
                logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
            else:
                tol, (eiou_cup, eiou_disc, edice_cup, edice_disc) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
                logger.info(f'Total score: {tol}, IOU_CUP: {eiou_cup}, IOU_DISC: {eiou_disc}, DICE_CUP: {edice_cup}, DICE_DISC: {edice_disc} || @ epoch {epoch}.')

            # swanlab.log(
            #     {
            #         "epoch": epoch,
            #         "val/total_score": tol,
            #         "val/iou": eiou,
            #         "val/dice": edice,
            #     }
            # )

            if args.distributed != 'none':
                sd = net.module.state_dict()
            else:
                sd = net.state_dict()


            is_best = False
            if edice > best_dice:
                best_dice = edice # ⚠️ 修复：更新 best_dice
                is_best = True

            # 始终保存一个最新的检查点 (latest_checkpoint.pth)
            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.net,
                'state_dict': sd,
                'optimizer': optimizer.state_dict(),
                'best_tol': best_dice, # 修复：保存的是当前的 best_dice
                'path_helper': args.path_helper,
            }, False, args.path_helper['ckpt_path'], filename="latest_checkpoint.pth")

            # 如果是最佳，则单独保存最佳检查点
            if is_best:
                save_checkpoint({
                'epoch': epoch + 1,
                'model': args.net,
                'state_dict': sd,
                'optimizer': optimizer.state_dict(),
                'best_tol': best_dice,
                'path_helper': args.path_helper,
            }, True, args.path_helper['ckpt_path'], filename="best_dice_checkpoint.pth")
                
        # 2. 核心修复：更新学习率
        # 学习率调度器通常在验证或训练结束后调用
        scheduler.step()
        logger.info(f'Current learning rate: {optimizer.param_groups[0]["lr"]} || @ epoch {epoch}.')

    writer.close()


if __name__ == '__main__':
    main()
