#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import argparse
import logging
from pathlib import Path
import time
import os

import numpy as np
import torch

from torch.utils import data
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from dataset.datasets import WFLWDatasets, A300WDatasets
from models.pfld import PFLDInference
from models.pfld_vovnet2 import vovnet_pfld
from pfld.loss import PFLDLoss, WingLoss, AdaptiveWingLoss
from pfld.utils import AverageMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info('Save checkpoint to {0:}'.format(filename))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def train(train_loader, pfld_backbone, criterion, optimizer,
          epoch):
    losses = AverageMeter()

    for img, landmark_gt, attribute_gt, euler_angle_gt in train_loader:
        img = img.to(device)
        attribute_gt = attribute_gt.to(device)
        landmark_gt = landmark_gt.to(device)
        euler_angle_gt = euler_angle_gt.to(device)
        pfld_backbone = pfld_backbone.to(device)
        landmarks = pfld_backbone(img)
        loss = criterion(landmark_gt, landmarks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item())
    return loss


def validate(val_dataloader, pfld_backbone, criterion):
    pfld_backbone.eval()
    losses = []
    with torch.no_grad():
        for img, landmark_gt, attribute_gt, euler_angle_gt in val_dataloader:
            img = img.to(device)
            # attribute_gt = attribute_gt.to(device)
            landmark_gt = landmark_gt.to(device)
            # euler_angle_gt = euler_angle_gt.to(device)
            pfld_backbone = pfld_backbone.to(device)
            landmark = pfld_backbone(img)
            # loss = torch.mean(torch.sum((landmark_gt - landmark)**2, axis=1))
            loss = criterion(landmark_gt, landmark)
            losses.append(loss.cpu().numpy())
    print("===> Evaluate:")
    print('Eval set: Average loss: {:.4f} '.format(np.mean(losses)))
    return np.mean(losses)


def main(args):
    # Step 1: parse args config
    logging.basicConfig(
        format=
        '[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log_file, mode='w'),
            logging.StreamHandler()
        ])
    print_args(args)
    assert(args.num_landmarks in [68,98])
    # Step 2: model, criterion, optimizer, scheduler
    if(args.backbone=='VoVNet'):
        pfld_backbone = vovnet_pfld(num_landmarks = args.num_landmarks).to(device)
    elif(args.backbone=='MobileNet'):
        pfld_backbone = PFLDInference(num_landmarks = args.num_landmarks).to(device)
    criterion = WingLoss()
    optimizer = torch.optim.Adam(
        [{
            'params': pfld_backbone.parameters()
        }, 
        ],
        lr=args.base_lr,
        weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.lr_patience, verbose=True)

    # step 3: data
    # argumetion
    transform = transforms.Compose([transforms.ToTensor()])
    if(args.num_landmarks == 68):
        train_dataset = A300WDatasets(args.dataroot, transform, True)
        val_dataset = A300WDatasets(args.val_dataroot, transform, False)
    elif(args.num_landmarks == 98):
        train_dataset = WFLWDatasets(args.dataroot, transform, True)
        val_dataset = WFLWDatasets(args.val_dataroot, transform, False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batchsize,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.workers)

    # step 4: run
    writer = SummaryWriter(args.tensorboard)
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        train_loss = train(train_dataloader, pfld_backbone,
                                      criterion, optimizer, epoch)
        print("train loss: " + str(train_loss.item()))
        if(epoch%50==0):
            filename = os.path.join(
            str(args.snapshot), "checkpoint_epoch_" + str(epoch) + '.pth.tar')
            save_checkpoint({
                'epoch': epoch,
                'plfd_backbone': pfld_backbone.state_dict(),
            }, filename)

        val_loss = validate(val_dataloader, pfld_backbone, criterion)
        writer.add_scalars('data/loss', {'val loss': val_loss, 'train loss': train_loss}, epoch)
        scheduler.step(val_loss)
        
    writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description='pfld')
    # general
    parser.add_argument('-j', '--workers', default=4, type=int)
    parser.add_argument('--devices_id', default='0', type=str)  #TBD
    parser.add_argument('--test_initial', default='false', type=str2bool)  #TBD

    # training
    ##  -- optimizer
    parser.add_argument('--base_lr', default=0.0001, type=int)
    parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float)

    # -- lr
    parser.add_argument("--lr_patience", default=30, type=int)

    # -- epoch
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--end_epoch', default=300, type=int)
    
    num=1
    # -- snapshot、tensorboard log and checkpoint
    parser.add_argument(
        '--snapshot',
        default='./checkpoint/'+str(num)+'/',
        type=str,
        metavar='PATH')
    parser.add_argument(
        '--log_file', default="./checkpoint/"+str(num)+"/train.logs", type=str)
    parser.add_argument(
        '--tensorboard', default="./checkpoint/"+str(num)+"/tensorboard", type=str)
    parser.add_argument(
        '--resume', default='', type=str, metavar='PATH')  # TBD

    # --dataset
    parser.add_argument(
        '--dataroot',
        default='./data/train_data/list.txt',
        type=str,
        metavar='PATH')
    parser.add_argument(
        '--val_dataroot',
        default='./data/test_data/list.txt',
        type=str,
        metavar='PATH')
    parser.add_argument('--train_batchsize', default=256, type=int)
    parser.add_argument('--val_batchsize', default=8, type=int)
    parser.add_argument('--num_landmarks', default=68, type=int)
    parser.add_argument('--backbone', default='MobileNet', type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
