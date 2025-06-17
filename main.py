from __future__ import print_function

#----------------------------------------------------
#  Pytorch
#----------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.utils.data.distributed
import torch.distributed as dist

#--------------
#  Datalodader
#--------------
from loader import custom_dataloader

#----------------------------------------------------
#  Load CNN-architecture
#----------------------------------------------------
from models.network import get_network

#--------------
#  Datalodader
#--------------
from loss.pskd_loss import Custom_CrossEntropy_PSKD

#--------------
# Util
#--------------
from utils.dir_maker import DirectroyMaker
from utils.AverageMeter import AverageMeter
from utils.metric import metric_ece_aurc_eaurc
from utils.color import Colorer
from utils.etc import progress_bar, is_main_process, save_on_master, paser_config_save, set_logging_defaults

#----------------------------------------------------
#  Etc
#----------------------------------------------------
import os, logging
import argparse
import numpy as np


#----------------------------------------------------
#  Training Setting parser（修改为直接固定参数）
#----------------------------------------------------
def parse_args():
    # 直接定义并设置参数值
    args = argparse.Namespace()
    args.lr = 0.1
    args.lr_decay_rate = 0.1
    args.lr_decay_schedule = [150, 225]
    args.weight_decay = 5e-4
    args.start_epoch = 0
    args.end_epoch = 300
    args.PSKD = True  # 根据需求决定是否开启 PSKD
    args.batch_size = 128
    args.experiments_dir = 'models'
    args.classifier_type = 'ResNet18'
    args.data_path = '/kaggle/working/'  # 按实际数据路径调整
    args.data_type = 'cifar100'
    args.alpha_T = 0.8
    args.saveckp_freq = 299
    args.rank = -1
    args.world_size = 1
    args.dist_backend = 'nccl'
    args.dist_url = 'tcp://127.0.0.1:8080'
    args.workers = 40
    args.multiprocessing_distributed = False  # 根据需求调整
    args.resume = None  # 若要加载模型，填对应路径

    # 简单参数校验（保留必要逻辑，也可简化/删除）
    assert args.end_epoch >= 1, 'number of epochs must be larger than or equal to one'
    assert args.batch_size >= 1, 'batch size must be larger than or equal to one'
    return args


#----------------------------------------------------
#  Adjust_learning_rate & get_learning_rate  
#----------------------------------------------------
def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    for milestone in args.lr_decay_schedule:
        lr *= args.lr_decay_rate if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


#----------------------------------------------------
#  Top-1 / Top -5 accuracy
#----------------------------------------------------
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


#----------------------------------------------------
#  Colour print 
#----------------------------------------------------
C = Colorer.instance()


def main():
    args = parse_args()
    print(C.green("[!] Start the PS-KD."))
    print(C.green("[!] Created by LG CNS AI Research(LAIR)"))

    dir_maker = DirectroyMaker(root=args.experiments_dir, save_model=True, save_log=True, save_config=True)
    model_log_config_dir = dir_maker.experiments_dir_maker(args)
    model_dir, log_dir, config_dir = model_log_config_dir

    paser_config_save(args, config_dir)

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, model_dir, log_dir, args))
        print(C.green("[!] Multi/Single Node, Multi-GPU All multiprocessing_distributed Training Done."))
        print(C.underline(C.red2('[Info] Save Model dir:')), C.red2(model_dir))
        print(C.underline(C.red2('[Info] Log dir:')), C.red2(log_dir))
        print(C.underline(C.red2('[Info] Config dir:')), C.red2(config_dir))
    else:
        print(C.green("[!] Multi/Single Node, Single-GPU per node, multiprocessing_distributed Training Done."))
        main_worker(0, ngpus_per_node, model_dir, log_dir, args)
        print(C.green("[!] All Single GPU Training Done"))
        print(C.underline(C.red2('[Info] Save Model dir:')), C.red2(model_dir))
        print(C.underline(C.red2('[Info] Log dir:')), C.red2(log_dir))
        print(C.underline(C.red2('[Info] Config dir:')), C.red2(config_dir))


def main_worker(gpu, ngpus_per_node, model_dir, log_dir, args):
    best_acc = 0
    net = get_network(args)

    args.ngpus_per_node = ngpus_per_node
    args.gpu = gpu
    if args.gpu is not None:
        print(C.underline(C.yellow("[Info] Use GPU : {} for training".format(args.gpu))))

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                                rank=args.rank)
    print(C.green("[!] [Rank {}] Distributed Init Setting Done.".format(args.rank)))

    if not torch.cuda.is_available():
        print(C.red2("[Warnning] Using CPU, this will be slow."))
    elif args.distributed:
        if args.gpu is not None:
            print(C.green("[!] [Rank {}] Distributed DataParallel Setting Start".format(args.rank)))
            torch.cuda.set_device(args.gpu)
            net.cuda(args.gpu)
            args.workers = int((args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
            args.batch_size = int(args.batch_size / args.ngpus_per_node)
            print(C.underline(C.yellow("[Info] [Rank {}] Workers: {}".format(args.rank, args.workers))))
            print(C.underline(C.yellow("[Info] [Rank {}] Batch_size: {}".format(args.rank, args.batch_size))))
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])
            print(C.green("[!] [Rank {}] Distributed DataParallel Setting End".format(args.rank)))
        else:
            net.cuda()
            net = torch.nn.parallel.DistributedDataParallel(net)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        net = net.cuda(args.gpu)
    else:
        net = torch.nn.DataParallel(net).cuda()

    set_logging_defaults(log_dir, args)

    train_loader, valid_loader, train_sampler = custom_dataloader.dataloader(args)

    criterion_CE = nn.CrossEntropyLoss().cuda(args.gpu)
    if args.PSKD:
        criterion_CE_pskd = Custom_CrossEntropy_PSKD().cuda(args.gpu)
    else:
        criterion_CE_pskd = None
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
                                nesterov=True)

    if args.PSKD:
        all_predictions = torch.zeros(len(train_loader.dataset), len(train_loader.dataset.classes),
                                      dtype=torch.float32)
        print(C.underline(C.yellow("[Info] all_predictions matrix shape {}".format(all_predictions.shape))))
    else:
        all_predictions = None

    if args.resume:
        if args.gpu is None:
            checkpoint = torch.load(args.resume)
        else:
            if args.distributed:
                dist.barrier()
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
        args.start_epoch = checkpoint['epoch'] + 1
        alpha_t = checkpoint['alpha_t']
        best_acc = checkpoint['best_acc']
        all_predictions = checkpoint['prev_predictions'].cpu()
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(C.green("[!] [Rank {}] Model loaded".format(args.rank)))
        del checkpoint

    cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.end_epoch):
        adjust_learning_rate(optimizer, epoch, args)
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if args.PSKD:
            alpha_t = args.alpha_T * ((epoch + 1) / args.end_epoch)
            alpha_t = max(0, alpha_t)
        else:
            alpha_t = -1

        all_predictions = train(all_predictions, criterion_CE, criterion_CE_pskd, optimizer, net, epoch, alpha_t,
                                train_loader, args)

        if args.distributed:
            dist.barrier()

        acc = val(criterion_CE, net, epoch, valid_loader, args)

        save_dict = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'best_acc': best_acc,
            'accuracy': acc,
            'alpha_t': alpha_t,
            'prev_predictions': all_predictions
        }

        if acc > best_acc:
            best_acc = acc
            save_on_master(save_dict, os.path.join(model_dir, 'checkpoint_best.pth'))
            if is_main_process():
                print(C.green("[!] Save best checkpoint."))

        if args.saveckp_freq and (epoch + 1) % args.saveckp_freq == 0:
            save_on_master(save_dict, os.path.join(model_dir, f'checkpoint_{epoch:03}.pth'))
            if is_main_process():
                print(C.green("[!] Save checkpoint."))

    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()
        print(C.green("[!] [Rank {}] Distroy Distributed process".format(args.rank)))


#-------------------------------
# Train 
#------------------------------- 
def train(all_predictions, criterion_CE, criterion_CE_pskd, optimizer, net, epoch, alpha_t, train_loader, args):
    train_top1 = AverageMeter()
    train_top5 = AverageMeter()
    train_losses = AverageMeter()
    correct = 0
    total = 0

    net.train()
    current_LR = get_learning_rate(optimizer)[0]

    for batch_idx, (inputs, targets, input_indices) in enumerate(train_loader):
        if args.gpu is not None:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        if args.PSKD:
            targets_numpy = targets.cpu().detach().numpy()
            identity_matrix = torch.eye(len(train_loader.dataset.classes))
            targets_one_hot = identity_matrix[targets_numpy]

            if epoch == 0:
                all_predictions[input_indices] = targets_one_hot

            soft_targets = ((1 - alpha_t) * targets_one_hot) + (alpha_t * all_predictions[input_indices])
            soft_targets = soft_targets.cuda()

            outputs = net(inputs)
            softmax_output = F.softmax(outputs, dim=1)
            loss = criterion_CE_pskd(outputs, soft_targets)

            if args.distributed:
                gathered_prediction = [torch.ones_like(softmax_output) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_prediction, softmax_output)
                gathered_prediction = torch.cat(gathered_prediction, dim=0)

                gathered_indices = [torch.ones_like(input_indices.cuda()) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_indices, input_indices.cuda())
                gathered_indices = torch.cat(gathered_indices, dim=0)
        else:
            outputs = net(inputs)
            loss = criterion_CE(outputs, targets)

        train_losses.update(loss.item(), inputs.size(0))
        err1, err5 = accuracy(outputs.data, targets, topk=(1, 5))
        train_top1.update(err1.item(), inputs.size(0))
        train_top5.update(err5.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if args.PSKD:
            if args.distributed:
                for jdx in range(len(gathered_prediction)):
                    all_predictions[gathered_indices[jdx]] = gathered_prediction[jdx].detach()
            else:
                all_predictions[input_indices] = softmax_output.cpu().detach()

        progress_bar(epoch, batch_idx, len(train_loader), args,
                     'lr: {:.1e} | alpha_t: {:.3f} | loss: {:.3f} | top1_acc: {:.3f} | top5_acc: {:.3f} | correct/total({}/{})'.format(
                         current_LR, alpha_t, train_losses.avg, train_top1.avg, train_top5.avg, correct, total))

    if args.distributed:
        dist.barrier()

    logger = logging.getLogger('train')
    logger.info('[Rank {}] [Epoch {}] [PSKD {}] [lr {:.1e}] [alpht_t {:.3f}] [train_loss {:.3f}] [train_top1_acc {:.3f}] [train_top5_acc {:.3f}] [correct/total {}/{}]'.format(
        args.rank,
        epoch,
        args.PSKD,
        current_LR,
        alpha_t,
        train_losses.avg,
        train_top1.avg,
        train_top5.avg,
        correct,
        total))

    return all_predictions


#-------------------------------          
# Validation
#------------------------------- 
def val(criterion_CE, net, epoch, val_loader, args):
    val_top1 = AverageMeter()
    val_top5 = AverageMeter()
    val_losses = AverageMeter()

    targets_list = []
    confidences = []

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(val_loader):
            if args.gpu is not None:
                inputs = inputs.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True)

            targets_numpy = targets.cpu().numpy()
            targets_list.extend(targets_numpy.tolist())

            outputs = net(inputs)

            softmax_predictions = F.softmax(outputs, dim=1)
            softmax_predictions = softmax_predictions.cpu().numpy()
            for values_ in softmax_predictions:
                confidences.append(values_.tolist())

            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            loss = criterion_CE(outputs, targets)
            val_losses.update(loss.item(), inputs.size(0))

            err1, err5 = accuracy(outputs.data, targets, topk=(1, 5))
            val_top1.update(err1.item(), inputs.size(0))
            val_top5.update(err5.item(), inputs.size(0))

            progress_bar(epoch, batch_idx, len(val_loader), args,
                         'val_loss')
