# from __future__ import print_function
from models import NvDoc
from dataset import Dataset

import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import logging
import numpy as np
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import os #兼容性
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

#  abs path
root_path = './'

#  pth file path
#pth_path = '/data/research4/mae_mask/save/net/'

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that process with rank 0 has the averaged results.
    """
    # world_size = get_world_size()
    # if world_size < 2:
        # return inp
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp


def sequence_loss(flow_preds, flow_gt, gamma=0.8):
    """ Loss function defined over sequence of flow predictions """

    i_loss = (flow_preds - flow_gt).abs().mean()

    return i_loss


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
                                              pct_start=0.1, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


def reload_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cpu')
        #print(pretrained_dict.keys())
        print(len(pretrained_dict.keys()))
        #change
        
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        
 
        
        print(len(pretrained_dict.keys()))
        #print(pretrained_dict.keys())
        
 
        
        model_dict.update(pretrained_dict)
        
        model.load_state_dict(model_dict)

        return model

def reload_pre_train_model(model, device, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        print(len(model_dict.keys()))
        pretrained_dict = torch.load(path, map_location='cuda:' + str(device))
        print("***************")
        print(len(pretrained_dict.keys()))
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if  k[7:] in model_dict and (k[7:18] == 'patch_embed' or k[7:15] == 'blocks.0'\
         or k[7:15] == 'blocks.1' or k[7:15] == 'blocks.2' or k[7:15] == 'blocks.3' or k[7:15] == 'blocks.4' or k[7:15] == 'blocks.5')}
        print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def train(train_dataloader, net, optimizer, epoch, device, logger, scheduler, opt):
    net.train()
    running_all, running_loss, running_loss_wc, running_loss_text, running_loss_bm = 0., 0., 0., 0., 0.
    for batch_idx, (im, bm_gt) in enumerate(train_dataloader, 0):
        optimizer.zero_grad()
        im = im.to(device)
        bm_gt = bm_gt.to(device)

        #print(im.shape, bm_gt.shape)
        bm_pred = net(im)
        #print(bm_pred.shape, bm_pred.dtype)
        loss = (bm_pred - bm_gt).abs().mean()

        reduced_loss = reduce_tensor(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.module.parameters(), opt.clip) #打开
        optimizer.step()

        scheduler.step()

        batch_size = im.size(0)
        running_all += batch_size
        running_loss += reduced_loss.item() * batch_size
        #print(running_loss) #调试打开
        # do checkpointing
        if opt.local_rank == 0 and epoch > 10:
            if (batch_idx+1) % int(len(train_dataloader) / 4) == 0 or batch_idx == len(train_dataloader) - 1:
                torch.save(net.state_dict(), root_path + 'save/net/epoch_%d_iter_%d.pth' % (epoch, batch_idx))

    # log
    if opt.local_rank == 0:
        logger.info('train: Epoch:{:2}\ttotal_loss: {:.8f}'.format(
            epoch,
            running_loss / (running_all * opt.world_size)
        ))

    return net, scheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--world_size', default=0)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--batchSize', type=int, default=1) #change to 1
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--net_if_need_pretrained', default=False)
    parser.add_argument('--net_trained_path', default=root_path + 'save/net/epoch_16.pth')
    parser.add_argument('--optimizer', default=root_path + 'save/optimizer/epoch_16.pth')
    parser.add_argument('--manualSeed', type=int, default=1234)

    parser.add_argument('--num_epochs', type=int, default=65)
    parser.add_argument('--num_steps', type=int)
    parser.add_argument('--wdecay', type=float, default=0.00001)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)

    opt = parser.parse_args()

    dist.init_process_group(backend="nccl", init_method='env://')  # 使用NCCL后端进行分布式 GPU 训练

    # logger
    filename = root_path + 'save/log.txt'
    logger_name = "mylog"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename, mode='a')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    # random seed
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    if opt.local_rank == 0:
        logger.info("Random Seed: {}".format(opt.manualSeed))
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)

    # world_size or gpu nums
    opt.world_size = torch.distributed.get_world_size()
    if opt.local_rank == 0:
        logger.info("World Size: {}".format(opt.world_size))

    cudnn.benchmark = True

    # train: dataset, dataloader
    train_dataset = Dataset(mode='train')
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=opt.batchSize,  # 32
                                                   shuffle=False,
                                                   num_workers=int(opt.workers),
                                                   pin_memory=True,
                                                   drop_last=True,
                                                   sampler=train_sampler)

    # net
    device = torch.device('cuda:{}'.format(opt.local_rank))  # torch.device("cuda:0")
    torch.cuda.set_device(opt.local_rank)
    model =  NvDoc(
    image_size = 256,
    patch_size = 16,
    num_classes = 1000,
    dim = 768,
    heads = 8,
    mlp_dim=1024,
    dropout=0.1,
    emb_dropout=0.1,
    token_dropout_prob=0,
    depth=8
)
    if opt.local_rank == 0:
        print(get_parameter_number(model))
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # reload_model(model, root_path+'epoch_65_iter_12757.pth')
    # reload_model(model, root_path + 'epoch_65_iter_3124.pth')
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank],
                                                output_device=opt.local_rank, find_unused_parameters=True)

    # setup optimizer
    opt.num_steps = len(train_dataloader) * opt.num_epochs
    if opt.local_rank == 0:
        logger.info('train: images numbers: {:6}\t'.format(len(train_dataset)))
        logger.info('train: epochs: {:2}\ntrain iters per epoch: {:2}\ntrain total iters: {:2}\n'.format(opt.num_epochs,
                                                                                                         len(train_dataloader),
                                                                                                         opt.num_steps))
    optimizer, scheduler = fetch_optimizer(opt, model)

    # train and test
    for epoch in range(opt.num_epochs):
        lr = scheduler.optimizer.param_groups[0]['lr']
        if opt.local_rank == 0:
            logger.info('train: Epoch:{:2}\tLr: {:.8f}\t'.format(epoch + 1, lr))
        train_sampler.set_epoch(epoch)
        model, scheduler = train(train_dataloader, model, optimizer, epoch + 1, device, logger, scheduler, opt)


if __name__ == '__main__':
    main()
