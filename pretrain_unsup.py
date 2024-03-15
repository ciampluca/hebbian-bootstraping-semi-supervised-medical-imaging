from torchvision import transforms, datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import time
import os
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.backends import cudnn
import random

from config.dataset_config.dataset_cfg import dataset_cfg
from config.train_test_config.train_test_config import print_val_loss_sup, print_train_eval_sup, print_val_eval_sup, save_val_best_sup_2d, draw_pred_sup, print_best_sup
from config.visdom_config.visual_visdom import visdom_initialization_sup, visualization_sup, visual_image_sup
from config.warmup_config.warmup import GradualWarmupScheduler
from config.augmentation.online_aug import data_transform_2d, data_normalize_2d
from loss.loss_function import segmentation_loss
from models.getnetwork import get_network
from dataload.dataset_2d import imagefloder_itn

from hebb.makehebbian import makehebbian

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


# TODO aggiungere supporto tensorboard
# TODO aggiungere suppporto salvataggio risultati
# TODO aggiungere supporto per fare esperimenti su una frazione di dati, cosi come è ora è fatto a merda; in realtà guardare meglio c'è la possibilità di passare num images nel dataset


def init_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def print_train_loss_unsup(train_loss, num_batches, print_num, print_num_minus):
    train_epoch_loss = train_loss / num_batches['pretrain_unsup']
    print('-' * print_num)
    print('| Train Loss: {:.4f}'.format(train_epoch_loss).ljust(print_num_minus, ' '), '|')
    print('-' * print_num)
    return train_epoch_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--path_trained_models', default='results/XNet/checkpoints/unsup')
    parser.add_argument('--path_seg_results', default='results/XNet/seg_pred/unsup')
    parser.add_argument('--path_dataset', default='datasets/GlaS')
    parser.add_argument('--dataset_name', default='GlaS', help='CREMI, ISIC-2017, GlaS')
    parser.add_argument('--input1', default='image')
    parser.add_argument('--sup_mark', default='100')
    parser.add_argument('-b', '--batch_size', default=4, type=int)
    parser.add_argument('-e', '--num_epochs', default=200, type=int)
    parser.add_argument('-s', '--step_size', default=50, type=int)
    parser.add_argument('-l', '--lr', default=0.5, type=float)
    parser.add_argument('-g', '--gamma', default=0.5, type=float)
    parser.add_argument('--loss', default='dice', type=str)
    parser.add_argument('-ds', '--deep_supervision', default=False)
    parser.add_argument('-w', '--warm_up_duration', default=20)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow')

    parser.add_argument('-i', '--display_iter', default=5, type=int)
    parser.add_argument('-n', '--network', default='unet', type=str)
    parser.add_argument('--exclude', nargs='*', default=['Conv_1x1'], type=str, help="Full name of the layers to exclude from conversion to Hebbian. These names depend on how they were called in the specific network that you wish to use.")
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--rank_index', default=0, help='0, 1, 2, 3')
    parser.add_argument('-v', '--vis', default=False, help='need visualization or not')
    parser.add_argument('--visdom_port', default=16672, help='16672')
    
    parser.add_argument('--hebb_mode', default='swta_t', type=str)
    parser.add_argument('--hebb_inv_temp', default=50., type=float)
    parser.add_argument('--hebb_w_nrm', default=True, type=bool)
    parser.add_argument('--hebb_alpha', default=0., type=float)
    
    args = parser.parse_args()

    #torch.cuda.set_device(args.local_rank)
    #dist.init_process_group(backend='nccl', init_method='env://')

    #rank = torch.distributed.get_rank()
    #ngpus_per_node = torch.cuda.device_count()

    torch.cuda.set_device(0)
    init_seeds(1)

    dataset_name = args.dataset_name
    cfg = dataset_cfg(dataset_name)

    print_num = 42 + (cfg['NUM_CLASSES'] - 3) * 7
    print_num_minus = print_num - 2

    path_trained_models = args.path_trained_models + '/' + str(os.path.split(args.path_dataset)[1])
    os.makedirs(path_trained_models, exist_ok=True)
    path_trained_models = path_trained_models+'/'+str(args.network)+'-l='+str(args.lr)+'-e='+str(args.num_epochs)+'-s='+str(args.step_size)+'-g='+str(args.gamma)+'-b='+str(args.batch_size)+'-w='+str(args.warm_up_duration)+'-'+str(args.sup_mark)
    os.makedirs(path_trained_models, exist_ok=True)

    path_seg_results = args.path_seg_results + '/' + str(os.path.split(args.path_dataset)[1])
    os.makedirs(path_seg_results, exist_ok=True)
    path_seg_results = path_seg_results+'/'+str(args.network)+'-l='+str(args.lr)+'-e='+str(args.num_epochs)+'-s='+str(args.step_size)+'-g='+str(args.gamma)+'-b='+str(args.batch_size)+'-w='+str(args.warm_up_duration)+'-'+str(args.sup_mark)
    os.makedirs(path_seg_results, exist_ok=True)

    if args.vis:
        visdom_env = str('Sup-'+str(os.path.split(args.path_dataset)[1])+'-'+args.network+'-l='+str(args.lr)+'-e='+str(args.num_epochs)+'-s='+str(args.step_size)+'-g='+str(args.gamma)+'-b='+str(args.batch_size)+'-w='+str(args.warm_up_duration)+'-'+str(args.sup_mark))
        visdom = visdom_initialization_sup(env=visdom_env, port=args.visdom_port)

    if args.input1 == 'image':
        input1_mean = 'MEAN'
        input1_std = 'STD'
    else:
        input1_mean = 'MEAN_' + args.input1
        input1_std = 'STD_' + args.input1

    # Dataset
    data_transforms = data_transform_2d()
    data_normalize = data_normalize_2d(cfg[input1_mean], cfg[input1_std])

    dataset_train = imagefloder_itn(
        data_dir=args.path_dataset + '/train',
        input1=args.input1,
        data_transform_1=data_transforms['train'],
        data_normalize_1=data_normalize,
        sup=True,
        num_images=None,
    )
    dataset_val = imagefloder_itn(
        data_dir=args.path_dataset + '/val',
        input1=args.input1,
        data_transform_1=data_transforms['val'],
        data_normalize_1=data_normalize,
        sup=True,
        num_images=None,
    )

    #train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, shuffle=True)
    #val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False)

    dataloaders = dict()
    dataloaders['train'] = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8)
    dataloaders['val'] = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8)

    num_batches = {'pretrain_unsup': len(dataloaders['train']), 'val': len(dataloaders['val'])}

    # Model
    model = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'])
    makehebbian(model, exclude=args.exclude, hebb_params={'mode': args.hebb_mode, 'k': args.hebb_inv_temp, 'w_nrm': args.hebb_w_nrm, 'alpha': args.hebb_alpha})
    #for p in model.parameters(): p.requires_grad = True # Use this to enable supervision on internal layers: to be used in the supervised fine-tuning step, as makehebbian disables gradients.
    model = model.cuda()
    #model = DistributedDataParallel(model, device_ids=[args.local_rank])

    # Training Strategy
    criterion = segmentation_loss(args.loss, False).cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5*10**args.wd)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=args.warm_up_duration, after_scheduler=exp_lr_scheduler)

    # Train & Val
    since = time.time()
    count_iter = 0
    best_val_eval_list = [0 for i in range(4)]

    for epoch in range(args.num_epochs):

        count_iter += 1
        if (count_iter-1) % args.display_iter == 0:
            begin_time = time.time()

        #dataloaders['train'].sampler.set_epoch(epoch)
        model.train()

        train_loss = 0.0
        val_loss = 0.0

        #dist.barrier()
        for i, data in enumerate(dataloaders['train']):

            inputs_train = Variable(data['image'].cuda())
            mask_train = Variable(data['mask'].cuda())
            if mask_train.dim() == 3:
                mask_train = torch.unsqueeze(mask_train, dim=1)

            optimizer.zero_grad()
            outputs_train = model(inputs_train)
            torch.cuda.empty_cache()

            if args.deep_supervision:
                loss_train = 0
                for output_train in outputs_train:
                    loss_train += criterion(output_train, mask_train)
                loss_train /= len(outputs_train)
                outputs_train = outputs_train[0]
            else:
                loss_train = criterion(outputs_train, mask_train)

            loss_train.backward()
            for m in model.modules():
                if hasattr(m, 'local_update'): m.local_update()
            optimizer.step()
            train_loss += loss_train.item()

            if count_iter % args.display_iter == 0:
                if i == 0:
                    score_list_train = outputs_train
                    mask_list_train = mask_train
                # else:
                elif 0 < i <= num_batches['pretrain_unsup'] / 4:
                    score_list_train = torch.cat((score_list_train, outputs_train), dim=0)
                    mask_list_train = torch.cat((mask_list_train, mask_train), dim=0)

        scheduler_warmup.step()
        torch.cuda.empty_cache()

        if count_iter % args.display_iter == 0:

            #score_gather_list_train = [torch.zeros_like(score_list_train) for _ in range(ngpus_per_node)]
            #torch.distributed.all_gather(score_gather_list_train, score_list_train)
            #score_list_train = torch.cat(score_gather_list_train, dim=0)

            #mask_gather_list_train = [torch.zeros_like(mask_list_train) for _ in range(ngpus_per_node)]
            #torch.distributed.all_gather(mask_gather_list_train, mask_list_train)
            #mask_list_train = torch.cat(mask_gather_list_train, dim=0)

            #if rank == args.rank_index:
            torch.cuda.empty_cache()
            print('=' * print_num)
            print('| Epoch {}/{}'.format(epoch + 1, args.num_epochs).ljust(print_num_minus, ' '), '|')
            train_epoch_loss = print_train_loss_unsup(train_loss, num_batches, print_num, print_num_minus)
            train_eval_list, train_m_jc = print_train_eval_sup(cfg['NUM_CLASSES'], score_list_train, mask_list_train, print_num_minus)
            torch.cuda.empty_cache()

            with torch.no_grad():
                model.eval()

                for i, data in enumerate(dataloaders['val']):

                    # if 0 <= i <= num_batches['val']:

                    inputs_val = Variable(data['image'].cuda())
                    mask_val = Variable(data['mask'].cuda())
                    name_val = data['ID']

                    optimizer.zero_grad()
                    outputs_val = model(inputs_val)
                    torch.cuda.empty_cache()

                    if args.deep_supervision:
                        loss_val = 0
                        for output_val in outputs_val:
                            loss_val += criterion(output_val, mask_val)
                        loss_val /= len(outputs_val)
                        outputs_val = outputs_val[0]
                    else:
                        loss_val = criterion(outputs_val, mask_val)
                    val_loss += loss_val.item()

                    if i == 0:
                        score_list_val = outputs_val
                        mask_list_val = mask_val
                        name_list_val = name_val
                    else:
                        score_list_val = torch.cat((score_list_val, outputs_val), dim=0)
                        mask_list_val = torch.cat((mask_list_val, mask_val), dim=0)
                        name_list_val = np.append(name_list_val, name_val, axis=0)

                torch.cuda.empty_cache()
                #score_gather_list_val = [torch.zeros_like(score_list_val) for _ in range(ngpus_per_node)]
                #torch.distributed.all_gather(score_gather_list_val, score_list_val)
                #score_list_val = torch.cat(score_gather_list_val, dim=0)

                #mask_gather_list_val = [torch.zeros_like(mask_list_val) for _ in range(ngpus_per_node)]
                #torch.distributed.all_gather(mask_gather_list_val, mask_list_val)
                #mask_list_val = torch.cat(mask_gather_list_val, dim=0)

                #name_gather_list_val = [None for _ in range(ngpus_per_node)]
                #torch.distributed.all_gather_object(name_gather_list_val, name_list_val)
                #name_list_val = np.concatenate(name_gather_list_val, axis=0)
                #torch.cuda.empty_cache()

                #if rank == args.rank_index:
                val_epoch_loss = print_val_loss_sup(val_loss, num_batches, print_num, print_num_minus)
                val_eval_list, val_m_jc = print_val_eval_sup(cfg['NUM_CLASSES'], score_list_val, mask_list_val, print_num_minus)
                best_val_eval_list = save_val_best_sup_2d(cfg['NUM_CLASSES'], best_val_eval_list, model, score_list_val, name_list_val, val_eval_list, path_trained_models, path_seg_results, cfg['PALETTE'], args.network)
                torch.cuda.empty_cache()

                if args.vis:
                    draw_img = draw_pred_sup(cfg['NUM_CLASSES'], mask_train, mask_val, outputs_train, outputs_val, train_eval_list, val_eval_list)
                    visualization_sup(visdom, epoch+1, train_epoch_loss, train_m_jc, val_epoch_loss, val_m_jc)
                    visual_image_sup(visdom, draw_img[0], draw_img[1], draw_img[2], draw_img[3])

                print('-' * print_num)
                print('| Epoch Time: {:.4f}s'.format((time.time() - begin_time) / args.display_iter).ljust(print_num_minus, ' '), '|')
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()

    #if rank == args.rank_index:
    time_elapsed = time.time() - since
    m, s = divmod(time_elapsed, 60)
    h, m = divmod(m, 60)

    print('=' * print_num)
    print('| Training Completed In {:.0f}h {:.0f}mins {:.0f}s'.format(h, m, s).ljust(print_num_minus, ' '), '|')
    print('-' * print_num)
    print_best_sup(cfg['NUM_CLASSES'], best_val_eval_list, print_num_minus)
    print('=' * print_num)