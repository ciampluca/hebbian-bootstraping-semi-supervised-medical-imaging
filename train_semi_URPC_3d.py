import argparse
import time
import os
import numpy as np
import pandas as pd
import json

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchio as tio

from config.dataset_config.dataset_cfg import dataset_cfg
from config.warmup_config.warmup import GradualWarmupScheduler
from config.augmentation.online_aug import data_transform_3d
from loss.loss_function import segmentation_loss
from models.getnetwork import get_network
from dataload.dataset_3d import dataset_it

from hebb.makehebbian import makehebbian
from models.networks_3d.unet3d import init_weights as init_weights_unet3d
from models.networks_3d.vnet import init_weights as init_weights_vnet
from utils import save_snapshot, init_seeds, print_best_val_metrics, save_preds_3d, compute_val_epoch_loss_MT, evaluate_val_MT, compute_epoch_loss_XNet, evaluate_XNet

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--path_root_exp', default='./runs')
    parser.add_argument('--path_dataset', default='data/Atrial')
    parser.add_argument('--dataset_name', default='Atrial', help='Atrial')
    parser.add_argument('--input1', default='image')
    parser.add_argument('--regime', default=20, type=int, help="percentage of labeled data to be used")
    parser.add_argument('-b', '--batch_size', default=1, type=int)
    parser.add_argument('-e', '--num_epochs', default=200, type=int)
    parser.add_argument('-s', '--step_size', default=50, type=int)
    parser.add_argument('--optimizer', default="sgd", type=str, help="adam, sgd")
    parser.add_argument('-l', '--lr', default=0.1, type=float)
    parser.add_argument('-g', '--gamma', default=0.5, type=float)
    parser.add_argument('--patch_size', default=(96, 96, 80))
    parser.add_argument('--loss', default='dice', type=str)
    parser.add_argument('-w', '--warm_up_duration', default=20)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('-c', '--unsup_weight', default=50, type=float)
    parser.add_argument('-i', '--display_iter', default=1, type=int)
    parser.add_argument('--validate_iter', default=2, type=int)
    parser.add_argument('--queue_length', default=48, type=int)
    parser.add_argument('--samples_per_volume_train', default=4, type=int)
    parser.add_argument('--samples_per_volume_val', default=8, type=int)
    parser.add_argument('-n', '--network', default='unet3d_urpc', type=str)
    parser.add_argument('--debug', default=True)

    parser.add_argument('--load_hebbian_weights', default=None, type=str, help='path of hebbian pretrained weights')
    parser.add_argument('--hebbian-rule', default='swta_t', type=str, help='hebbian rules to be used')
    parser.add_argument('--hebb_inv_temp', default=1, type=int, help='hebbian temp')

    args = parser.parse_args()

     # set cuda device
    torch.cuda.set_device(args.device)
    init_seeds(args.seed)

    # load dataset config
    dataset_name = args.dataset_name
    cfg = dataset_cfg(dataset_name)

    print_num = 42 + (cfg['NUM_CLASSES'] - 3) * 7
    print_num_minus = print_num - 2
    print_num_half = int(print_num / 2 - 1)

    if isinstance(args.patch_size, str):
        args.patch_size = eval(args.patch_size)

    # create folders
    net_name = "unet" if args.network == "unet_urpc" else args.network
    if args.regime < 100:
        if args.load_hebbian_weights:
            path_run = os.path.join(args.path_root_exp, os.path.split(args.path_dataset)[1], "semi_sup", "h_urpc_{}_{}".format(net_name, args.hebbian_rule), "inv_temp-{}".format(args.hebb_inv_temp), "regime-{}".format(args.regime), "run-{}".format(args.seed))
        else:
            path_run = os.path.join(args.path_root_exp, os.path.split(args.path_dataset)[1], "semi_sup", "urpc_{}".format(net_name), "inv_temp-1", "regime-{}".format(args.regime), "run-{}".format(args.seed))
    else:
        path_run = os.path.join(args.path_root_exp, os.path.split(args.path_dataset)[1], "fully_sup", "urpc_{}".format(net_name), "inv_temp-1", "regime-{}".format(args.regime), "run-{}".format(args.seed))
    if not os.path.exists(path_run):
        os.makedirs(path_run)
    path_trained_models = os.path.join(os.path.join(path_run, "checkpoints"))
    if not os.path.exists(path_trained_models):
        os.makedirs(path_trained_models)  
    path_tensorboard = os.path.join(os.path.join(path_run, "runs"))
    if not os.path.exists(path_tensorboard):
        os.makedirs(path_tensorboard)
    path_seg_results = os.path.join(os.path.join(path_run, "val_seg_preds"))
    if not os.path.exists(path_seg_results):
        os.makedirs(path_seg_results)
    if args.debug:
        path_train_seg_results = os.path.join(os.path.join(path_run, "train_seg_preds"))
        if not os.path.exists(path_train_seg_results):
            os.makedirs(path_train_seg_results)

    # create tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(path_run, 'runs'))

    # save config to file
    with open(os.path.join(path_run, "config.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Dataset
    data_transform = data_transform_3d(cfg['NORMALIZE'])

    dataset_train_unsup = dataset_it(
        data_dir=args.path_dataset + '/train_unsup_' + args.unsup_mark,
        input1=args.input1,
        transform_1=data_transform['train'],
        queue_length=args.queue_length,
        samples_per_volume=args.samples_per_volume_train,
        patch_size=args.patch_size,
        num_workers=8,
        shuffle_subjects=True,
        shuffle_patches=True,
        sup=False,
        num_images=None
    )
    num_images_unsup = len(dataset_train_unsup.dataset_1)

    dataset_train_sup = dataset_it(
        data_dir=args.path_dataset + '/train_sup_' + args.sup_mark,
        input1=args.input1,
        transform_1=data_transform['train'],
        queue_length=args.queue_length,
        samples_per_volume=args.samples_per_volume_train,
        patch_size=args.patch_size,
        num_workers=8,
        shuffle_subjects=True,
        shuffle_patches=True,
        sup=True,
        num_images=num_images_unsup
    )
    dataset_val = dataset_it(
        data_dir=args.path_dataset + '/val',
        input1=args.input1,
        transform_1=data_transform['val'],
        queue_length=args.queue_length,
        samples_per_volume=args.samples_per_volume_val,
        patch_size=args.patch_size,
        num_workers=8,
        shuffle_subjects=False,
        shuffle_patches=False,
        sup=True,
        num_images=None
    )

    dataloaders = dict()
    dataloaders['train_sup'] = DataLoader(dataset_train_sup.queue_train_set_1, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)
    dataloaders['train_unsup'] = DataLoader(dataset_train_unsup.queue_train_set_1, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)
    dataloaders['val'] = DataLoader(dataset_val.queue_train_set_1, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)

    num_batches = {'train_sup': len(dataloaders['train_sup']), 'train_unsup': len(dataloaders['train_unsup']), 'val': len(dataloaders['val'])}

    # create models
    model = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'])

    # eventually load hebbian weights
    hebb_params, exclude, exclude_layer_names = None, None, None
    if args.load_hebbian_weights:
        print("Loading Hebbian pre-trained weights")
        state_dict = torch.load(args.load_hebbian_weights, map_location='cpu')
        hebb_params = state_dict['hebb_params']
        hebb_params['alpha'] = 0
        exclude = state_dict['excluded_layers']
        model = makehebbian(model, exclude=exclude, hebb_params=hebb_params)
        model.load_state_dict(state_dict['model'])

        exclude_layer_names = exclude
        if exclude is None: exclude = []
        exclude = [(n, m) for n, m in model.named_modules() if any([n == e for e in exclude])]
        exclude = [m for _, p in exclude for m in [*p.modules()]]

        if args.network == 'unet3d':
            init_weights = init_weights_unet3d
        elif args.network == 'vnet':
            init_weights = init_weights_vnet
        for m in exclude:
            init_weights(m, init_type='kaiming')

        for p in model.parameters():
            p.requires_grad = True

    model = model.cuda()




    # Training Strategy
    criterion = segmentation_loss(args.loss, False).cuda()
    kl_distance = nn.KLDivLoss(reduction='none')

    optimizer1 = optim.SGD(model1.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5 * 10 ** args.wd)
    exp_lr_scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=args.step_size, gamma=args.gamma)
    scheduler_warmup1 = GradualWarmupScheduler(optimizer1, multiplier=1.0, total_epoch=args.warm_up_duration, after_scheduler=exp_lr_scheduler1)

    # Train & Val
    since = time.time()
    count_iter = 0

    best_val_eval_list = [0 for i in range(4)]

    for epoch in range(args.num_epochs):

        count_iter += 1
        if (count_iter - 1) % args.display_iter == 0:
            begin_time = time.time()

        # dataloaders['train_sup'].sampler.set_epoch(epoch)
        # dataloaders['train_unsup'].sampler.set_epoch(epoch)
        model1.train()

        train_loss_sup_1 = 0.0
        train_loss_unsup = 0.0
        train_loss = 0.0

        val_loss_sup_1 = 0.0

        unsup_weight = args.unsup_weight * (epoch + 1) / args.num_epochs

        #dist.barrier()

        dataset_train_sup = iter(dataloaders['train_sup'])
        dataset_train_unsup = iter(dataloaders['train_unsup'])

        for i in range(num_batches['train_sup']):

            unsup_index = next(dataset_train_unsup)
            img_train_unsup_1 = Variable(unsup_index['image'][tio.DATA].cuda())

            optimizer1.zero_grad()

            pred_train_unsup1, pred_train_unsup2, pred_train_unsup3, pred_train_unsup4 = model1(img_train_unsup_1)
            pred_train_unsup1 = torch.softmax(pred_train_unsup1, 1)
            pred_train_unsup2 = torch.softmax(pred_train_unsup2, 1)
            pred_train_unsup3 = torch.softmax(pred_train_unsup3, 1)
            pred_train_unsup4 = torch.softmax(pred_train_unsup4, 1)

            preds = (pred_train_unsup1 + pred_train_unsup2 + pred_train_unsup3 + pred_train_unsup4) / 4

            variance_aux1 = torch.sum(kl_distance(torch.log(pred_train_unsup1), preds), dim=1, keepdim=True)
            exp_variance_aux1 = torch.exp(-variance_aux1)

            variance_aux2 = torch.sum(kl_distance(torch.log(pred_train_unsup2), preds), dim=1, keepdim=True)
            exp_variance_aux2 = torch.exp(-variance_aux2)

            variance_aux3 = torch.sum(kl_distance(torch.log(pred_train_unsup3), preds), dim=1, keepdim=True)
            exp_variance_aux3 = torch.exp(-variance_aux3)

            variance_aux4 = torch.sum(kl_distance(torch.log(pred_train_unsup4), preds), dim=1, keepdim=True)
            exp_variance_aux4 = torch.exp(-variance_aux4)

            consistency_dist_aux1 = (preds - pred_train_unsup1) ** 2
            consistency_loss_aux1 = torch.mean(consistency_dist_aux1 * exp_variance_aux1) / (torch.mean(exp_variance_aux1) + 1e-8) + torch.mean(variance_aux1)

            consistency_dist_aux2 = (preds - pred_train_unsup2) ** 2
            consistency_loss_aux2 = torch.mean(consistency_dist_aux2 * exp_variance_aux2) / (torch.mean(exp_variance_aux2) + 1e-8) + torch.mean(variance_aux2)

            consistency_dist_aux3 = (preds - pred_train_unsup3) ** 2
            consistency_loss_aux3 = torch.mean(consistency_dist_aux3 * exp_variance_aux3) / (torch.mean(exp_variance_aux3) + 1e-8) + torch.mean(variance_aux3)

            consistency_dist_aux4 = (preds - pred_train_unsup4) ** 2
            consistency_loss_aux4 = torch.mean(consistency_dist_aux4 * exp_variance_aux4) / (torch.mean(exp_variance_aux4) + 1e-8) + torch.mean(variance_aux4)

            loss_train_unsup = (consistency_loss_aux1 + consistency_loss_aux2 + consistency_loss_aux3 + consistency_loss_aux4) / 4

            loss_train_unsup = loss_train_unsup * unsup_weight
            loss_train_unsup.backward(retain_graph=True)
            torch.cuda.empty_cache()

            sup_index = next(dataset_train_sup)
            img_train_sup_1 = Variable(sup_index['image'][tio.DATA].cuda())
            mask_train_sup = Variable(sup_index['mask'][tio.DATA].squeeze(1).long().cuda())

            pred_train_sup1, pred_train_sup2, pred_train_sup3, pred_train_sup4 = model1(img_train_sup_1)

            if count_iter % args.display_iter == 0:
                if i == 0:
                    score_list_train1 = pred_train_sup1
                    mask_list_train = mask_train_sup
                # else:
                elif 0 < i <= num_batches['train_sup'] / 32:
                    score_list_train1 = torch.cat((score_list_train1, pred_train_sup1), dim=0)
                    mask_list_train = torch.cat((mask_list_train, mask_train_sup), dim=0)

            loss_train_sup1 = (criterion(pred_train_sup1, mask_train_sup)+criterion(pred_train_sup2, mask_train_sup)+criterion(pred_train_sup3, mask_train_sup)+criterion(pred_train_sup4, mask_train_sup)) / 4
            loss_train_sup = loss_train_sup1

            loss_train_sup.backward()
            optimizer1.step()
            torch.cuda.empty_cache()

            loss_train = loss_train_unsup + loss_train_sup
            train_loss_unsup += loss_train_unsup.item()
            train_loss_sup_1 += loss_train_sup1.item()
            train_loss += loss_train.item()

        scheduler_warmup1.step()
        torch.cuda.empty_cache()

        if count_iter % args.display_iter == 0:

            # score_gather_list_train1 = [torch.zeros_like(score_list_train1) for _ in range(ngpus_per_node)]
            # torch.distributed.all_gather(score_gather_list_train1, score_list_train1)
            # score_list_train1 = torch.cat(score_gather_list_train1, dim=0)

            # mask_gather_list_train = [torch.zeros_like(mask_list_train) for _ in range(ngpus_per_node)]
            # torch.distributed.all_gather(mask_gather_list_train, mask_list_train)
            # mask_list_train = torch.cat(mask_gather_list_train, dim=0)

            #if rank == args.rank_index:
            torch.cuda.empty_cache()
            print('=' * print_num)
            print('| Epoch {}/{}'.format(epoch + 1, args.num_epochs).ljust(print_num_minus, ' '), '|')
            train_epoch_loss_sup_1, train_epoch_loss_cps, train_epoch_loss = print_train_loss_EM(train_loss_sup_1, train_loss_unsup, train_loss, num_batches, print_num, print_num_minus)
            train_eval_list_1, train_m_jc_1 = print_train_eval_sup(cfg['NUM_CLASSES'], score_list_train1, mask_list_train, print_num_minus)
            torch.cuda.empty_cache()

            with torch.no_grad():
                model1.eval()

                for i, data in enumerate(dataloaders['val']):

                    # if 0 <= i <= num_batches['val']:

                    inputs_val_1 = Variable(data['image'][tio.DATA].cuda())
                    mask_val = Variable(data['mask'][tio.DATA].squeeze(1).long().cuda())

                    optimizer1.zero_grad()
                    outputs_val_1, outputs_val_2, outputs_val_3, outputs_val_4 = model1(inputs_val_1)
                    torch.cuda.empty_cache()

                    if i == 0:
                        score_list_val_1 = outputs_val_1
                        mask_list_val = mask_val
                    else:
                        score_list_val_1 = torch.cat((score_list_val_1, outputs_val_1), dim=0)
                        mask_list_val = torch.cat((mask_list_val, mask_val), dim=0)

                    loss_val_sup_1 = criterion(outputs_val_1, mask_val)
                    val_loss_sup_1 += loss_val_sup_1.item()

                torch.cuda.empty_cache()
                # score_gather_list_val_1 = [torch.zeros_like(score_list_val_1) for _ in range(ngpus_per_node)]
                # torch.distributed.all_gather(score_gather_list_val_1, score_list_val_1)
                # score_list_val_1 = torch.cat(score_gather_list_val_1, dim=0)

                # mask_gather_list_val = [torch.zeros_like(mask_list_val) for _ in range(ngpus_per_node)]
                # torch.distributed.all_gather(mask_gather_list_val, mask_list_val)
                # mask_list_val = torch.cat(mask_gather_list_val, dim=0)
                torch.cuda.empty_cache()

                #if rank == args.rank_index:
                val_epoch_loss_sup_1 = print_val_loss_sup(val_loss_sup_1, num_batches, print_num, print_num_minus)
                val_eval_list_1, val_m_jc_1 = print_val_eval_sup(cfg['NUM_CLASSES'], score_list_val_1, mask_list_val, print_num_minus)
                best_val_eval_list = save_val_best_sup_3d(cfg['NUM_CLASSES'], best_val_eval_list, model1, score_list_val_1, mask_list_val, val_eval_list_1, path_trained_models, path_seg_results, path_mask_results, 'URPC', cfg['FORMAT'])
                torch.cuda.empty_cache()

                if args.vis:
                    visualization_EM(visdom, epoch + 1, train_epoch_loss, train_epoch_loss_sup_1, train_epoch_loss_cps, train_m_jc_1, val_epoch_loss_sup_1, val_m_jc_1)

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