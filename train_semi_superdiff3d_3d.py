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
from utils import save_snapshot, init_seeds, print_best_val_metrics, save_preds, save_preds_3d, compute_val_epoch_loss_MT, evaluate_val_MT, compute_epoch_loss_XNet, evaluate_XNet

from models.networks_3d.unet3d_ddpm import SuperDiffusion

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
    parser.add_argument('-n', '--network', default='unet3d_ddpm', type=str)
    parser.add_argument('--debug', default=True)
    parser.add_argument('--init_weights', default='kaiming', type=str)
    parser.add_argument('--load_weights', default=None, type=str, help='path of pretrained weights')
    parser.add_argument('--timestamp_diffusion', default=1000, type=int, help='diffusion timesteps')
    parser.add_argument('--diff_rounds', default=3, type=int, help='number of generation iterations from which pseudo-labels are generated')
    parser.add_argument('--diff_weight', default=1, type=float, help='coefficient of the loss part determined from the generation process')

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
    if args.regime < 100:
        if args.load_weights:
            path_run = os.path.join(args.path_root_exp, os.path.split(args.path_dataset)[1], "semi_sup", "superdiff_{}".format(args.network), "diff_rounds-{}".format(args.diff_rounds), "regime-{}".format(args.regime), "run-{}".format(args.seed))
        else:
            if args.init_weights != "kaiming":
                path_run = os.path.join(args.path_root_exp, os.path.split(args.path_dataset)[1], "semi_sup", "{}_superdiff_{}".format(args.init_weights, args.network), "diff_rounds-{}".format(args.diff_rounds), "regime-{}".format(args.regime), "run-{}".format(args.seed))
            else:
                path_run = os.path.join(args.path_root_exp, os.path.split(args.path_dataset)[1], "semi_sup", "superdiff_{}".format(args.network), "diff_rounds-{}".format(args.diff_rounds), "regime-{}".format(args.regime), "run-{}".format(args.seed))
    else:
        path_run = os.path.join(args.path_root_exp, os.path.split(args.path_dataset)[1], "fully_sup", "superdiff_{}".format(args.network), "diff_rounds-{}".format(args.diff_rounds), "regime-{}".format(args.regime), "run-{}".format(args.seed))
    if not os.path.exists(path_run):
        os.makedirs(path_run)
    path_trained_models = os.path.join(os.path.join(path_run, "checkpoints"))
    if not os.path.exists(path_trained_models):
        os.makedirs(path_trained_models)
    path_trained_models2 = os.path.join(os.path.join(path_run, "checkpoints2"))
    if not os.path.exists(path_trained_models2):
        os.makedirs(path_trained_models2)    
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
        path_train_seg_results2 = os.path.join(os.path.join(path_run, "train_seg_preds2"))
        if not os.path.exists(path_train_seg_results2):
            os.makedirs(path_train_seg_results2)   

    # create tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(path_run, 'runs'))

    # save config to file
    with open(os.path.join(path_run, "config.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # data loading
    data_transform = data_transform_3d(cfg['NORMALIZE'])

    dataset_train_sup = dataset_it(
        data_dir=args.path_dataset + '/train',
        input1=args.input1,
        transform_1=data_transform['train'],
        queue_length=args.queue_length,
        samples_per_volume=args.samples_per_volume_train,
        patch_size=args.patch_size,
        num_workers=2,
        shuffle_subjects=True,
        shuffle_patches=True,
        sup=True,
        regime=args.regime,
        seed=args.seed,
        num_classes=cfg['NUM_CLASSES'],
    )
    dataset_train_unsup = dataset_it(
        data_dir=args.path_dataset + '/train',
        input1=args.input1,
        transform_1=data_transform['train'],
        queue_length=args.queue_length,
        samples_per_volume=args.samples_per_volume_train,
        patch_size=args.patch_size,
        num_workers=2,
        shuffle_subjects=True,
        shuffle_patches=True,
        sup=False,
        regime=args.regime,
        seed=args.seed,
        num_classes=cfg['NUM_CLASSES'],
    )
    dataset_val = dataset_it(
        data_dir=args.path_dataset + '/val',
        input1=args.input1,
        transform_1=data_transform['val'],
        queue_length=args.queue_length,
        samples_per_volume=args.samples_per_volume_val,
        patch_size=args.patch_size,
        num_workers=2,
        shuffle_subjects=False,
        shuffle_patches=False,
        sup=True,
        num_classes=cfg['NUM_CLASSES'],
    )

    dataloaders = dict()
    dataloaders['train_sup'] = DataLoader(dataset_train_sup.queue_train_set_1, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)
    dataloaders['train_unsup'] = DataLoader(dataset_train_unsup.queue_train_set_1, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)
    dataloaders['val'] = DataLoader(dataset_val.queue_train_set_1, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)

    num_batches = {'train_sup': len(dataloaders['train_sup']), 'train_unsup': len(dataloaders['train_unsup']), 'val': len(dataloaders['val'])}

    # create models
    model1 = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'], args.init_weights, timestamp_diffusion=args.timestamp_diffusion)
    model2 = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'], args.init_weights, timestamp_diffusion=args.timestamp_diffusion)

    # define criterion, optimizer, and scheduler
    diffusion1 = SuperDiffusion(
        model1.net,
        image_size = args.patch_size[0],
        timesteps = args.timestamp_diffusion,    # number of steps
        objective='pred_noise', #'pred_x0', #'pred_v', #'pred_noise', #
    )
    diffusion1 = diffusion1.to(args.device)
    diffusion_seg1 = SuperDiffusion(
        model1.net_seg,
        image_size = args.patch_size[0],
        timesteps = args.timestamp_diffusion,    # number of steps
        objective='pred_x0', #'pred_x0', #'pred_v', #'pred_noise', #
    )
    diffusion_seg1 = diffusion_seg1.to(args.device)
    # define criterion, optimizer, and scheduler
    diffusion2 = SuperDiffusion(
        model2.net,
        image_size = args.patch_size[0],
        timesteps = args.timestamp_diffusion,    # number of steps
        objective='pred_noise', #'pred_x0', #'pred_v', #'pred_noise', #
    )
    diffusion2 = diffusion2.to(args.device)
    diffusion_seg2 = SuperDiffusion(
        model2.net_seg,
        image_size = args.patch_size[0],
        timesteps = args.timestamp_diffusion,    # number of steps
        objective='pred_x0', #'pred_x0', #'pred_v', #'pred_noise', #
    )
    diffusion_seg2 = diffusion_seg2.to(args.device)
    
    if args.load_weights:
        print("Loading pre-trained weights")
        state_dict = torch.load(args.load_weights, map_location='cpu')
        model2.load_state_dict(state_dict['model'])
    model1 = model1.cuda()
    model2 = model2.cuda()

    # define criterion, optimizer, and scheduler
    criterion = segmentation_loss(args.loss, False).cuda()

    if args.optimizer == "adam":
        optimizer1 = optim.Adam(model1.parameters(), lr=args.lr)
        optimizer2 = optim.Adam(model2.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer1 = optim.SGD(model1.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5*10**args.wd)
        optimizer2 = optim.SGD(model2.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5*10**args.wd)
    else:
        print("Optimizer not implemented")

    exp_lr_scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=args.step_size, gamma=args.gamma)
    scheduler_warmup1 = GradualWarmupScheduler(optimizer1, multiplier=1.0, total_epoch=args.warm_up_duration, after_scheduler=exp_lr_scheduler1)
    exp_lr_scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=args.step_size, gamma=args.gamma)
    scheduler_warmup2 = GradualWarmupScheduler(optimizer2, multiplier=1.0, total_epoch=args.warm_up_duration, after_scheduler=exp_lr_scheduler1)

    # training loop
    since = time.time()
    count_iter = 0
    best_val_eval_list = [0 for i in range(4)]
    train_metrics, val_metrics = [], []

    for epoch in range(args.num_epochs):

        count_iter += 1
        if (count_iter-1) % args.display_iter == 0:
            begin_time = time.time()

        model1.train()
        model2.train()

        train_loss_sup_1 = 0.0
        train_loss_sup_2 = 0.0
        train_loss_unsup = 0.0
        train_loss = 0.0
        val_loss_sup_1 = 0.0
        val_loss_sup_2 = 0.0

        unsup_weight = args.unsup_weight * (epoch + 1) / args.num_epochs
        diff_weight = args.diff_weight
        diff_rounds = args.diff_rounds

        dataset_train_sup = iter(dataloaders['train_sup'])
        dataset_train_unsup = iter(dataloaders['train_unsup'])

        for i in range(num_batches['train_sup']):
            unsup_index = next(dataset_train_unsup)
            img_train_unsup = Variable(unsup_index['image'][tio.DATA].cuda())
            slice_idx = img_train_unsup.shape[-1] // 2
            #img_train_unsup = img_train_unsup[:, :, :, :, slice_idx]

            sup_index = next(dataset_train_sup)
            img_train_sup = Variable(sup_index['image'][tio.DATA].cuda())
            mask_train_sup = Variable(sup_index['mask'][tio.DATA].squeeze(1).long().cuda())
            slice_idx = img_train_sup.shape[-1] // 2
            #img_train_sup, mask_train_sup = img_train_sup[:, :, :, :, slice_idx], mask_train_sup[:, :, :, slice_idx]
            name_train = sup_index['ID']
            affine_train = sup_index['image']['affine']

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            img_train_unsup = img_train_unsup[:img_train_sup.shape[0]]
            zero_mask = torch.zeros((img_train_unsup.shape[0], *mask_train_sup.shape[1:]), dtype=mask_train_sup.dtype, device=mask_train_sup.device)
            _, pred_train_unsup2 = diffusion_seg2(img_train_unsup, zero_mask, conditioner='img')
            _, pred_train_unsup1 = diffusion_seg1(img_train_unsup, zero_mask, conditioner='img')
            loss_reconstr_unsup, loss_train_unsup_pseudo = 0., 0.
            pred_train_unsup_pseudo = pred_train_unsup2
            for j in range(diff_rounds):
                pred_train_unsup_pseudo = pred_train_unsup2
                loss_reconstr_unsup_pseudo, reconstr_unsup_pseudo = diffusion2(img_train_unsup, pred_train_unsup_pseudo.clone().detach(), conditioner='target')
                _, pred_train_unsup_pseudo = diffusion_seg2(reconstr_unsup_pseudo.clone().detach(), zero_mask, conditioner='img')
                loss_reconstr_unsup += loss_reconstr_unsup_pseudo
                loss_train_unsup_pseudo += criterion(pred_train_unsup_pseudo, torch.argmax(pred_train_unsup1, dim=1))
            loss_train_unsup = criterion(pred_train_unsup1, torch.argmax(pred_train_unsup2, dim=1)) + criterion(pred_train_unsup2, torch.argmax(pred_train_unsup1, dim=1)) \
                             + (diff_weight/diff_rounds) * loss_train_unsup_pseudo + (diff_weight/diff_rounds) * loss_reconstr_unsup
            loss_train_unsup = 1 * loss_train_unsup * unsup_weight
            loss_train_unsup.backward()

            _, pred_train_sup2 = diffusion_seg2(img_train_sup, torch.zeros_like(mask_train_sup), conditioner='img')
            _, pred_train_sup1 = diffusion_seg1(img_train_sup, torch.zeros_like(pred_train_sup2), conditioner='img')
            loss_reconstr_sup, loss_train_sup_pseudo = 0., 0.
            pred_train_sup_pseudo = pred_train_sup2
            for j in range(diff_rounds):
                pred_train_sup_pseudo = pred_train_sup2
                loss_reconstr_sup_pseudo, reconstr_sup_pseudo = diffusion2(img_train_sup, pred_train_sup_pseudo.clone().detach(), conditioner='target')
                _, pred_train_sup_pseudo = diffusion_seg2(reconstr_sup_pseudo.clone().detach(), torch.zeros_like(mask_train_sup), conditioner='img')
                loss_reconstr_sup += loss_reconstr_sup_pseudo
                loss_train_sup_pseudo += criterion(pred_train_sup_pseudo, mask_train_sup)
            loss_train_sup = criterion(pred_train_sup1, mask_train_sup) + criterion(pred_train_sup2, mask_train_sup) \
                           + (diff_weight/diff_rounds) * loss_train_sup_pseudo + (diff_weight/diff_rounds) * loss_reconstr_sup
            loss_train_sup.backward()

            optimizer1.step()
            optimizer2.step()

            if count_iter % args.display_iter == 0:
                if i == 0:
                    score_list_train1 = pred_train_sup1
                    score_list_train2 = pred_train_sup2
                    mask_list_train = mask_train_sup
                    name_list_train = name_train
                    affine_list_train = affine_train
                else:
                    score_list_train1 = torch.cat((score_list_train1, score_list_train1[-1:]), dim=0) #Dummy output to save memory #torch.cat((score_list_train1, pred_train_sup1), dim=0)
                    score_list_train2 = torch.cat((score_list_train2, score_list_train2[-1:]), dim=0) #torch.cat((score_list_train2, pred_train_sup2), dim=0)
                    mask_list_train = torch.cat((mask_list_train, mask_list_train[-1:]), dim=0) #torch.cat((mask_list_train, mask_train_sup), dim=0)
                    name_list_train = np.append(name_list_train, name_train, axis=0)
                    affine_list_train = torch.cat((affine_list_train, affine_train), dim=0)

            loss_train = loss_train_unsup + loss_train_sup
            train_loss_unsup += loss_train_unsup.item()
            train_loss_sup_1 += loss_train_sup.item()
            train_loss_sup_2 += loss_train_sup.item()
            train_loss += loss_train.item()

        scheduler_warmup1.step()
        scheduler_warmup2.step()

        if count_iter % args.display_iter == 0:
            print('=' * print_num)
            print('| Epoch {}/{}'.format(epoch + 1, args.num_epochs).ljust(print_num_minus, ' '), '|')


        if count_iter % args.display_iter == 0:
            train_epoch_loss_sup1, train_epoch_loss_sup2, train_epoch_loss_unsup, train_epoch_loss = compute_epoch_loss_XNet(train_loss_sup_1, train_loss_sup_2, train_loss_unsup, train_loss, num_batches, print_num, print_num_half, print_num_minus)
            train_eval_list1, train_eval_list2 = evaluate_XNet(cfg['NUM_CLASSES'], score_list_train1, score_list_train2, mask_list_train, print_num_minus)
            if args.debug:
                ext = name_list_train[0].rsplit(".", 1)[1]
                name_list_train = [name.rsplit(".", 1)[0] for name in name_list_train]
                name_list_train = [a if not (s:=sum(j == a for j in name_list_train[:i])) else f'{a}-{s+1}'
                    for i, a in enumerate(name_list_train)]
                name_list_train = [name + ".{}".format(ext) for name in name_list_train]
                save_preds_3d(score_list_train1, train_eval_list1[0], name_list_train, path_train_seg_results, affine_list_train, num_classes=cfg['NUM_CLASSES'])
                save_preds_3d(score_list_train2, train_eval_list2[0], name_list_train, path_train_seg_results2, affine_list_train, num_classes=cfg['NUM_CLASSES'])
                #save_preds(score_list_train1, train_eval_list1[0], name_list_train, path_train_seg_results, cfg['PALETTE'])
                #save_preds(score_list_train2, train_eval_list2[0], name_list_train, path_train_seg_results2, cfg['PALETTE'])

            # saving metrics to tensorboard writer
            writer.add_scalar('train/segm_loss', train_epoch_loss_sup1, count_iter)
            writer.add_scalar('train/segm_loss2', train_epoch_loss_sup2, count_iter)
            writer.add_scalar('train/unsup_loss', train_epoch_loss_unsup, count_iter)
            writer.add_scalar('train/total_loss', train_epoch_loss, count_iter)
            writer.add_scalar('train/lr', optimizer1.param_groups[0]['lr'], count_iter)
            writer.add_scalar('train/lr2', optimizer2.param_groups[0]['lr'], count_iter)
            writer.add_scalar('train/DC', train_eval_list1[2], count_iter)
            writer.add_scalar('train/JI', train_eval_list1[1], count_iter)
            writer.add_scalar('train/DC2', train_eval_list2[2], count_iter)
            writer.add_scalar('train/JI2', train_eval_list2[1], count_iter)
            writer.add_scalar('train/lambda_unsup', unsup_weight, count_iter)
            if cfg['NUM_CLASSES'] == 2:
                writer.add_scalar('train/thresh', train_eval_list1[0], count_iter)
                writer.add_scalar('train/thresh2', train_eval_list2[0], count_iter)

            # saving metrics to list
            train_metrics.append({
                'epoch': count_iter,
                'segm/loss': train_epoch_loss_sup1,
                'segm/loss2': train_epoch_loss_sup2,
                'segm/unsup_loss': train_epoch_loss_unsup,
                'segm/total_loss': train_epoch_loss,
                'segm/dice': train_eval_list1[2],
                'segm/jaccard': train_eval_list1[1],
                'lr': optimizer1.param_groups[0]['lr'],
                'thresh': train_eval_list1[0],
                'segm/dice2': train_eval_list2[2],
                'segm/jaccard2': train_eval_list2[1],
                'lr2': optimizer2.param_groups[0]['lr'],
                'thresh2': train_eval_list2[0],
            })

        if count_iter % args.validate_iter == 0:
            metrics_debug = []

            with torch.no_grad():
                model1.eval()
                model2.eval()

                for i, data in enumerate(dataloaders['val']):
                    inputs_val = Variable(data['image'][tio.DATA].cuda())
                    mask_val = Variable(data['mask'][tio.DATA].squeeze(1).long().cuda())
                    slice_idx = inputs_val.shape[-1] // 2
                    #inputs_val, mask_val = inputs_val[:, :, :, :, slice_idx], mask_val[:, :, :, slice_idx]
                    name_val = data['ID']
                    affine_val = data['image']['affine']

                    optimizer1.zero_grad()
                    optimizer2.zero_grad()

                    _, outputs_val1 = diffusion_seg1(inputs_val, torch.zeros_like(mask_val), conditioner='img')
                    _, outputs_val2 = diffusion_seg2(inputs_val, torch.zeros_like(mask_val), conditioner='img')
                    
                    loss_val_sup_1 = criterion(outputs_val1, mask_val)
                    loss_val_sup_2 = criterion(outputs_val2, mask_val)
                    val_loss_sup_1 += loss_val_sup_1.item()
                    val_loss_sup_2 += loss_val_sup_2.item()

                    if i == 0:
                        score_list_val1 = outputs_val1
                        score_list_val2 = outputs_val2
                        mask_list_val = mask_val
                        name_list_val = name_val
                        affine_list_val = affine_val
                    else:
                        score_list_val1 = torch.cat((score_list_val1, outputs_val1), dim=0)
                        score_list_val2 = torch.cat((score_list_val2, outputs_val2), dim=0)
                        mask_list_val = torch.cat((mask_list_val, mask_val), dim=0)
                        name_list_val = np.append(name_list_val, name_val, axis=0)
                        affine_list_val = torch.cat((affine_list_val, affine_val), dim=0)

                    # TODO save val metrics for single images
                    if args.debug:
                        pass

                # TODO save val metrics for single images
                if args.debug:
                    pass

                val_epoch_loss1, val_epoch_loss2 = compute_val_epoch_loss_MT(val_loss_sup_1, val_loss_sup_2, num_batches, print_num, print_num_half)
                val_eval_list1, val_eval_list2 = evaluate_val_MT(cfg['NUM_CLASSES'], score_list_val1, score_list_val2, mask_list_val, print_num_half)

                # check if best model (in terms of JI) and eventually save it
                best = 0
                if val_eval_list2[1] > val_eval_list1[1]:
                    if val_eval_list2[1] > best_val_eval_list[1]:
                        best_val_eval_list = val_eval_list2
                        best_score_list_val = score_list_val2
                        best_model = model2
                        best = 1
                elif val_eval_list1[1] >= val_eval_list2[1]:
                    if val_eval_list1[1] > best_val_eval_list[1]:
                        best_val_eval_list = val_eval_list1
                        best_model = model1
                        best_score_list_val = score_list_val1
                        best = 1

                if best:
                    save_snapshot(best_model, path_trained_models, threshold=best_val_eval_list[0], save_best=True)
                    # save val best preds
                    ext = name_list_val[0].rsplit(".", 1)[1]
                    name_list_val = [name.rsplit(".", 1)[0] for name in name_list_val]
                    name_list_val = [a if not (s:=sum(j == a for j in name_list_val[:i])) else f'{a}-{s+1}'
                        for i, a in enumerate(name_list_val)]
                    name_list_val = [name + ".{}".format(ext) for name in name_list_val]
                    save_preds_3d(best_score_list_val, best_val_eval_list[0], name_list_val, os.path.join(path_seg_results, 'best_model'), affine_list_val, num_classes=cfg['NUM_CLASSES'])
                    #save_preds(best_score_list_val, best_val_eval_list[0], name_list_val, os.path.join(path_seg_results, 'best_model'), cfg['PALETTE'])
                
                # saving metrics to tensorboard writer
                writer.add_scalar('val/segm_loss', val_epoch_loss1, count_iter)
                writer.add_scalar('val/DC', val_eval_list1[2], count_iter)
                writer.add_scalar('val/JI', val_eval_list1[1], count_iter)
                writer.add_scalar('val/segm_loss2', val_epoch_loss2, count_iter)
                writer.add_scalar('val/DC2', val_eval_list2[2], count_iter)
                writer.add_scalar('val/JI2', val_eval_list2[1], count_iter)
                if cfg['NUM_CLASSES'] == 2:
                    writer.add_scalar('val/thresh', val_eval_list1[0], count_iter)
                    writer.add_scalar('val/thresh2', val_eval_list2[0], count_iter)

                # saving metrics to list
                val_metrics.append({
                    'epoch': count_iter,
                    'segm/loss': val_epoch_loss1,
                    'segm/dice': val_eval_list1[2],
                    'segm/jaccard': val_eval_list1[1],
                    'thresh2': val_eval_list1[0],
                    'segm/loss2': val_epoch_loss2,
                    'segm/dice2': val_eval_list2[2],
                    'segm/jaccard2': val_eval_list2[1],
                    'thresh2': val_eval_list2[0],
                })

                print('-' * print_num)
                print('| Epoch Time: {:.4f}s'.format((time.time() - begin_time) / args.display_iter).ljust(print_num_minus, ' '), '|')

    # save val last preds
    ext = name_list_val[0].rsplit(".", 1)[1]
    name_list_val = [name.rsplit(".", 1)[0] for name in name_list_val]
    name_list_val = [a if not (s:=sum(j == a for j in name_list_val[:i])) else f'{a}-{s+1}'
        for i, a in enumerate(name_list_val)]
    name_list_val = [name + ".{}".format(ext) for name in name_list_val]
    save_preds_3d(score_list_val1, val_eval_list1[0], name_list_val, os.path.join(path_seg_results, 'last_model'), affine_list_val, num_classes=cfg['NUM_CLASSES'])
    save_preds_3d(score_list_val2, val_eval_list2[0], name_list_val, os.path.join(path_seg_results, 'last_model2'), affine_list_val, num_classes=cfg['NUM_CLASSES'])
    #save_preds(score_list_val1, val_eval_list1[0], name_list_val, os.path.join(path_seg_results, 'last_model'), cfg['PALETTE'])
    #save_preds(score_list_val2, val_eval_list2[0], name_list_val, os.path.join(path_seg_results, 'last_model2'), cfg['PALETTE'])

    # save last model
    save_snapshot(model1, path_trained_models, threshold=val_eval_list1[0], save_best=False)
    save_snapshot(model2, path_trained_models2, threshold=val_eval_list2[0], save_best=False)

    # save train and val metrics in csv file
    train_metrics = pd.DataFrame(train_metrics)
    train_metrics.to_csv(os.path.join(path_run, 'train_log.csv'), index=False)
    val_metrics = pd.DataFrame(val_metrics)
    val_metrics.to_csv(os.path.join(path_run, 'val_log.csv'), index=False)

    time_elapsed = time.time() - since
    m, s = divmod(time_elapsed, 60)
    h, m = divmod(m, 60)

    time_elapsed = time.time() - since
    m, s = divmod(time_elapsed, 60)
    h, m = divmod(m, 60)

    print('=' * print_num)
    print('| Training Completed In {:.0f}h {:.0f}mins {:.0f}s'.format(h, m, s).ljust(print_num_minus, ' '), '|')
    print('-' * print_num)
    print_best_val_metrics(cfg['NUM_CLASSES'], best_val_eval_list, print_num_minus)
    print('=' * print_num)
