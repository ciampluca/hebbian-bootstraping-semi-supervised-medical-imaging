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
from utils import save_snapshot, init_seeds, compute_epoch_loss, evaluate, print_best_val_metrics, save_preds_3d, superpix_segment_3d, elbo_metric

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--path_root_exp', default='./runs')
    parser.add_argument('--path_dataset', default='data/GlaS')
    parser.add_argument('--dataset_name', default='Atrial', help='Atrial')
    parser.add_argument('--input1', default='image')
    parser.add_argument('-b', '--batch_size', default=2, type=int)
    parser.add_argument('-e', '--num_epochs', default=200, type=int)
    parser.add_argument('-s', '--step_size', default=50, type=int)
    parser.add_argument('--optimizer', default="adam", type=str, help="adam, sgd")
    parser.add_argument('-l', '--lr', default=0.5, type=float)
    parser.add_argument('-g', '--gamma', default=0.5, type=float)
    parser.add_argument('--patch_size', default=(96, 96, 80))
    parser.add_argument('--loss', default='dice', type=str)
    parser.add_argument('-w', '--warm_up_duration', default=20)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('-i', '--display_iter', default=1, type=int)
    parser.add_argument('--validate_iter', default=2, type=int)
    parser.add_argument('--threshold', default=None,  type=float)
    parser.add_argument('--thr_interval', default=0.02,  type=float)
    parser.add_argument('--queue_length', default=48, type=int)
    parser.add_argument('--samples_per_volume_train', default=4, type=int)
    parser.add_argument('--samples_per_volume_val', default=8, type=int)
    parser.add_argument('-n', '--network', default='unet3d', type=str)
    parser.add_argument('--debug', default=True)
    
    args = parser.parse_args()

    # set cuda device
    torch.cuda.set_device(args.device)
    init_seeds(args.seed)

    # load dataset config
    dataset_name = args.dataset_name
    cfg = dataset_cfg(dataset_name)

    print_num = 42 + (cfg['NUM_CLASSES'] - 3) * 7
    print_num_minus = print_num - 2

    if isinstance(args.patch_size, str):
        args.patch_size = eval(args.patch_size)

    # create folders
    path_run = os.path.join(args.path_root_exp, os.path.split(args.path_dataset)[1], "vae_unsup", "{}".format(args.network), "inv_temp-1", "regime-100", "run-{}".format(args.seed))
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

    # data loading
    data_transform = data_transform_3d(cfg['NORMALIZE'])

    dataset_train = dataset_it(
        data_dir=args.path_dataset + '/train',
        input1=args.input1,
        transform_1=data_transform['train'],
        queue_length=args.queue_length,
        samples_per_volume=args.samples_per_volume_train,
        patch_size=args.patch_size,
        num_workers=8,
        shuffle_subjects=True,
        shuffle_patches=True,
        sup=True,
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
        num_workers=8,
        shuffle_subjects=False,
        shuffle_patches=False,
        sup=True,
        num_classes=cfg['NUM_CLASSES'],
    )

    dataloaders = dict()
    dataloaders['train'] = DataLoader(dataset_train.queue_train_set_1, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)
    dataloaders['val'] = DataLoader(dataset_val.queue_train_set_1, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)

    num_batches = {'pretrain_unsup': len(dataloaders['train']), 'val': len(dataloaders['val'])}

    # create model
    model = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'])
    model = model.cuda()

    # define criterion, optimizer, and scheduler
    criterion = segmentation_loss(args.loss, False).cuda()

    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5*10**args.wd)
    else:
        print("Optimizer not implemented")
    
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=args.warm_up_duration, after_scheduler=exp_lr_scheduler)

    # training loop
    since = time.time()
    count_iter = 0
    best_val_eval_list = [0 for i in range(4)]
    train_metrics, val_metrics = [], []

    for epoch in range(args.num_epochs):

        count_iter += 1
        if (count_iter-1) % args.display_iter == 0:
            begin_time = time.time()

        model.train()

        train_loss = 0.0
        val_loss = 0.0
        train_loss_vae, val_loss_vae = 0.0, 0.0

        for i, data in enumerate(dataloaders['train']):
            inputs_train = Variable(data['image'][tio.DATA].cuda())
            mask_train = Variable(data['mask'][tio.DATA].squeeze(1).long().cuda())
            name_train = data['ID']
            affine_train = data['image']['affine']

            optimizer.zero_grad()

            out_dict = model(inputs_train)
            outputs_train = out_dict['output']
            reconstr, mu, log_var = out_dict['reconstr'], out_dict['mu'], out_dict['log_var']

            loss_train = criterion(outputs_train, mask_train)
            loss_vae = elbo_metric(out_dict, inputs_train)

            loss_train.backward(retain_graph=True)

            for m in model.modules():
                if hasattr(m, 'local_update'): m.local_update()
            if hasattr(model, 'reset_internal_grads'): model.reset_internal_grads()
            loss_vae.backward()

            optimizer.step()
            train_loss += loss_train.item()
            train_loss_vae += loss_vae.item()

            if i == 0:
                score_list_train = torch.ones_like(outputs_train) #outputs_train # Dummy output for training eval to avoid oom
                mask_list_train = torch.ones_like(mask_train) #mask_train
                name_list_train = name_train
                affine_list_train = affine_train
            else:
                #score_list_train = torch.cat((score_list_train, outputs_train), dim=0)
                #mask_list_train = torch.cat((mask_list_train, mask_train), dim=0)
                score_list_train = torch.cat((score_list_train, score_list_train[-1:]), dim=0)
                mask_list_train = torch.cat((mask_list_train, mask_list_train[-1:]), dim=0)
                name_list_train = np.append(name_list_train, name_train, axis=0)
                affine_list_train = torch.cat((affine_list_train, affine_train), dim=0)

        scheduler_warmup.step()

        if count_iter % args.display_iter == 0:
            print('=' * print_num)
            print('| Epoch {}/{}'.format(epoch + 1, args.num_epochs).ljust(print_num_minus, ' '), '|')
            train_epoch_loss = compute_epoch_loss(train_loss, num_batches, print_num, print_num_minus, unsup_pretrain=True)
            train_epoch_loss_vae = compute_epoch_loss(train_loss_vae, num_batches, print_num, print_num_minus, unsup_pretrain=True)
            if args.threshold:
                train_eval_list = evaluate(cfg['NUM_CLASSES'], score_list_train, mask_list_train, print_num_minus, thr_ranges=[args.threshold, args.threshold+(args.thr_interval/2)])
            else:
                train_eval_list = evaluate(cfg['NUM_CLASSES'], score_list_train, mask_list_train, print_num_minus)
            if args.debug:
                ext = name_list_train[0].rsplit(".", 1)[1]
                name_list_train = [name.rsplit(".", 1)[0] for name in name_list_train]
                name_list_train = [a if not (s:=sum(j == a for j in name_list_train[:i])) else f'{a}-{s+1}'
                    for i, a in enumerate(name_list_train)]
                name_list_train = [name + ".{}".format(ext) for name in name_list_train]
                save_preds_3d(score_list_train, train_eval_list[0], name_list_train, path_train_seg_results, affine_list_train, num_classes=cfg['NUM_CLASSES'])

            # saving metrics to tensorboard writer
            writer.add_scalar('train/segm_loss', train_epoch_loss, count_iter)
            writer.add_scalar('train/vae_loss', train_epoch_loss_vae, count_iter)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], count_iter)
            writer.add_scalar('train/DC', train_eval_list[2], count_iter)
            writer.add_scalar('train/JI', train_eval_list[1], count_iter)
            if cfg['NUM_CLASSES'] == 2:
                writer.add_scalar('train/thresh', train_eval_list[0], count_iter)

            # saving metrics to list
            train_metrics.append({
                'epoch': count_iter,
                'segm/loss': train_epoch_loss,
                'vae/loss': train_epoch_loss_vae,
                'segm/dice': train_eval_list[2],
                'segm/jaccard': train_eval_list[1],
                'lr': optimizer.param_groups[0]['lr'],
                'thresh': train_eval_list[0],
            })

        if count_iter % args.validate_iter == 0:
            metrics_debug = []

            with torch.no_grad():
                model.eval()

                for i, data in enumerate(dataloaders['val']):
                    inputs_val = Variable(data['image'][tio.DATA].cuda())
                    mask_val = Variable(data['mask'][tio.DATA].squeeze(1).long().cuda())
                    name_val = data['ID']
                    affine_val = data['image']['affine']

                    optimizer.zero_grad()

                    out_val_dict = model(inputs_val)
                    outputs_val = out_val_dict['output']
                    reconstr, mu, log_var = out_val_dict['reconstr'], out_val_dict['mu'], out_val_dict['log_var']
                    
                    loss_val = criterion(outputs_val, mask_val)
                    loss_vae_val = elbo_metric(out_val_dict, inputs_val)
                    
                    val_loss += loss_val.item()
                    val_loss_vae += loss_vae_val.item()

                    if i == 0:
                        score_list_val = outputs_val
                        mask_list_val = mask_val
                        name_list_val = name_val
                        affine_list_val = affine_val
                    else:
                        score_list_val = torch.cat((score_list_val, outputs_val), dim=0)
                        mask_list_val = torch.cat((mask_list_val, mask_val), dim=0)
                        name_list_val = np.append(name_list_val, name_val, axis=0)
                        affine_list_val = torch.cat((affine_list_val, affine_val), dim=0)

                    # TODO save val metrics for single images
                    if args.debug:
                        pass

                # TODO save val metrics for single images
                if args.debug:
                    pass
                    # path_val_output_debug = os.path.join(os.path.join(path_run, "val_output_debug"))
                    # if not os.path.exists(path_val_output_debug):
                    #     os.makedirs(path_val_output_debug)
                    # metrics_debug = pd.DataFrame(metrics_debug)
                    # metrics_debug.to_csv(os.path.join(path_val_output_debug, "val_metrics_epoch_{}.csv").format(count_iter), index=False)

                val_epoch_loss = compute_epoch_loss(val_loss, num_batches, print_num, print_num_minus, train=False, unsup_pretrain=True)
                val_epoch_loss_vae = compute_epoch_loss(val_loss_vae, num_batches, print_num, print_num_minus, train=False, unsup_pretrain=True)
                if args.threshold:
                    val_eval_list = evaluate(cfg['NUM_CLASSES'], score_list_val, mask_list_val, print_num_minus, train=False, thr_ranges=[args.threshold, args.threshold+(args.thr_interval/2)])
                else:
                    val_eval_list = evaluate(cfg['NUM_CLASSES'], score_list_val, mask_list_val, print_num_minus, train=False)
                # check if best model (in terms of JI) and eventually save it
                if best_val_eval_list[1] < val_eval_list[1]:
                    best_val_eval_list = val_eval_list
                    save_snapshot(model, path_trained_models, threshold=val_eval_list[0], save_best=True)
                    # save val best preds
                    ext = name_list_val[0].rsplit(".", 1)[1]
                    name_list_val = [name.rsplit(".", 1)[0] for name in name_list_val]
                    name_list_val = [a if not (s:=sum(j == a for j in name_list_val[:i])) else f'{a}-{s+1}'
                        for i, a in enumerate(name_list_val)]
                    name_list_val = [name + ".{}".format(ext) for name in name_list_val]
                    save_preds_3d(score_list_val, val_eval_list[0], name_list_val, os.path.join(path_seg_results, 'best_model'), affine_list_val, num_classes=cfg['NUM_CLASSES'])
                
                # saving metrics to tensorboard writer
                writer.add_scalar('val/segm_loss', val_epoch_loss, count_iter)
                writer.add_scalar('val/vae_loss', val_epoch_loss_vae, count_iter)
                writer.add_scalar('val/DC', val_eval_list[2], count_iter)
                writer.add_scalar('val/JI', val_eval_list[1], count_iter)
                if cfg['NUM_CLASSES'] == 2:
                    writer.add_scalar('val/thresh', val_eval_list[0], count_iter)

                # saving metrics to list
                val_metrics.append({
                    'epoch': count_iter,
                    'segm/loss': val_epoch_loss,
                    'vae/loss': val_epoch_loss_vae,
                    'segm/dice': val_eval_list[2],
                    'segm/jaccard': val_eval_list[1],
                    'thresh': val_eval_list[0],
                })
                
                print('-' * print_num)
                print('| Epoch Time: {:.4f}s'.format((time.time() - begin_time) / args.display_iter).ljust(print_num_minus, ' '), '|')

    # save val last preds
    ext = name_list_val[0].rsplit(".", 1)[1]
    name_list_val = [name.rsplit(".", 1)[0] for name in name_list_val]
    name_list_val = [a if not (s:=sum(j == a for j in name_list_val[:i])) else f'{a}-{s+1}'
        for i, a in enumerate(name_list_val)]
    name_list_val = [name + ".{}".format(ext) for name in name_list_val]
    save_preds_3d(score_list_val, val_eval_list[0], name_list_val, os.path.join(path_seg_results, 'last_model'), affine_list_val, num_classes=cfg['NUM_CLASSES'])

    # save last model
    save_snapshot(model, path_trained_models, threshold=val_eval_list[0], save_best=False)

    # save train and val metrics in csv file
    train_metrics = pd.DataFrame(train_metrics)
    train_metrics.to_csv(os.path.join(path_run, 'train_log.csv'), index=False)
    val_metrics = pd.DataFrame(val_metrics)
    val_metrics.to_csv(os.path.join(path_run, 'val_log.csv'), index=False)

    time_elapsed = time.time() - since
    m, s = divmod(time_elapsed, 60)
    h, m = divmod(m, 60)

    print('=' * print_num)
    print('| Training Completed In {:.0f}h {:.0f}mins {:.0f}s'.format(h, m, s).ljust(print_num_minus, ' '), '|')
    print('-' * print_num)
    print_best_val_metrics(cfg['NUM_CLASSES'], best_val_eval_list, print_num_minus)
    print('=' * print_num)
