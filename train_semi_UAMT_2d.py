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

from models.getnetwork import get_network
from config.dataset_config.dataset_cfg import dataset_cfg
from config.augmentation.online_aug import data_transform_2d, data_normalize_2d
from loss.loss_function import segmentation_loss, softmax_mse_loss
from dataload.dataset_2d import imagefloder_itn
from config.warmup_config.warmup import GradualWarmupScheduler
from config.ramps import ramps
from utils import save_snapshot, save_preds, init_seeds, evaluate, print_best_val_metrics, update_ema_variables, compute_epoch_loss_MT, compute_val_epoch_loss_MT, evaluate_val_MT

from hebb.makehebbian import makehebbian
from models.networks_2d.unet import init_weights as init_weights_unet

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--path_root_exp', default='./runs')
    parser.add_argument('--path_dataset', default='data/GlaS')
    parser.add_argument('--dataset_name', default='GlaS', help='GlaS')
    parser.add_argument('--input1', default='image')
    parser.add_argument('--regime', default=20, type=int, help="percentage of labeled data to be used")
    parser.add_argument('-b', '--batch_size', default=2, type=int)
    parser.add_argument('-e', '--num_epochs', default=200, type=int)
    parser.add_argument('-s', '--step_size', default=50, type=int)
    parser.add_argument('--optimizer', default="sgd", type=str, help="adam, sgd")
    parser.add_argument('-l', '--lr', default=0.5, type=float)
    parser.add_argument('-g', '--gamma', default=0.5, type=float)
    parser.add_argument('--loss', default='dice', type=str)
    parser.add_argument('-w', '--warm_up_duration', default=20)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('-u', '--unsup_weight', default=1, type=float)
    parser.add_argument('-i', '--display_iter', default=1, type=int)
    parser.add_argument('--validate_iter', default=2, type=int)
    parser.add_argument('-n', '--network', default='unet', type=str)
    parser.add_argument('--debug', default=True)
    parser.add_argument('--ema_decay', default=0.99, type=float)
    parser.add_argument('--init_weights', default='kaiming', type=str)
    
    parser.add_argument('--load_hebbian_weights', default=None, type=str, help='path of hebbian pretrained weights')
    parser.add_argument('--hebbian_rule', default='swta_t', type=str, help='hebbian rules to be used')
    parser.add_argument('--hebb_inv_temp', default=1, type=int, help='hebbian temp')

    args = parser.parse_args()

     # set cuda device
    torch.cuda.set_device(args.device)
    init_seeds(args.seed)

    dataset_name = args.dataset_name
    cfg = dataset_cfg(dataset_name)

    # load dataset config
    dataset_name = args.dataset_name
    cfg = dataset_cfg(dataset_name)

    print_num = 77 + (cfg['NUM_CLASSES'] - 3) * 14
    print_num_minus = print_num - 2
    print_num_half = int(print_num / 2 - 1)

    # create folders
    if args.regime < 100:
        if args.load_hebbian_weights:
            path_run = os.path.join(args.path_root_exp, os.path.split(args.path_dataset)[1], "semi_sup", "h_uamt_{}_{}".format(args.network, args.hebbian_rule), "inv_temp-{}".format(args.hebb_inv_temp), "regime-{}".format(args.regime), "run-{}".format(args.seed))
        else:
            if args.init_weights != "kaiming":
                path_run = os.path.join(args.path_root_exp, os.path.split(args.path_dataset)[1], "semi_sup", "{}_uamt_{}".format(args.init_weights, args.network), "inv_temp-1", "regime-{}".format(args.regime), "run-{}".format(args.seed))
            else:
                path_run = os.path.join(args.path_root_exp, os.path.split(args.path_dataset)[1], "semi_sup", "uamt_{}".format(args.network), "inv_temp-1", "regime-{}".format(args.regime), "run-{}".format(args.seed))
    else:
        path_run = os.path.join(args.path_root_exp, os.path.split(args.path_dataset)[1], "fully_sup", "uamt_{}".format(args.network), "inv_temp-1", "regime-{}".format(args.regime), "run-{}".format(args.seed))
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

    # create tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(path_run, 'runs'))

    # save config to file
    with open(os.path.join(path_run, "config.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # set input type
    if args.input1 == 'image':
        input1_mean = 'MEAN'
        input1_std = 'STD'
    else:
        input1_mean = 'MEAN_' + args.input1
        input1_std = 'STD_' + args.input1

    # data loading
    data_transforms = data_transform_2d()
    data_normalize = data_normalize_2d(cfg[input1_mean], cfg[input1_std])

    dataset_train_sup = imagefloder_itn(
        data_dir=args.path_dataset + '/train',
        input1=args.input1,
        data_transform_1=data_transforms['train'],
        data_normalize_1=data_normalize,
        sup=True,
        regime=args.regime,
        seed=args.seed,
    )
    dataset_train_unsup = imagefloder_itn(
        data_dir=args.path_dataset + '/train',
        input1=args.input1,
        data_transform_1=data_transforms['train'],
        data_normalize_1=data_normalize,
        sup=False,
        regime=args.regime,
        seed=args.seed,
    )
    dataset_val = imagefloder_itn(
        data_dir=args.path_dataset + '/val',
        input1=args.input1,
        data_transform_1=data_transforms['val'],
        data_normalize_1=data_normalize,
        sup=True,
    )

    dataloaders = dict()
    dataloaders['train_sup'] = DataLoader(dataset_train_sup, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8)
    dataloaders['train_unsup'] = DataLoader(dataset_train_unsup, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8)
    dataloaders['val'] = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8)

    num_batches = {'train_sup': len(dataloaders['train_sup']), 'train_unsup': len(dataloaders['train_unsup']), 'val': len(dataloaders['val'])}

    # create models
    model1 = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'], args.init_weights)
    model2 = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'], args.init_weights)

    # eventually load hebbian weights
    hebb_params, exclude, exclude_layer_names = None, None, None
    if args.load_hebbian_weights:
        print("Loading Hebbian pre-trained weights")
        state_dict = torch.load(args.load_hebbian_weights, map_location='cpu')
        hebb_params = state_dict['hebb_params']
        hebb_params['alpha'] = 0
        exclude = state_dict['excluded_layers']
        model1 = makehebbian(model1, exclude=exclude, hebb_params=hebb_params)
        model1.load_state_dict(state_dict['model'])
        model2 = makehebbian(model2, exclude=exclude, hebb_params=hebb_params)

        exclude_layer_names = exclude
        if exclude is None: exclude = []
        exclude = [(n, m) for n, m in model1.named_modules() if any([n == e for e in exclude])]
        exclude = [m for _, p in exclude for m in [*p.modules()]]
        for m in exclude:
            init_weights_unet(m, init_type='kaiming')

        model2_params = {n: p for n, p in model2.named_parameters()}

        for n, p in model1.named_parameters():
            p.requires_grad = True
            model2_params[n].data += p
            model2_params[n].requires_grad = True

        exclude = [(n, m) for n, m in model2.named_modules() if any([n == e for e in exclude])]
        exclude = [m for _, p in exclude for m in [*p.modules()]]
        for m in exclude:
            init_weights_unet(m, init_type='kaiming')

    model1 = model1.cuda()
    model2 = model2.cuda()

    # define criterion, optimizer, and scheduler
    criterion = segmentation_loss(args.loss, False).cuda()

    if args.optimizer == "adam":
        optimizer1 = optim.Adam(model1.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer1 = optim.SGD(model1.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5 * 10 ** args.wd)
    else:
        print("Optimizer not implemented")

    exp_lr_scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=args.step_size, gamma=args.gamma)
    scheduler_warmup1 = GradualWarmupScheduler(optimizer1, multiplier=1.0, total_epoch=args.warm_up_duration, after_scheduler=exp_lr_scheduler1)

    # training loop
    since = time.time()
    count_iter = 0
    best_val_eval_list = [0 for i in range(4)]
    train_metrics, val_metrics = [], []

    best_model = model1
    best_score_list_val = []

    for epoch in range(args.num_epochs):

        count_iter += 1
        if (count_iter - 1) % args.display_iter == 0:
            begin_time = time.time()

        model1.train()
        model2.train()

        train_loss_sup_1 = 0.0
        train_loss_unsup = 0.0
        train_loss = 0.0
        val_loss_sup_1 = 0.0
        val_loss_sup_2 = 0.0

        unsup_weight = args.unsup_weight * (epoch + 1) / args.num_epochs

        dataset_train_sup = iter(dataloaders['train_sup'])
        dataset_train_unsup = iter(dataloaders['train_unsup'])

        for i in range(num_batches['train_sup']):
            unsup_index = next(dataset_train_unsup)
            img_train_unsup_1 = unsup_index['image']
            img_train_unsup_1 = Variable(img_train_unsup_1.cuda(non_blocking=True))

            noise = torch.clamp(torch.randn_like(img_train_unsup_1) * 0.1, -0.2, 0.2)
            img_train_unsup_2 = img_train_unsup_1 + noise

            optimizer1.zero_grad()

            pred_train_unsup1 = model1(img_train_unsup_1)
            with torch.no_grad():
                pred_train_unsup2 = model2(img_train_unsup_2)

            T = 8
            _, _, w, h = img_train_unsup_1.shape
            volume_batch_r = img_train_unsup_1.repeat(2, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, cfg['NUM_CLASSES'], w, h]).cuda()
            for i_ in range(T // 2):
                ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    preds[2 * stride * i_:2 * stride * (i_ + 1)] = model2(ema_inputs)
            preds = torch.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, cfg['NUM_CLASSES'], w, h)
            preds = torch.mean(preds, dim=0)
            uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)

            consistency_dist = softmax_mse_loss(pred_train_unsup1, pred_train_unsup2)  # (batch, 2, 112,112,80)
            threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(epoch, args.num_epochs)) * np.log(2)
            mask = (uncertainty < threshold).float()
            loss_train_unsup = torch.sum(mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)

            loss_train_unsup = loss_train_unsup * unsup_weight
            loss_train_unsup.backward(retain_graph=True)
            torch.cuda.empty_cache()

            sup_index = next(dataset_train_sup)
            img_train_sup = sup_index['image']
            img_train_sup = Variable(img_train_sup.cuda(non_blocking=True))
            mask_train_sup = sup_index['mask']
            mask_train_sup = Variable(mask_train_sup.cuda(non_blocking=True))
            name_train = sup_index['ID']

            pred_train_sup1 = model1(img_train_sup)

            if count_iter % args.display_iter == 0:
                if i == 0:
                    score_list_train = pred_train_sup1
                    mask_list_train = mask_train_sup
                    name_list_train = name_train
                else:
                    score_list_train = torch.cat((score_list_train, pred_train_sup1), dim=0)
                    mask_list_train = torch.cat((mask_list_train, mask_train_sup), dim=0)
                    name_list_train = np.append(name_list_train, name_train, axis=0)

            loss_train_sup1 = criterion(pred_train_sup1, mask_train_sup)

            loss_train_sup = loss_train_sup1
            loss_train_sup.backward()

            optimizer1.step()
            update_ema_variables(model1, model2, args.ema_decay, epoch)

            loss_train = loss_train_unsup + loss_train_sup
            train_loss_unsup += loss_train_unsup.item()
            train_loss_sup_1 += loss_train_sup1.item()
            train_loss += loss_train.item()

        scheduler_warmup1.step()

        if count_iter % args.display_iter == 0:
            print('=' * print_num)
            print('| Epoch {}/{}'.format(epoch + 1, args.num_epochs).ljust(print_num_minus, ' '), '|')
            train_epoch_loss_sup, train_epoch_loss_unsup, train_epoch_loss = compute_epoch_loss_MT(train_loss_sup_1, train_loss_unsup, train_loss, num_batches, print_num, print_num_half, print_num_minus)
            train_eval_list = evaluate(cfg['NUM_CLASSES'], score_list_train, mask_list_train, print_num_minus)
            if args.debug:
                save_preds(score_list_train, train_eval_list[0], name_list_train, path_train_seg_results, cfg['PALETTE'])

            # saving metrics to tensorboard writer
            writer.add_scalar('train/segm_loss', train_epoch_loss_sup, count_iter)
            writer.add_scalar('train/unsup_loss', train_epoch_loss_unsup, count_iter)
            writer.add_scalar('train/total_loss', train_epoch_loss, count_iter)
            writer.add_scalar('train/lr', optimizer1.param_groups[0]['lr'], count_iter)
            writer.add_scalar('train/DC', train_eval_list[2], count_iter)
            writer.add_scalar('train/JI', train_eval_list[1], count_iter)
            writer.add_scalar('train/thresh', train_eval_list[0], count_iter)
            writer.add_scalar('train/lambda_unsup', unsup_weight, count_iter)

            # saving metrics to list
            train_metrics.append({
                'epoch': count_iter,
                'segm/loss': train_epoch_loss_sup,
                'segm/unsup_loss': train_epoch_loss_unsup,
                'segm/total_loss': train_epoch_loss,
                'segm/dice': train_eval_list[2],
                'segm/jaccard': train_eval_list[1],
                'lr': optimizer1.param_groups[0]['lr'],
                'thresh': train_eval_list[0],
            })

        if count_iter % args.validate_iter == 0:
            metrics_debug = []

            with torch.no_grad():
                model1.eval()
                model2.eval()

                for i, data in enumerate(dataloaders['val']):
                    inputs_val = Variable(data['image'].cuda(non_blocking=True))
                    mask_val = Variable(data['mask'].cuda(non_blocking=True))
                    name_val = data['ID']

                    optimizer1.zero_grad()

                    outputs_val1 = model1(inputs_val)
                    outputs_val2 = model2(inputs_val)

                    loss_val_sup1 = criterion(outputs_val1, mask_val)
                    loss_val_sup2 = criterion(outputs_val2, mask_val)
                    val_loss_sup_1 += loss_val_sup1.item()
                    val_loss_sup_2 += loss_val_sup2.item()

                    if i == 0:
                        score_list_val1 = outputs_val1
                        score_list_val2 = outputs_val2
                        mask_list_val = mask_val
                        name_list_val = name_val
                    else:
                        score_list_val1 = torch.cat((score_list_val1, outputs_val1), dim=0)
                        score_list_val2 = torch.cat((score_list_val2, outputs_val2), dim=0)
                        mask_list_val = torch.cat((mask_list_val, mask_val), dim=0)
                        name_list_val = np.append(name_list_val, name_val, axis=0)

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
                    save_snapshot(best_model, path_trained_models, threshold=best_val_eval_list[0], save_best=True, hebb_params=hebb_params, layers_excluded=exclude_layer_names)
                    # save val best preds
                    save_preds(best_score_list_val, best_val_eval_list[0], name_list_val, os.path.join(path_seg_results, 'best_model'), cfg['PALETTE'])
                
                # saving metrics to tensorboard writer
                writer.add_scalar('val/segm_loss', val_epoch_loss1, count_iter)
                writer.add_scalar('val/DC', val_eval_list1[2], count_iter)
                writer.add_scalar('val/JI', val_eval_list1[1], count_iter)
                writer.add_scalar('val/thresh', val_eval_list1[0], count_iter)
                writer.add_scalar('val/segm_loss2', val_epoch_loss2, count_iter)
                writer.add_scalar('val/DC2', val_eval_list2[2], count_iter)
                writer.add_scalar('val/JI2', val_eval_list2[1], count_iter)
                writer.add_scalar('val/thresh2', val_eval_list2[0], count_iter)

                # saving metrics to list
                val_metrics.append({
                    'epoch': count_iter,
                    'segm/loss': val_epoch_loss1,
                    'segm/dice': val_eval_list1[2],
                    'segm/jaccard': val_eval_list1[1],
                    'thresh': val_eval_list1[0],
                    'segm/loss2': val_epoch_loss2,
                    'segm/dice2': val_eval_list2[2],
                    'segm/jaccard2': val_eval_list2[1],
                    'thresh2': val_eval_list2[0],
                })

                print('-' * print_num)
                print('| Epoch Time: {:.4f}s'.format((time.time() - begin_time) / args.display_iter).ljust(print_num_minus, ' '), '|')

    # save val last preds
    save_preds(score_list_val1, val_eval_list1[0], name_list_val, os.path.join(path_seg_results, 'last_model'), cfg['PALETTE'])
    save_preds(score_list_val2, val_eval_list2[0], name_list_val, os.path.join(path_seg_results, 'last_model2'), cfg['PALETTE'])

    # save last model
    save_snapshot(model1, path_trained_models, threshold=val_eval_list1[0], save_best=False, hebb_params=hebb_params, layers_excluded=exclude_layer_names)
    save_snapshot(model2, path_trained_models2, threshold=val_eval_list2[0], save_best=False, hebb_params=hebb_params, layers_excluded=exclude_layer_names)

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