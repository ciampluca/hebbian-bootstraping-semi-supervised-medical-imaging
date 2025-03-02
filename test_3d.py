import argparse
import time
import os
import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchio as tio

from config.dataset_config.dataset_cfg import dataset_cfg
from config.augmentation.online_aug import data_transform_3d
from models.getnetwork import get_network
from dataload.dataset_3d import dataset_it
from utils import evaluate, evaluate_distance, save_preds, save_test_3d, postprocess_3d_pred, offline_eval

from hebb.makehebbian import makehebbian
from models.networks_2d.unet_ddpm import SuperDiffusion

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--path_exp', default='./runs/Atrial/semi_sup/kaiming_unet3d/inv_temp-1/regime-1/run-0')
    parser.add_argument('--best', default='JI', type=str, help="JI, DC, last")
    parser.add_argument('--path_dataset', default='data/Atrial')
    parser.add_argument('--dataset_name', default='Atrial', help='Atrial')
    parser.add_argument('--input1', default='image')
    parser.add_argument('--threshold', default=None, type=float)
    parser.add_argument('--thr_interval', default=0.02,  type=float)
    parser.add_argument('--patch_size', default=(112, 112, 32))
    parser.add_argument('--patch_overlap', default=(56, 56, 16))
    parser.add_argument('-b', '--batch_size', default=1, type=int)
    parser.add_argument('-n', '--network', default='unet3d')
    parser.add_argument('--hebbian_pretrain', default=False)
    parser.add_argument('--fill_hole_thr', default=500, type=int, help='300-500')     # 100 for LiTS, 500 for Atrial
    parser.add_argument('--postprocessing', default=False)
    parser.add_argument('--timestamp_diffusion', default=1000, type=int)

    args = parser.parse_args()

     # set cuda device
    torch.cuda.set_device(args.device)

    # load dataset config
    dataset_name = args.dataset_name
    cfg = dataset_cfg(dataset_name)

    print_num = 42 + (cfg['NUM_CLASSES'] - 3) * 7
    print_num_minus = print_num - 2

    # parse args params
    if isinstance(args.patch_size, str):
        args.patch_size = eval(args.patch_size)
    if isinstance(args.patch_overlap, str):
        args.patch_overlap = eval(args.patch_overlap)

    # create folders
    path_seg_results = os.path.join(os.path.join(args.path_exp, "test_seg_preds"))
    if not os.path.exists(path_seg_results):
        os.makedirs(path_seg_results)

    # data loading
    data_transform = data_transform_3d(cfg['NORMALIZE'])

    dataset_val = dataset_it(
        data_dir=args.path_dataset + '/val',
        input1=args.input1,
        transform_1=data_transform['test'],
        num_classes=cfg['NUM_CLASSES'],
    )

    # create model and load weights
    model = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'], img_shape=args.patch_size)
    if args.network == 'unet_ddpm' or args.network == 'unet3d_ddpm':
        if args.network == 'unet3d_ddpm':
            from models.networks_3d.unet3d_ddpm import SuperDiffusion
        diffusion = SuperDiffusion(
            model.net,
            image_size = args.patch_size[0],
            timesteps = args.timestamp_diffusion,    # number of steps
            objective='pred_noise', #'pred_x0', #'pred_v', #'pred_noise', #
        )
        diffusion = diffusion.to(args.device)
        diffusion_seg = SuperDiffusion(
            model.net_seg,
            image_size = args.patch_size[0],
            timesteps = args.timestamp_diffusion,    # number of steps
            objective='pred_x0', #'pred_x0', #'pred_v', #'pred_noise', #
        )
        diffusion_seg = diffusion_seg.to(args.device)
    name_snapshot = 'last' if args.best == 'last' else 'best_{}'.format(args.best)
    path_snapshot = os.path.join(args.path_exp, 'checkpoints', '{}.pth'.format(name_snapshot))
    state_dict = torch.load(path_snapshot, map_location='cpu')
    if args.hebbian_pretrain:
        hebb_params = state_dict['hebb_params']
        exclude = state_dict['excluded_layers']
        model = makehebbian(model, exclude=exclude, hebb_params=hebb_params)
    model.load_state_dict(state_dict['model'])
    threshold = state_dict['threshold'] if args.threshold is None else args.threshold
    model = model.cuda()

    # test loop
    print('-' * print_num)
    print('| Starting Testing'.ljust(print_num_minus, ' '), '|')
    print('=' * print_num)
    since = time.time()

    score_list_test, mask_list_test, name_list_test = None, None, None
    for i, subject in enumerate(dataset_val.dataset_1):

        grid_sampler = tio.inference.GridSampler(
            subject=subject,
            patch_size=args.patch_size,
            patch_overlap=args.patch_overlap
        )

        dataloaders = dict()
        dataloaders['test'] = DataLoader(grid_sampler, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=16)
        aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='average')

        with torch.no_grad():
            model.eval()

            for data in dataloaders['test']:

                inputs_test = Variable(data['image'][tio.DATA].cuda())
                location_test = data[tio.LOCATION]

                if args.network == "unet3d_urpc" or args.network == "unet3d_cct" or args.network == "vnet_urpc" or args.network == "vnet_cct":
                    outputs_test, _, _, _ = model(inputs_test)
                elif args.network == "vnet_dtc" or args.network == "unet3d_dtc":
                    _, outputs_test = model(inputs_test)
                elif args.network == "unet3d_vae":
                    outputs_test = model(inputs_test)['output']
                elif args.network == "unet3d_superpix":
                    outputs_test, _ = model(inputs_test)
                elif args.network == "unet3d_ddpm":
                    zero_mask = torch.zeros((inputs_test.shape[0], cfg['NUM_CLASSES'], *inputs_test.shape[2:]), device=inputs_test.device, dtype=torch.int64)
                    _, outputs_test = diffusion_seg(inputs_test, zero_mask, conditioner='img')
                elif args.network == "unet_ddpm":
                    name_test = data['ID']
                    mask_test = Variable(data['mask'][tio.DATA].squeeze(1).long().cuda())
                    slice_idx = inputs_test.shape[-1] // 2
                    inputs_test, mask_test = inputs_test[:, :, :, :, slice_idx], mask_test[:, :, :, slice_idx]
                    zero_mask = torch.zeros((inputs_test.shape[0], cfg['NUM_CLASSES'], *inputs_test.shape[2:]), device=inputs_test.device, dtype=torch.int64)
                    _, outputs_test = diffusion_seg(inputs_test, zero_mask, conditioner='img')
                else:
                    outputs_test = model(inputs_test)

                if args.network == 'unet_ddpm':
                    score_list_test = torch.cat((score_list_test, outputs_test), dim=0) if score_list_test is not None else outputs_test
                    mask_list_test = torch.cat((mask_list_test, mask_test), dim=0) if mask_list_test is not None else mask_test
                    name_list_test = np.append(name_list_test, name_test, axis=0) if name_list_test is not None else name_test
                else:
                    aggregator.add_batch(outputs_test, location_test)

        if args.network == 'unet_ddpm':
            pass
        else:
            outputs_tensor = aggregator.get_output_tensor()
            save_test_3d(cfg['NUM_CLASSES'], outputs_tensor, subject['ID'], threshold, path_seg_results, subject['image']['affine'])

    time_elapsed = time.time() - since
    m, s = divmod(time_elapsed, 60)
    h, m = divmod(m, 60)
    print('-' * print_num)
    print('| Testing Completed In {:.0f}h {:.0f}mins {:.0f}s'.format(h, m, s).ljust(print_num_minus, ' '), '|')
    print('=' * print_num)

    if args.network == 'unet_ddpm':
        thr_ranges = [threshold, threshold+(args.interval/2)] if cfg['NUM_CLASSES'] == 2 else None
        pixel_metrics = evaluate(cfg['NUM_CLASSES'], score_list_test, mask_list_test, print_num_minus, train=False, thr_ranges=thr_ranges)
        distance_metrics = evaluate_distance(cfg['NUM_CLASSES'], score_list_test, mask_list_test, thr_ranges=thr_ranges)
        ext = 'png' #name_list_train[0].rsplit(".", 1)[1]
        name_list_test = [name.rsplit(".", 1)[0] for name in name_list_test]
        name_list_test = [a if not (s:=sum(j == a for j in name_list_test[:i])) else f'{a}-{s+1}'
                for i, a in enumerate(name_list_test)]
        name_list_test = [name + ".{}".format(ext) for name in name_list_test]
        save_preds(score_list_test, pixel_metrics[0], name_list_test, path_seg_results, cfg['PALETTE'], num_classes=cfg['NUM_CLASSES'])
        #print("Dc: {}, Jc: {}, Thr: {}".format(pixel_metrics[2], pixel_metrics[1], pixel_metrics[0]))
        test_results = {'dice': pixel_metrics[2], 'jaccard': pixel_metrics[1], 'thr': pixel_metrics[0], 'sd': distance_metrics[1], 'hd': distance_metrics[0]}

    else:
        path_seg_postprocessed_results = path_seg_results
        if args.postprocessing:
            # pred post-processing
            print('-' * print_num)
            print('| Starting Preds Post-Processing'.ljust(print_num_minus, ' '), '|')
            print('=' * print_num)
            since = time.time()

            path_seg_postprocessed_results = os.path.join(os.path.join(args.path_exp, "test_seg_preds_postprocessed"))
            if not os.path.exists(path_seg_postprocessed_results):
                os.makedirs(path_seg_postprocessed_results)
            postprocess_3d_pred(args.dataset_name, path_seg_results, path_seg_postprocessed_results, args.fill_hole_thr)

            time_elapsed = time.time() - since
            m, s = divmod(time_elapsed, 60)
            h, m = divmod(m, 60)
            print('-' * print_num)
            print('| Preds Post-Processing Completed In {:.0f}h {:.0f}mins {:.0f}s'.format(h, m, s).ljust(print_num_minus, ' '), '|')
            print('=' * print_num)

        # evaluation
        print('-' * print_num)
        print('| Starting Eval'.ljust(print_num_minus, ' '), '|')
        print('=' * print_num)
        since = time.time()

        test_results = offline_eval(path_seg_postprocessed_results, os.path.join(args.path_dataset, "val", "mask"), num_classes=cfg['NUM_CLASSES'])

        time_elapsed = time.time() - since
        m, s = divmod(time_elapsed, 60)
        h, m = divmod(m, 60)
        print('-' * print_num)
        print('| Eval Completed In {:.0f}h {:.0f}mins {:.0f}s'.format(h, m, s).ljust(print_num_minus, ' '), '|')
        print('=' * print_num)

    # save test metrics in csv file
    test_metrics = pd.DataFrame([{
        'segm/dice': test_results['dice'],
        'segm/jaccard': test_results['jaccard'],
        'segm/asd': test_results['sd'],
        'segm/95hd': test_results['hd'], 
        #'thresh': pixel_metrics[0],
    }])
    test_metrics.to_csv(os.path.join(args.path_exp, 'test.csv'), index=False)
