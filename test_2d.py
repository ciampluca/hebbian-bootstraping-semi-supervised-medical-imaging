import argparse
import time
import os
import pandas as pd
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from config.dataset_config.dataset_cfg import dataset_cfg
from config.augmentation.online_aug import data_transform_2d, data_normalize_2d
from models.getnetwork import get_network
from dataload.dataset_2d import imagefloder_itn
from utils import save_preds, evaluate, evaluate_distance

from hebb.makehebbian import makehebbian

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--path_exp', default='./runs/GlaS/semi_sup/kaiming_unet/inv_temp-1/regime-1/run-0')
    parser.add_argument('--best', default='JI', type=str, help="JI, DC, last")
    parser.add_argument('--path_dataset', default='data/GlaS')
    parser.add_argument('--dataset_name', default='GlaS', help='GlaS')
    parser.add_argument('--input1', default='image')
    parser.add_argument('--if_mask', default=True)
    parser.add_argument('--threshold', default=None,  type=float)    # help: 0.56, 0.54 ?
    parser.add_argument('--thr_interval', default=0.02,  type=float)
    parser.add_argument('-b', '--batch_size', default=4, type=int)
    parser.add_argument('-n', '--network', default='unet', type=str)

    parser.add_argument('--hebbian_pretrain', default=False)

    args = parser.parse_args()

     # set cuda device
    torch.cuda.set_device(args.device)

    # load dataset config
    dataset_name = args.dataset_name
    cfg = dataset_cfg(dataset_name)

    print_num = 42 + (cfg['NUM_CLASSES'] - 3) * 7
    print_num_minus = print_num - 2

    # create folders
    path_seg_results = os.path.join(os.path.join(args.path_exp, "test_seg_preds"))
    if not os.path.exists(path_seg_results):
        os.makedirs(path_seg_results)

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

    dataset_val = imagefloder_itn(
        data_dir=args.path_dataset + '/val',
        input1=args.input1,
        data_transform_1=data_transforms['val'],
        data_normalize_1=data_normalize,
        sup=True,
    )

    dataloaders = dict()
    dataloaders['val'] = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=16)

    num_batches = {'val': len(dataloaders['val'])}

    # create model and load weights
    model = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'])
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
    since = time.time()

    with torch.no_grad():
        model.eval()

        for i, data in enumerate(dataloaders['val']):
            inputs_test = data['image']
            inputs_test = Variable(inputs_test.cuda(non_blocking=True))
            name_test = data['ID']
            if args.if_mask:
                mask_test = data['mask']
                mask_test = Variable(mask_test.cuda(non_blocking=True))

            if args.network == "unet_urpc" or args.network == "unet_cct":
                outputs_test, _, _, _ = model(inputs_test)
            else:
                outputs_test = model(inputs_test)

            if args.if_mask:
                if i == 0:
                    score_list_test = outputs_test
                    name_list_test = name_test
                    mask_list_test = mask_test
                else:
                    score_list_test = torch.cat((score_list_test, outputs_test), dim=0)
                    name_list_test = np.append(name_list_test, name_test, axis=0)
                    mask_list_test = torch.cat((mask_list_test, mask_test), dim=0)
            else:
                save_preds(outputs_test, threshold, name_test, path_seg_results, cfg['PALETTE'])

        if args.if_mask:
            print('=' * print_num)
            pixel_metrics = evaluate(cfg['NUM_CLASSES'], score_list_test, mask_list_test, print_num_minus, train=False, thr_ranges=[threshold, threshold+(args.thr_interval/2)])
            distance_metrics = evaluate_distance(cfg['NUM_CLASSES'], score_list_test, mask_list_test, thr_ranges=[threshold, threshold+(args.thr_interval/2)])
            save_preds(score_list_test, threshold, name_list_test, path_seg_results, cfg['PALETTE'])

    # save test metrics in csv file
    test_metrics = pd.DataFrame([{
        'segm/dice': pixel_metrics[2],
        'segm/jaccard': pixel_metrics[1],
        'segm/asd': distance_metrics[1],
        'segm/95hd': distance_metrics[0], 
        'thresh': pixel_metrics[0],
    }])
    test_metrics.to_csv(os.path.join(args.path_exp, 'test.csv'), index=False)

    time_elapsed = time.time() - since
    m, s = divmod(time_elapsed, 60)
    h, m = divmod(m, 60)
    print('-' * print_num)
    print('| Testing Completed In {:.0f}h {:.0f}mins {:.0f}s'.format(h, m, s).ljust(print_num_minus, ' '), '|')
    print('=' * print_num)