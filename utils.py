import numpy as np
import os
import random

import torch

from PIL import Image
import albumentations as A
from medpy.metric.binary import hd95, assd
from sklearn.metrics import confusion_matrix



def init_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_snapshot(model, path, threshold=None, save_best=False, hebb_params=None, layers_excluded=None):

    def save(model, save_path, hebb_params, layers_excluded):
        if hebb_params:
            torch.save({
                'model': model.state_dict(),
                'threshold': threshold,
                'hebb_params': hebb_params, 
                'excluded_layers': layers_excluded,
                },
                save_path,
            )
        else:
            torch.save({
                'model': model.state_dict(),
                'threshold': threshold,
                },
                save_path,
            )

    model_name = 'last.pth'
    
    if save_best:
        model_name = 'best_JI.pth'
        save(model, os.path.join(path, model_name), hebb_params, layers_excluded)
    else:
        save(model, os.path.join(path, model_name), hebb_params, layers_excluded)


def save_preds(score_list_val, threshold, name_list_val, path_seg_results, palette):
    score_list_val = torch.softmax(score_list_val, dim=1)
    pred_results = score_list_val[:, 1, :, :].cpu().detach().numpy()
    pred_results[pred_results > threshold] = 1
    pred_results[pred_results <= threshold] = 0

    assert len(name_list_val) == pred_results.shape[0]

    if not os.path.exists(path_seg_results):
            os.makedirs(path_seg_results)
    for i in range(len(name_list_val)):
        color_results = Image.fromarray(pred_results[i].astype(np.uint8), mode='P')
        color_results.putpalette(palette)
        color_results.save(os.path.join(path_seg_results, name_list_val[i]))


def compute_epoch_loss(loss, num_batches, print_num, print_num_minus, train=True, print_on_screen=True, unsup_pretrain=False):
    epoch_loss = loss / num_batches['pretrain_unsup'] if unsup_pretrain else loss / num_batches['train_sup']

    if print_on_screen:
        print('-' * print_num)
        text = 'Train Loss' if train else 'Val Loss'
        print('| {}: {:.4f}'.format(text, epoch_loss).ljust(print_num_minus, ' '), '|')
        print('-' * print_num)

    return epoch_loss


def print_best_val_metrics(num_classes, best_val_list, print_num):
    if num_classes == 2:
        print('| Best Val Thr: {:.4f}'.format(best_val_list[0]).ljust(print_num, ' '), '|')
        print('| Best Val  Jc: {:.4f}'.format(best_val_list[1]).ljust(print_num, ' '), '|')
        print('| Best Val  Dc: {:.4f}'.format(best_val_list[2]).ljust(print_num, ' '), '|')
    else:
        np.set_printoptions(precision=4, suppress=True)
        print('| Best Val  Jc: {}'.format(best_val_list[0]).ljust(print_num, ' '), '|')
        print('| Best Val  Dc: {}'.format(best_val_list[2]).ljust(print_num, ' '), '|')
        print('| Best Val mJc: {:.4f}'.format(best_val_list[1]).ljust(print_num, ' '), '|')
        print('| Best Val mDc: {:.4f}'.format(best_val_list[3]).ljust(print_num, ' '), '|')


def evaluate(num_classes, score_list, mask_list, print_num, print_on_screen=True, train=True, thr_ranges=[0, 0.9], thr_interval=0.02):

    if num_classes == 2:
        eval_list = eval_single_class(score_list, mask_list, thr_ranges=thr_ranges, thr_interval=thr_interval)

        if print_on_screen:
            text = 'Train' if train else 'Val'
            print('| {} Thr: {:.4f}'.format(text, eval_list[0]).ljust(print_num, ' '), '|')
            print('| {}  Jc: {:.4f}'.format(text, eval_list[1]).ljust(print_num, ' '), '|')
            print('| {}  Dc: {:.4f}'.format(text, eval_list[2]).ljust(print_num, ' '), '|')
    else:
        # TODO
        pass
        # eval_list = evaluate_multi(score_list, mask_list)

        # np.set_printoptions(precision=4, suppress=True)
        # print('| Train  Jc: {}'.format(eval_list[0]).ljust(print_num, ' '), '|')
        # print('| Train  Dc: {}'.format(eval_list[2]).ljust(print_num, ' '), '|')
        # print('| Train mJc: {:.4f}'.format(eval_list[1]).ljust(print_num, ' '), '|')
        # print('| Train mDc: {:.4f}'.format(eval_list[3]).ljust(print_num, ' '), '|')
        # train_m_jc = eval_list[1]

    return eval_list


def eval_single_class(y_scores, y_true, thr_ranges=[0, 0.9], thr_interval=0.02):

    y_scores = torch.softmax(y_scores, dim=1)
    y_scores = y_scores[:, 1, ...].cpu().detach().numpy().flatten()
    y_true = y_true.data.cpu().numpy().flatten()

    thresholds = np.arange(thr_ranges[0], thr_ranges[1], thr_interval)
    jaccard = np.zeros(len(thresholds))
    dice = np.zeros(len(thresholds))
    y_true.astype(np.int8)

    for indy in range(len(thresholds)):
        threshold = thresholds[indy]
        y_pred = (y_scores > threshold).astype(np.int8)

        sum_area = (y_pred + y_true)
        tp = float(np.sum(sum_area == 2))
        union = np.sum(sum_area == 1)
        jaccard[indy] = tp / float(union + tp)
        dice[indy] = 2 * tp / float(union + 2 * tp)

    thred_indx = np.argmax(jaccard)
    m_jaccard = jaccard[thred_indx]
    m_dice = dice[thred_indx]

    return thresholds[thred_indx], m_jaccard, m_dice


def eval_distance(mask_list, seg_result_list, num_classes):

    print_num = 42 + (num_classes - 3) * 7
    print_num_minus = print_num - 2

    assert len(mask_list) == len(seg_result_list)

    hd_list = []
    sd_list = []

    if num_classes == 2:
        for i in range(len(mask_list)):
            if np.any(seg_result_list[i]) and np.any(mask_list[i]):
                hd_ = hd95(seg_result_list[i], mask_list[i])
                sd_ = assd(seg_result_list[i], mask_list[i])
                hd_list.append(hd_)
                sd_list.append(sd_)

        hd = np.mean(hd_list)
        sd = np.mean(sd_list)

        print('| Hd: {:.4f}'.format(hd).ljust(print_num_minus, ' '), '|')
        print('| Sd: {:.4f}'.format(sd).ljust(print_num_minus, ' '), '|')

    else:
        for cls in range(num_classes-1):
            hd_list_ = []
            sd_list_ = []

            for i in range(len(mask_list)):

                mask_list_ = mask_list[i].copy()
                seg_result_list_ = seg_result_list[i].copy()

                mask_list_[mask_list[i] != (cls + 1)] = 0
                seg_result_list_[seg_result_list[i] != (cls + 1)] = 0

                if np.any(seg_result_list_) and np.any(mask_list_):
                    hd_ = hd95(seg_result_list_, mask_list_)
                    sd_ = assd(seg_result_list_, mask_list_)
                    hd_list_.append(hd_)
                    sd_list_.append(sd_)

            hd = np.mean(hd_list_)
            sd = np.mean(sd_list_)

            hd_list.append(hd)
            sd_list.append(sd)

        hd_list = np.array(hd_list)
        sd_list = np.array(sd_list)

        m_hd = np.mean(hd_list)
        m_sd = np.mean(sd_list)

        np.set_printoptions(precision=4, suppress=True)
        print('|  Hd: {}'.format(hd_list).ljust(print_num_minus, ' '), '|')
        print('|  Sd: {}'.format(sd_list).ljust(print_num_minus, ' '), '|')
        print('| mHd: {:.4f}'.format(m_hd).ljust(print_num_minus, ' '), '|')
        print('| mSd: {:.4f}'.format(m_sd).ljust(print_num_minus, ' '), '|')

    print('-' * print_num) 

    # TODO multiclass
    return (hd, sd) if num_classes == 2 else None


def eval_pixel(mask_list, seg_result_list, num_classes):

    c = confusion_matrix(mask_list, seg_result_list)

    hist_diag = np.diag(c)
    hist_sum_0 = c.sum(axis=0)
    hist_sum_1 = c.sum(axis=1)

    jaccard = hist_diag / (hist_sum_1 + hist_sum_0 - hist_diag)
    dice = 2 * hist_diag / (hist_sum_1 + hist_sum_0)

    print_num = 42 + (num_classes - 3) * 7
    print_num_minus = print_num - 2

    print('-' * print_num)
    if num_classes > 2:
        m_jaccard = np.nanmean(jaccard)
        m_dice = np.nanmean(dice)
        np.set_printoptions(precision=4, suppress=True)
        print('|  Jc: {}'.format(jaccard).ljust(print_num_minus, ' '), '|')
        print('|  Dc: {}'.format(dice).ljust(print_num_minus, ' '), '|')
        print('| mJc: {:.4f}'.format(m_jaccard).ljust(print_num_minus, ' '), '|')
        print('| mDc: {:.4f}'.format(m_dice).ljust(print_num_minus, ' '), '|')
    else:
        print('| Jc: {:.4f}'.format(jaccard[1]).ljust(print_num_minus, ' '), '|')
        print('| Dc: {:.4f}'.format(dice[1]).ljust(print_num_minus, ' '), '|')

    return (jaccard[1], dice[1]) if num_classes == 2 else None


# def test_2d(pred_paths, mask_paths, num_classes=2, resize_shape=(128, 128)):
#     pred_list = []
#     mask_list = []

#     pred_flatten_list = []
#     mask_flatten_list = []

#     for num, i in enumerate(os.listdir(pred_paths)):
#         pred_path = os.path.join(pred_paths, i)
#         mask_path = os.path.join(mask_paths, i)

#         pred = Image.open(pred_path)
#         pred = np.array(pred).astype(np.int8)

#         mask = Image.open(mask_path)
#         mask = np.array(mask)

#         resize = A.Resize(resize_shape[1], resize_shape[0], p=1)(image=pred, mask=mask)
#         mask = resize['mask']
#         pred = resize['image']

#         pred_list.append(pred)
#         mask_list.append(mask)

#         if num == 0:
#             pred_flatten_list = pred.flatten()
#             mask_flatten_list = mask.flatten()
#         else:
#             pred_flatten_list = np.append(pred_flatten_list, pred.flatten())
#             mask_flatten_list = np.append(mask_flatten_list, mask.flatten())

#         num += 1

#     eval_pixel_list = eval_pixel(mask_flatten_list, pred_flatten_list, num_classes)
#     eval_distance_list = eval_distance(mask_list, pred_list, num_classes)

#     return eval_pixel_list, eval_distance_list
