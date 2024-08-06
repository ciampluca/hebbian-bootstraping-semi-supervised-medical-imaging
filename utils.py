import numpy as np
import os
import random

import torch

from PIL import Image
import skimage
from skimage.morphology import remove_small_holes
from medpy.metric.binary import hd95, assd
from sklearn.metrics import confusion_matrix
import torchio as tio
import SimpleITK as sitk



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


def save_preds_3d(score_list_val, threshold, name_list_val, path_seg_results, affine_list, num_classes=2):

    if num_classes == 2:
        score_list_val = torch.softmax(score_list_val, dim=1)
        pred_results = score_list_val[:, 1, ...].cpu().detach()
        pred_results[pred_results > threshold] = 1
        pred_results[pred_results <= threshold] = 0
        pred_results = pred_results.type(torch.uint8)
    else:
        pred_results = torch.max(score_list_val, 1)[1].cpu().detach()
        pred_results = pred_results.type(torch.uint8)

    if not os.path.exists(path_seg_results):
            os.makedirs(path_seg_results)
    for i in range(len(name_list_val)):
        output_image = tio.ScalarImage(tensor=pred_results[i].unsqueeze(0), affine=affine_list[i])
        output_image.save(os.path.join(path_seg_results, name_list_val[i]))


def save_test_3d(num_classes, score_test, name_test, threshold, path_seg_results, affine):

    if num_classes == 2:
        score_list_test = torch.softmax(score_test, dim=0)
        pred_results = score_list_test[1, ...].cpu()
        pred_results[pred_results > threshold] = 1
        pred_results[pred_results <= threshold] = 0
        pred_results = pred_results.type(torch.uint8)
    else:
        pred_results = torch.max(score_test, 0)[1].cpu().detach()
        pred_results = pred_results.type(torch.uint8)

    output_image = tio.ScalarImage(tensor=pred_results.unsqueeze(0), affine=affine)
    output_image.save(os.path.join(path_seg_results, name_test))


def compute_epoch_loss(loss, num_batches, print_num, print_num_minus, train=True, print_on_screen=True, unsup_pretrain=False):
    epoch_loss = loss / num_batches['pretrain_unsup'] if unsup_pretrain else loss / num_batches['train_sup']

    if print_on_screen:
        print('-' * print_num)
        text = 'Train Loss' if train else 'Val Loss'
        print('| {}: {:.4f}'.format(text, epoch_loss).ljust(print_num_minus, ' '), '|')
        print('-' * print_num)

    return epoch_loss


def compute_epoch_loss_EM(train_loss_sup, train_loss_cps, train_loss, num_batches, print_num, print_num_minus, print_on_screen=True):
    train_epoch_loss_sup = train_loss_sup / num_batches['train_sup']
    train_epoch_loss_cps = train_loss_cps / num_batches['train_sup']
    train_epoch_loss = train_loss / num_batches['train_sup']

    if print_on_screen:
        print('-' * print_num)
        print('| Train  Sup  Loss: {:.4f}'.format(train_epoch_loss_sup).ljust(print_num_minus, ' '), '|')
        print('| Train Unsup Loss: {:.4f}'.format(train_epoch_loss_cps).ljust(print_num_minus, ' '), '|')
        print('| Train Total Loss: {:.4f}'.format(train_epoch_loss).ljust(print_num_minus, ' '), '|')
        print('-' * print_num)

    return train_epoch_loss_sup, train_epoch_loss_cps, train_epoch_loss


def compute_epoch_loss_MT(train_loss_sup, train_loss_cps, train_loss, num_batches, print_num, print_num_half, print_num_minus, print_on_screen=True):
    train_loss_sup = train_loss_sup / num_batches['train_sup']
    train_epoch_loss_cps = train_loss_cps / num_batches['train_sup']
    train_epoch_loss = train_loss / num_batches['train_sup']

    if print_on_screen:
        print('-' * print_num)
        print('| Train  Sup  Loss: {:.4f}'.format(train_loss_sup).ljust(print_num_half, ' '), '| Train Unsup Loss: {:.4f}'.format(train_epoch_loss_cps).ljust(print_num_half, ' '), '|')
        print('| Train Total Loss: {:.4f}'.format(train_epoch_loss).ljust(print_num_minus, ' '), '|')
        print('-' * print_num)
    
    return train_loss_sup, train_epoch_loss_cps, train_epoch_loss


def compute_epoch_loss_XNet(train_loss_sup_1, train_loss_sup_2, train_loss_cps, train_loss, num_batches, print_num, print_num_half, print_on_screen=True):
    train_epoch_loss_sup1 = train_loss_sup_1 / num_batches['train_sup']
    train_epoch_loss_sup2 = train_loss_sup_2 / num_batches['train_sup']
    train_epoch_loss_cps = train_loss_cps / num_batches['train_sup']
    train_epoch_loss = train_loss / num_batches['train_sup']

    if print_on_screen:
        print('-' * print_num)
        print('| Train Sup Loss 1: {:.4f}'.format(train_epoch_loss_sup1).ljust(print_num_half, ' '), '| Train SUP Loss 2: {:.4f}'.format(train_epoch_loss_sup2).ljust(print_num_half, ' '), '|')
        print('| Train Unsup Loss: {:.4f}'.format(train_epoch_loss_cps).ljust(print_num_half, ' '), '| Train Total Loss: {:.4f}'.format(train_epoch_loss).ljust(print_num_half, ' '), '|')
        print('-' * print_num)

    return train_epoch_loss_sup1, train_epoch_loss_sup2, train_epoch_loss_cps, train_epoch_loss


def compute_val_epoch_loss_MT(val_loss_sup_1, val_loss_sup_2, num_batches, print_num, print_num_half, print_on_screen=True):
    val_epoch_loss_sup1 = val_loss_sup_1 / num_batches['val']
    val_epoch_loss_sup2 = val_loss_sup_2 / num_batches['val']

    if print_on_screen:
        print('-' * print_num)
        print('| Val Sup Loss 1: {:.4f}'.format(val_epoch_loss_sup1).ljust(print_num_half, ' '), '| Val Sup Loss 2: {:.4f}'.format(val_epoch_loss_sup2).ljust(print_num_half, ' '), '|')
        print('-' * print_num)
    
    return val_epoch_loss_sup1, val_epoch_loss_sup2


def print_best_val_metrics(num_classes, best_val_list, print_num):
    if num_classes == 2:
        print('| Best Val Thr: {:.4f}'.format(best_val_list[0]).ljust(print_num, ' '), '|')
    print('| Best Val  Jc: {:.4f}'.format(best_val_list[1]).ljust(print_num, ' '), '|')
    print('| Best Val  Dc: {:.4f}'.format(best_val_list[2]).ljust(print_num, ' '), '|')


def evaluate(num_classes, score_list, mask_list, print_num, print_on_screen=True, train=True, thr_ranges=[0, 0.9], thr_interval=0.02):

    if num_classes == 2:
        eval_list = eval_single_class(score_list, mask_list, thr_ranges=thr_ranges, thr_interval=thr_interval)
    else:
        eval_list = eval_multi_class(score_list, mask_list)

    if print_on_screen:
        text = 'Train' if train else 'Val'
        if num_classes == 2:
            print('| {} Thr: {:.4f}'.format(text, eval_list[0]).ljust(print_num, ' '), '|')
        print('| {}  Jc: {:.4f}'.format(text, eval_list[1]).ljust(print_num, ' '), '|')
        print('| {}  Dc: {:.4f}'.format(text, eval_list[2]).ljust(print_num, ' '), '|')

    return eval_list


def evaluate_val_MT(num_classes, score_list_val1, score_list_val2, mask_list_val, print_num, print_on_screen=True, thr_ranges=[0, 0.9], thr_interval=0.02):
 
    if num_classes == 2:
        eval_list1 = eval_single_class(score_list_val1, mask_list_val, thr_ranges=thr_ranges, thr_interval=thr_interval)
        eval_list2 = eval_single_class(score_list_val2, mask_list_val, thr_ranges=thr_ranges, thr_interval=thr_interval)
    else:
        eval_list1 = eval_multi_class(score_list_val1, mask_list_val)
        eval_list2 = eval_multi_class(score_list_val2, mask_list_val)       

    if print_on_screen:
        if num_classes == 2:
            print('| Val Thr 1: {:.4f}'.format(eval_list1[0]).ljust(print_num, ' '), '| Val Thr 2: {:.4f}'.format(eval_list2[0]).ljust(print_num, ' '), '|')
        print('| Val  Jc 1: {:.4f}'.format(eval_list1[1]).ljust(print_num, ' '), '| Val  Jc 2: {:.4f}'.format(eval_list2[1]).ljust(print_num, ' '), '|')
        print('| Val  Dc 1: {:.4f}'.format(eval_list1[2]).ljust(print_num, ' '), '| Val  Dc 2: {:.4f}'.format(eval_list2[2]).ljust(print_num, ' '), '|')
        
    return eval_list1, eval_list2


# TODO probably we can use just one function between this and evaluate_val_MT
def evaluate_XNet(num_classes, score_list_train1, score_list_train2, mask_list_train, print_num, print_on_screen=True, train=True, thr_ranges=[0, 0.9], thr_interval=0.02):

    if num_classes == 2:
        eval_list1 = eval_single_class(score_list_train1, mask_list_train, thr_ranges=thr_ranges, thr_interval=thr_interval)
        eval_list2 = eval_single_class(score_list_train2, mask_list_train, thr_ranges=thr_ranges, thr_interval=thr_interval)
    else:
        eval_list1 = eval_multi_class(score_list_train1, mask_list_train)
        eval_list2 = eval_multi_class(score_list_train2, mask_list_train)

    if print_on_screen:
        text = 'Train' if train else 'Val'
        if num_classes == 2:
            print('| {} Thr 1: {:.4f}'.format(text, eval_list1[0]).ljust(print_num, ' '), '| {} Thr 2: {:.4f}'.format(text, eval_list2[0]).ljust(print_num, ' '), '|')
        print('| {}  Jc 1: {:.4f}'.format(text, eval_list1[1]).ljust(print_num, ' '), '| {} Jc 2: {:.4f}'.format(text, eval_list2[1]).ljust(print_num, ' '), '|')
        print('| {}  Dc 1: {:.4f}'.format(text, eval_list1[2]).ljust(print_num, ' '), '| {}  Dc 2: {:.4f}'.format(text, eval_list2[2]).ljust(print_num, ' '), '|')

    return eval_list1, eval_list2


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


def eval_multi_class(y_scores, y_true):
    
    y_scores = torch.softmax(y_scores, dim=1)
    y_pred = torch.max(y_scores, 1)[1]
    y_pred = y_pred.data.cpu().numpy().flatten()
    y_true = y_true.data.cpu().numpy().flatten()

    hist = confusion_matrix(y_true, y_pred)

    hist_diag = np.diag(hist)
    hist_sum_0 = hist.sum(axis=0)
    hist_sum_1 = hist.sum(axis=1)

    jaccard = hist_diag / (hist_sum_1 + hist_sum_0 - hist_diag)
    m_jaccard = np.nanmean(jaccard)
    dice = 2 * hist_diag / (hist_sum_1 + hist_sum_0)
    m_dice = np.nanmean(dice)

    return None, m_jaccard, m_dice


def evaluate_distance(num_classes, score_list, mask_list, thr_ranges=[0, 0.9], thr_interval=0.02):

    print_num = 42 + (num_classes - 3) * 7
    print_num_minus = print_num - 2

    assert len(mask_list) == len(score_list)

    score_list = torch.softmax(score_list, dim=1)
    score_list = score_list[:, 1, ...].cpu().detach().numpy()
    mask_list = mask_list.cpu().detach().numpy()

    hd_list = []
    sd_list = []

    if num_classes == 2:
        thresholds = np.arange(thr_ranges[0], thr_ranges[1], thr_interval)
        hd_ = np.zeros(len(thresholds))
        sd_ = np.zeros(len(thresholds))

        for i in range(len(mask_list)):
            for indy in range(len(thresholds)):
                threshold = thresholds[indy]
                score_list[i] = (score_list[i] > threshold).astype(np.int8)

                if np.any(score_list[i]) and np.any(mask_list[i] != 0):
                    hd_[indy] = hd95(score_list[i], mask_list[i])
                    sd_[indy] = assd(score_list[i], mask_list[i])

            hd_list.append(np.min(hd_))
            sd_list.append(np.min(sd_))

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
                seg_result_list_ = score_list[i].copy()

                mask_list_[mask_list[i] != (cls + 1)] = 0
                seg_result_list_[score_list[i] != (cls + 1)] = 0

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

    return (hd, sd) if num_classes == 2 else (m_hd, m_sd)


def eval_distance_offline(mask_list, seg_result_list, num_classes):
    print_num = 42 + (num_classes - 3) * 7
    print_num_minus = print_num - 2

    assert len(mask_list) == len(seg_result_list)
    
    if num_classes == 2:
        hd_list = []
        sd_list = []
        
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
        hd_list = []
        sd_list = []

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

    return (hd, sd) if num_classes == 2 else (m_hd, m_sd)


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
        #print('|  Jc: {}'.format(jaccard).ljust(print_num_minus, ' '), '|')
        #print('|  Dc: {}'.format(dice).ljust(print_num_minus, ' '), '|')
        print('| mJc: {:.4f}'.format(m_jaccard).ljust(print_num_minus, ' '), '|')
        print('| mDc: {:.4f}'.format(m_dice).ljust(print_num_minus, ' '), '|')
    else:
        print('| Jc: {:.4f}'.format(jaccard[1]).ljust(print_num_minus, ' '), '|')
        print('| Dc: {:.4f}'.format(dice[1]).ljust(print_num_minus, ' '), '|')

    return (jaccard[1], dice[1]) if num_classes == 2 else (m_jaccard, m_dice)


def postprocess_3d_pred(dataset_name, pred_path, save_path, fill_hole_thr=500):
    
    def save_max_objects(image):
        labeled_image = skimage.measure.label(image)
        labeled_list = skimage.measure.regionprops(labeled_image)
        box = []
        
        for i in range(len(labeled_list)):
            box.append(labeled_list[i].area)
            label_num = box.index(max(box)) + 1

        if not np.any(labeled_image):
            return labeled_image

        labeled_image[labeled_image != label_num] = 0
        labeled_image[labeled_image == label_num] = 1

        return labeled_image
    
    if dataset_name == 'Atrial':
        for i in os.listdir(pred_path):
            pred = sitk.ReadImage(os.path.join(pred_path, i))
            pred = sitk.GetArrayFromImage(pred)

            pred = pred.astype(bool)
            pred = remove_small_holes(pred, fill_hole_thr)
            pred = pred.astype(np.uint8)

            pred = save_max_objects(pred)
            pred = pred.astype(np.uint8)

            pred = sitk.GetImageFromArray(pred)
            sitk.WriteImage(pred, os.path.join(save_path, i))

    elif dataset_name == 'LiTS':
        for i in os.listdir(pred_path):
            pred = sitk.ReadImage(os.path.join(pred_path, i))
            pred = sitk.GetArrayFromImage(pred)

            pred_ = pred.copy()
            pred_[pred != 0] = 1

            pred_ = pred_.astype(bool)
            pred_ = remove_small_holes(pred_, fill_hole_thr)
            pred_ = pred_.astype(np.uint8)

            pred_ = save_max_objects(pred_)
            pred_[(pred_ == 1) & (pred == 2)] = 2
            pred_ = pred_.astype(np.uint8)

            pred_ = sitk.GetImageFromArray(pred_)
            sitk.WriteImage(pred_, os.path.join(save_path, i))

    else:
        print("Dataset not implemented")


def offline_eval(pred_path, mask_path, if_3D=True, num_classes=2):
    pred_list = []
    mask_list = []

    pred_flatten_list = []
    mask_flatten_list = []

    num = 0

    for i in os.listdir(pred_path):

        if if_3D:
            pred = sitk.ReadImage(os.path.join(pred_path, i))
            pred = sitk.GetArrayFromImage(pred)

            mask = sitk.ReadImage(os.path.join(mask_path, i))
            mask = sitk.GetArrayFromImage(mask)
            mask[mask==255] = 1

        pred_list.append(pred)
        mask_list.append(mask)

        if num == 0:
            pred_flatten_list = pred.flatten()
            mask_flatten_list = mask.flatten()
        else:
            pred_flatten_list = np.append(pred_flatten_list, pred.flatten())
            mask_flatten_list = np.append(mask_flatten_list, mask.flatten())

        num += 1

    jaccard, dice = eval_pixel(mask_flatten_list, pred_flatten_list, num_classes)
    hd, sd = eval_distance_offline(mask_list, pred_list, num_classes)

    return {'jaccard': jaccard, 'dice': dice, 'hd': hd, 'sd': sd}


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def update_ema_variables_3d(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)