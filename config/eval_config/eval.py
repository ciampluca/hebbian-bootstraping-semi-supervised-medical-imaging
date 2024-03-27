import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import directed_hausdorff
import torch




def evaluate_multi(y_scores, y_true):

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

    return jaccard, m_jaccard, dice, m_dice




