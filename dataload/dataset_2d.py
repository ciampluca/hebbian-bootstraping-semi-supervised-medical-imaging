import os
import numpy as np
import math
import random

from PIL import Image
import pywt

import torch
from torch.utils.data import Dataset



class dataset_itn(Dataset):
    def __init__(self, data_dir, input1, augmentation_1, normalize_1, sup=True, regime=100, seed=0, **kwargs):
        super(dataset_itn, self).__init__()

        img_paths_1 = []
        mask_paths = []

        image_dir_1 = data_dir + '/' + input1

        if sup:
            mask_dir = data_dir + '/mask'

        for image in os.listdir(image_dir_1):
            image_path_1 = os.path.join(image_dir_1, image)
            img_paths_1.append(image_path_1)

            if sup:
                mask_path = os.path.join(mask_dir, image)
                mask_paths.append(mask_path)

        if sup:
            assert len(img_paths_1) == len(mask_paths)

        if regime < 100:
            len_img_paths = len(img_paths_1)
            num_images = math.ceil((len_img_paths / 100) * regime)

            shuffled_img_paths_1 = img_paths_1.copy()
            random.Random(seed).shuffle(shuffled_img_paths_1)
            if sup:
                regime_img_paths_1 = shuffled_img_paths_1[:num_images]
                indices = [i for i in range(len(img_paths_1)) if img_paths_1[i] in regime_img_paths_1]
            else:
                regime_img_paths_1 = shuffled_img_paths_1[num_images:]
            img_paths_1 = sorted(regime_img_paths_1)

            if sup:
                regime_mask_paths = [mask_paths[i] for i in indices]
                mask_paths = sorted(regime_mask_paths)

        self.img_paths_1 = img_paths_1
        self.mask_paths = mask_paths
        self.augmentation_1 = augmentation_1
        self.normalize_1 = normalize_1
        self.sup = sup
        self.kwargs = kwargs

    def __getitem__(self, index):

        img_path_1 = self.img_paths_1[index]
        img_1 = Image.open(img_path_1)
        img_1 = np.array(img_1)

        if self.sup:
            mask_path = self.mask_paths[index]
            mask = Image.open(mask_path)
            mask = np.array(mask)
            mask[mask > 0] = 1
            mask = mask.astype(np.uint8)

            augment_1 = self.augmentation_1(image=img_1, mask=mask)
            img_1 = augment_1['image']
            mask_1 = augment_1['mask']

            normalize_1 = self.normalize_1(image=img_1, mask=mask_1)
            img_1 = normalize_1['image']
            mask_1 = normalize_1['mask']
            mask_1 = mask_1.long()

            sampel = {'image': img_1, 'mask': mask_1, 'ID': os.path.split(mask_path)[1]}

        else:
            augment_1 = self.augmentation_1(image=img_1)
            img_1 = augment_1['image']
            normalize_1 = self.normalize_1(image=img_1)
            img_1 = normalize_1['image']

            sampel = {'image': img_1, 'ID': os.path.split(img_path_1)[1]}

        return sampel

    def __len__(self):
        return len(self.img_paths_1)


def imagefloder_itn(data_dir, input1, data_transform_1, data_normalize_1, sup=True, regime=100, **kwargs):
    dataset = dataset_itn(data_dir=data_dir,
                           input1=input1,
                           augmentation_1=data_transform_1,
                           normalize_1=data_normalize_1,
                           sup=sup,
                           regime=regime,
                           **kwargs
                           )
    return dataset
