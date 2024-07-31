import numpy as np
import os
import argparse
import shutil
import random

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default='data/LiTS/train/image')
    parser.add_argument('--mask_path', default='data/LiTS/train/mask')
    parser.add_argument('--save_path', default='data/LiTS/val')
    parser.add_argument('--amount', default=30)
    parser.add_argument('--random_seed', default=10)
    args = parser.parse_args()

    random.seed(args.random_seed)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    save_image_path = args.save_path + '/image'
    save_mask_path = args.save_path + '/mask'
    if not os.path.exists(save_image_path):
        os.mkdir(save_image_path)
    if not os.path.exists(save_mask_path):
        os.mkdir(save_mask_path)

    image_path_list = os.listdir(args.image_path)

    image_path_list = random.sample(image_path_list, args.amount)

    for i in image_path_list:
        shutil.move(os.path.join(args.image_path, i), save_image_path)
        shutil.move(os.path.join(args.mask_path, i), save_mask_path)
