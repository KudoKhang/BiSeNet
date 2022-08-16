import cv2
import numpy as np
import os
from tqdm import tqdm

root = './Figaro_1k/'
mode = ['train', 'test', 'val']
total_ratio = []
for m in mode:
    path_image = [name for name in os.listdir(os.path.join(root, m, 'images')) if name.endswith('jpg')]
    for name in path_image:
        img = cv2.imread(os.path.join(root, m, 'images', name))
        h, w = img.shape[:2]
        ration_w_h = w / h
        total_ratio.append(ration_w_h)

avg_ratio = round(sum(total_ratio) / len(total_ratio), 4)
print('Ratio width / height of image is: ', avg_ratio)