import os
import cv2
import time
import random
import numpy as np

def ensure_dir(path):
    print(path)
    if not os.path.isdir(path):
        try:
            sleeptime = random.randint(0, 3)
            time.sleep(sleeptime)
            os.makedirs(path)
        except:
            print('conflict !!!')

def get_class_colors(*args):
    return np.array([[128, 64, 128], [244, 35, 232], [70, 70, 70],
            [102, 102, 156], [190, 153, 153], [153, 153, 153],
            [250, 170, 30], [220, 220, 0], [107, 142, 35],
            [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0],
            [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
            [0, 0, 230], [119, 11, 32], [0, 0, 0]])

mode = 'val'
root = 'segmentation/' + mode
target_dir = 'colored_gt/' + mode
ensure_dir(target_dir)

files = os.listdir(root)
color_map = get_class_colors()
for file in files:
	if 'png' in file:
		print(file)
		gt = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)
		gt[gt==255] = 19
		colored_gt = color_map[gt]
		cv2.imwrite(os.path.join(target_dir, file.replace('gtFine', 'gt_colored')), colored_gt)