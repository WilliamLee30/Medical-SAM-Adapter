import numpy as np
from PIL import Image
import cv2 

def random_click(mask, point_labels = 1):
    # check if all masks are black
    max_label = max(set(mask.flatten()))
    if max_label == 0:
        point_labels = max_label
    # max agreement position
    indices = np.argwhere(mask == max_label) 
    indices = indices[:, ::-1].copy()
    return point_labels, indices[np.random.randint(len(indices))]


mask = Image.open("/mnt/nasv3/liyuanwei/ReseachTask/mitochrondria_segmentation/data/Lucchi++_crop512/train/binary_mask/0_patch_0.png").convert('L')

point_label, pt = random_click(np.array(mask) / 255, 1)

print(point_label)

print(pt)