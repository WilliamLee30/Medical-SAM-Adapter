import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import random_box, random_click


class Lucchi(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'train', prompt = 'click'):

        self.data_path = data_path
        self.mode = mode

        self.input_image_path = os.path.join(self.data_path, self.mode, "input_image")
        self.input_image_list = os.listdir(self.input_image_path)

        self.binary_mask_path =  os.path.join(self.data_path, self.mode, "binary_mask")
        self.binary_mask_list = os.listdir(self.binary_mask_path)

        self.centroid_path =  os.path.join(self.data_path, self.mode, "point_label", "centroid")
        self.centroid_list = os.listdir(self.centroid_path)

        self.centroid_label_path =  os.path.join(self.data_path, self.mode, "point_label", "label")
        self.centroid_label_list = os.listdir(self.centroid_label_path)


        self.prompt = prompt

        # 对图像和mask的分辨率进行resize-deprecated-20251209
        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.input_image_list)

    def __getitem__(self, index):
        """Get the images"""

        # 输入图像和分割mask
        input_image_name = self.input_image_list[index]
        img_path = os.path.join(self.input_image_path, input_image_name)
        
        binary_mask_name = self.binary_mask_list[index]
        msk_path = os.path.join(self.binary_mask_path, binary_mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')


        # 点和标签
        centroid_name = self.centroid_list[index]
        centroid_path = os.path.join(self.centroid_path, centroid_name)

        centroid_label_name = self.centroid_label_list[index]
        centroid_label_path = os.path.join(self.centroid_label_path, centroid_label_name)

        centroid_list_array = np.load(centroid_path)
        centroid_list = centroid_list_array.tolist()

        centroid_label_list_array = np.load(centroid_label_path)
        centroid_label_list = centroid_label_list_array.tolist()

                
        name = input_image_name.split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':centroid_label_list,
            'pt':centroid_list,
            'image_meta_dict':image_meta_dict,
        }