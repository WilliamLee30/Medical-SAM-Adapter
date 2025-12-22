import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils import random_box, random_click
import random


class Lucchi(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'train', prompt = 'click'):

        self.data_path = data_path
        self.mode = mode

        self.input_image_path = os.path.join(self.data_path, self.mode, "input_image")
        self.input_image_list = sorted(os.listdir(self.input_image_path))

        # self.binary_mask_path =  os.path.join(self.data_path, self.mode, "binary_mask")
        self.binary_mask_path =  os.path.join(self.data_path, self.mode, "single_point_0logit")
        self.binary_mask_list = sorted(os.listdir(self.binary_mask_path))

        self.centroid_path =  os.path.join(self.data_path, self.mode, "point_label", "centroid")
        self.centroid_list = sorted(os.listdir(self.centroid_path))

        self.centroid_label_path =  os.path.join(self.data_path, self.mode, "point_label", "label")
        self.centroid_label_list = sorted(os.listdir(self.centroid_label_path))


        self.prompt = prompt

        self.transform = transform if transform is not None else transforms.Compose([
            transforms.ToTensor(), # 将 PIL Image 转换为 [0, 1] 的 Tensor
            transforms.Lambda(lambda x: x * 255)
        ])
        
        self.transform_msk = transform_msk if transform_msk is not None else transforms.ToTensor()

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

        # 仅选取一个点和标签
        n = len(centroid_list)
        random_index = random.randrange(n)
        single_pt = centroid_list[random_index]
        single_label = centroid_label_list[random_index]
        single_pt_tensor = torch.tensor(single_pt, dtype=torch.int64) 
        single_label_tensor = torch.tensor(single_label, dtype=torch.int64)

        # 转换为tensor
        img_tensor = self.transform(img)
        mask_tensor = self.transform_msk(mask).int()

                
        name = input_image_name.split(".png")[0]
        image_meta_dict = {'filename_or_obj':name}

        return {
            'image':img_tensor,
            'label': mask_tensor,
            'p_label':single_label_tensor,
            'pt':single_pt_tensor,
            'image_meta_dict':image_meta_dict,
        }