from PIL import Image
import os
from torch.utils.data.dataset import Dataset
import random
import torch
import numpy as np

def toTensor(img):
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255)
    # return img

class TrainSetLoader(Dataset):
    def __init__(self, cfg):
        super(TrainSetLoader, self).__init__()
        self.degraded_path = cfg.dataset_path + '/' + 'hazy'
        self.gt_path = cfg.dataset_path + '/' + 'clear'
        self.degraded_image_list = list()
        self.gt_image_list = list()
        
        for image in os.listdir(self.degraded_path):
            self.degraded_image_list.append(os.path.join(self.degraded_path, image))
            self.gt_image_list.append(os.path.join(self.gt_path, image))

        self.degraded_image_list.sort()
        self.gt_image_list.sort()


    def __getitem__(self, index):
        img_degraded = Image.open(self.degraded_image_list[index])
        img_GT = Image.open(self.gt_image_list[index])

        img_degraded = np.array(img_degraded, dtype=np.float32)
        img_GT = np.array(img_GT, dtype=np.float32)

        img_degraded, img_GT = detection(img_degraded), detection(img_GT)

        img_degraded, img_GT = np.ascontiguousarray(img_degraded), np.ascontiguousarray(img_GT)
        img_degraded, img_GT = toTensor(img_degraded), toTensor(img_GT)

        return img_degraded, img_GT

    def __len__(self):
        return len(self.degraded_image_list)

def detection(img):
    if len(img.shape) == 2:
        img = np.stack([img] * 3, 2)
    else:
        img = img
    return img


def augmentation(hr_image_left, hr_image_right, lr_image_left, lr_image_right):
        if random.random()<0.5:     # flip horizonly
            lr_image_left_ = lr_image_right[:, ::-1, :]
            lr_image_right_ = lr_image_left[:, ::-1, :]
            hr_image_left_ = hr_image_right[:, ::-1, :]
            hr_image_right_ = hr_image_left[:, ::-1, :]
            lr_image_left, lr_image_right = lr_image_left_, lr_image_right_
            hr_image_left, hr_image_right = hr_image_left_, hr_image_right_

        if random.random()<0.5:     #flip vertically
            lr_image_left = lr_image_left[::-1, :, :]
            lr_image_right = lr_image_right[::-1, :, :]
            hr_image_left = hr_image_left[::-1, :, :]
            hr_image_right = hr_image_right[::-1, :, :]

        return np.ascontiguousarray(hr_image_left), np.ascontiguousarray(hr_image_right), \
                np.ascontiguousarray(lr_image_left), np.ascontiguousarray(lr_image_right)

