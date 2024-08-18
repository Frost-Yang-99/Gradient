import os
import random
import copy
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor

from image_utils import random_augmentation, crop_img
from degradation_utils import Degradation
from torchvision.transforms import ToTensor
from torchvision import transforms
import torch

class TrainDataset(Dataset):
    def __init__(self, args):
        super(TrainDataset, self).__init__()
        self.args = args
        self.noise_list = []
        self.rs_ids = []
        self.hazy_ids = []
        self.D = Degradation(args)
        self.de_temp = 0
        self.de_type = self.args.de_type
        self.de_dict = {'denoise': 0, 'derain': 1, 'dehaze': 2}

        self._init_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'denoise' in self.de_type:
            self._init_clean_ids()
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'dehaze' in self.de_type:
            self._init_hazy_ids()

        random.shuffle(self.de_type)

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.denoise_dir + 'noise')
        self.noise_list += [self.args.denoise_dir + 'noise/' + id_ for id_ in name_list]
        self.noise_counter = 0
        self.num_noise = len(self.noise_list)

    def _init_hazy_ids(self):
        hazy = os.listdir(self.args.dehaze_dir + 'hazy')
        self.hazy_ids += [self.args.dehaze_dir + 'hazy/' + id for id in hazy]
        self.hazy_counter = 0
        self.num_hazy = len(self.hazy_ids)

    def _init_rs_ids(self):
        rs = os.listdir(self.args.derain_dir + 'rainy')
        self.rs_ids += [self.args.derain_dir + 'rainy/' + id for id in rs]
        self.rl_counter = 0
        self.num_rl = len(self.rs_ids)


    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_nonnoise_name(self, noise_name):
        gt_name = noise_name.split("noise")[0] + 'gt' + noise_name.split("noise")[-1]
        return gt_name

    def _get_gt_name(self, rainy_name):
        gt_name = rainy_name.split("rainy")[0] + 'gt/norain-' + rainy_name.split('rain-')[-1]
        return gt_name

    def _get_nonhazy_name(self, hazy_name):
        gt_name = hazy_name.split("hazy")[0] + 'gt' + hazy_name.split("hazy")[-1]
        return gt_name


    def __getitem__(self, _):
        de_id = self.de_dict[self.de_type[self.de_temp]]

        if de_id == 0:
            degrad_img = crop_img(np.array(Image.open(self.noise_list[self.noise_counter]).convert('RGB')), base=16)
            clean_name = self._get_nonnoise_name(self.noise_list[self.noise_counter])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            self.noise_counter = (self.noise_counter + 1) % self.num_noise
            if self.noise_counter == 0:
                random.shuffle(self.noise_list)

        elif de_id == 1:
            degrad_img = crop_img(np.array(Image.open(self.rs_ids[self.rl_counter]).convert('RGB')), base=16)
            clean_name = self._get_gt_name(self.rs_ids[self.rl_counter])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            self.rl_counter = (self.rl_counter + 1) % self.num_rl
            if self.rl_counter == 0:
                random.shuffle(self.rs_ids)

        elif de_id == 2:
            degrad_img = crop_img(np.array(Image.open(self.hazy_ids[self.hazy_counter]).convert('RGB')), base=16)
            clean_name = self._get_nonhazy_name(self.hazy_ids[self.hazy_counter])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            self.hazy_counter = (self.hazy_counter + 1) % self.num_hazy
            if self.hazy_counter == 0:
                random.shuffle(self.hazy_ids)

        degrad_patch_1, clean_patch_1 = random_augmentation(*self._crop_patch(degrad_img, clean_img))

        degrad_patch_1, clean_patch_1 = self.toTensor(degrad_patch_1), self.toTensor(clean_patch_1)


        self.de_temp = (self.de_temp + 1) % len(self.de_type)
        if self.de_temp == 0:
            random.shuffle(self.de_type)

        return [clean_name, de_id], degrad_patch_1, clean_patch_1

    def __len__(self):
        return 1000 * len(self.args.de_type)


class DenoiseTestDataset(Dataset):
    def __init__(self, args):
        super(DenoiseTestDataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.sigma = 15

        self._init_clean_ids()

        self.toTensor = ToTensor()

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.denoise_path)
        self.clean_ids += [self.args.denoise_path + id_ for id_ in name_list]

        self.num_clean = len(self.clean_ids)

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def set_sigma(self, sigma):
        self.sigma = sigma

    def __getitem__(self, clean_id):
        clean_img = crop_img(np.array(Image.open(self.clean_ids[clean_id]).convert('RGB')), base=16)
        clean_name = self.clean_ids[clean_id].split("/")[-1].split('.')[0]

        noisy_img, _ = self._add_gaussian_noise(clean_img)
        clean_img, noisy_img = self.toTensor(clean_img), self.toTensor(noisy_img)

        return [clean_name], noisy_img, clean_img

    def __len__(self):
        return self.num_clean


class DerainDehazeDataset(Dataset):
    def __init__(self, args, task="derain"):
        super(DerainDehazeDataset, self).__init__()
        self.ids = []
        self.task_idx = 0
        self.args = args

        self.task_dict = {'derain': 0, 'dehaze': 1}
        self.toTensor = ToTensor()

        self.set_dataset(task)

    def _init_input_ids(self):
        if self.task_idx == 0:
            self.ids = []
            name_list = os.listdir(self.args.derain_path + 'input/')
            self.ids += [self.args.derain_path + 'input/' + id_ for id_ in name_list]
        elif self.task_idx == 1:
            self.ids = []
            name_list = os.listdir(self.args.dehaze_path + 'input/')
            self.ids += [self.args.dehaze_path + 'input/' + id_ for id_ in name_list]

        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        if self.task_idx == 0:
            gt_name = degraded_name.replace("input", "target")
        elif self.task_idx == 1:
            dir_name = degraded_name.split("input")[0] + 'target/'
            name = degraded_name.split('/')[-1].split('_')[0] + '.png'
            gt_name = dir_name + name
        return gt_name

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)

        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = degraded_path.split('/')[-1][:-4]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length


class TestSpecificDataset(Dataset):
    def __init__(self, args):
        super(TestSpecificDataset, self).__init__()
        self.args = args
        self.degraded_ids = []
        self._init_clean_ids(args.test_path)

        self.toTensor = ToTensor()

    def _init_clean_ids(self, root):
        name_list = os.listdir(root)
        self.degraded_ids += [root + id_ for id_ in name_list]

        self.num_img = len(self.degraded_ids)

    def __getitem__(self, idx):
        degraded_img = crop_img(np.array(Image.open(self.degraded_ids[idx]).convert('RGB')), base=16)
        name = self.degraded_ids[idx].split('/')[-1][:-4]

        degraded_img = self.toTensor(degraded_img)

        return [name], degraded_img

    def __len__(self):
        return self.num_img
