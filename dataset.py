from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize, Compose
from os import listdir, path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import save_image


def normalize(x):
    rgb_mean = torch.tensor([0.4931, 0.4950, 0.4936]).cuda().view(1, 3, 1, 1)
    rgb_std = torch.tensor([0.2774, 0.2877, 0.3138]).cuda().view(1, 3, 1, 1)
    y = (x - rgb_mean) / rgb_std
    return y


def denormalize(x):
    rgb_mean = torch.tensor([0.4931, 0.4950, 0.4936]).cuda().view(1, 3, 1, 1)
    rgb_std = torch.tensor([0.2774, 0.2877, 0.3138]).cuda().view(1, 3, 1, 1)
    y = x * rgb_std + rgb_mean
    return y


def denormalize_error(x):
    inv_normalize_rgb = Normalize([-0.4931 / 0.2774, -0.4950 / 0.2877, -0.4936 / 0.3138],
                                  [1 / 0.2774, 1 / 0.2877, 1 / 0.3138])

    for i in range(x.size(0)):
        x[i] = inv_normalize_rgb(x[i])
    return x


class Raw2RgbDataset(Dataset):
    def __init__(self, raw_dir, rgb_dir=None, rgb_norm=True, full_res=False):
        self.raw_dir = raw_dir
        self.rgb_dir = rgb_dir
        self.raw_file_list = listdir(raw_dir)
        self.to_tensor = ToTensor()
        self.normalize_raw = Normalize([0.1425, 0.1829, 0.1137, 0.1830], [0.0954, 0.1350, 0.0610, 0.1349])
        self.normalize_rgb = Normalize([0.4931, 0.4950, 0.4936], [0.2774, 0.2877, 0.3138])
        self.inv_normalize_rgb = Normalize([-0.4931/0.2774, -0.4950/0.2877, -0.4936/0.3138],
                                           [1/0.2774, 1/0.2877, 1/0.3138])
        self.rgb_norm = rgb_norm
        self.full_res = full_res

    def __len__(self):
        return len(self.raw_file_list)

    def __getitem__(self, idx):
        if self.rgb_dir is None:
            if self.full_res:
                img_name = str(idx + 1)
            else:
                img_name = str(idx)
        else:
            img_name = self.raw_file_list[idx].split('.')[0]

        raw_img_path = path.join(self.raw_dir, img_name + '.png')
        raw_image_pil = Image.open(raw_img_path)
        raw_image_tensor = self.to_tensor(raw_image_pil)
        raw_image_tensor = self.normalize_raw(raw_image_tensor)

        if self.rgb_dir is None:
            return raw_image_tensor
        else:
            rgb_img_path = path.join(self.rgb_dir, img_name + '.png')
            rgb_image_pil = Image.open(rgb_img_path)
            rgb_image_tensor = self.to_tensor(rgb_image_pil)

            if self.rgb_norm:
                rgb_image_tensor = self.normalize_rgb(rgb_image_tensor)

            return raw_image_tensor, rgb_image_tensor
