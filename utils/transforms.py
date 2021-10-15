import numpy as np
import torchvision.transforms.functional as TF
import random
import torch
import matplotlib.pyplot as plt # 用于功能测试显示图片
from PIL import Image


# Mean_RGB = [192.2091, 147.6360, 194.3978]
# Std_RGB = [45.6902, 62.4272, 36.8595]

# Mean_RGB = [198.4715, 159.5626, 200.5592]
# Std_RGB = [46.1565, 66.7764, 38.2345]

class Transforms(object):
    """
    功能：封装数据变换形式
    注意：PIL读入的是RGB
    """
    def __init__(self, mode, crop_size, patch_size, mean_rgb, std_rgb):
        self.mode = mode
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.Mean_RGB = mean_rgb
        self.Std_RGB = std_rgb
        # self.Mean_RGB = np.array(Mean_RGB)
        # self.Std_RGB = np.array(Std_RGB)
    def __call__(self, sample, ):
        if self.mode == 'TRAIN':
            # 训练过程进行多种随机变换
            return self._train_(sample)
        else:
            # 验证过程只需要归一化
            return self._val_(sample)
    def _train_(self, sample):
        sample = self._random_flip_(sample)
        sample = self._random_color_(sample)
        sample = self._random_rotate_(sample)
        sample = self._random_crop_(sample, crop_size=self.crop_size, patch_size=self.patch_size)
        sample = self._normalize_(sample)

        return sample

    def _val_(self, sample):
        return self._normalize_(sample)

    def _normalize_(self, sample):
        """
        功能：分割归一化
        注意：需要修改pytorch的代码以防止出现warning
            /home/zhumeng/anaconda2/envs/py37/lib/python3.6/site-packages/torchvision/transforms/functional.py
            std = torch.tensor(std, dtype=torch.float32)
            torch.tensor 修改为 torch.as_tensor
        调用：增加噪声 _random_noise_
             拼接图像 _splice_part2_
        """
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.int64)

        img = self._random_noise_(img) # 增加噪声

        img -= self.Mean_RGB
        img /= self.Std_RGB
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()
        return {'image': img,
                'label': mask}

    def _random_flip_(self, sample):
        """ 随机翻转 """
        img = sample['image']
        mask = sample['label']
        temp = random.randint(0, 2)
        img = {0:img, 1:TF.hflip(img), 2:TF.vflip(img)}[temp]
        mask = {0:mask, 1:TF.hflip(mask), 2:TF.vflip(mask)}[temp]
        return {'image': img,
                'label': mask}

    def _random_color_(self, sample):
        """ 随机颜色变换 """
        img = sample['image']
        # 亮度
        img = TF.adjust_brightness(img, random.uniform(0.8,1.2))
        # 对比度
        img = TF.adjust_contrast(img, random.uniform(0.8,1.2))
        # 色度
        img = TF.adjust_hue(img, random.uniform(-0.05,0.05))
        # 饱和度
        img = TF.adjust_saturation(img, random.uniform(0.8,1.2))
        return {'image': img,
                'label': sample['label']}

    def _random_rotate_(self, sample, range_degree=90):
        if random.randint(0, 1):
            # 旋转
            img = sample['image']
            mask = sample['label']
            rotate_degree = random.uniform(-range_degree,range_degree)
            img = img.rotate(rotate_degree, Image.BILINEAR)
            mask = mask.rotate(rotate_degree, Image.NEAREST)
            return {'image': img,
                    'label': mask}
        else:
            # 不旋转
            return sample

    def _random_crop_(self, sample, crop_size, patch_size):
        """
        功能：随机裁剪
        参数：crop_size    用于训练的裁剪尺寸
             patch_size   patch的尺寸
        """
        img = sample['image']
        mask = sample['label']
        upper_coor = random.randint(0, patch_size - crop_size)
        left_coor = random.randint(0, patch_size - crop_size)
        img = TF.crop(img, upper_coor, left_coor, crop_size, crop_size)
        mask = TF.crop(mask, upper_coor, left_coor, crop_size, crop_size)
        return {'image' : img,
                'label' : mask}

    def _random_noise_(self, img, u=0, a=0.1):
        """
        功能：随机噪声
        参数：img numpy
        注意： normalize调用
        """
        temp = random.randint(0, 1)
        h, w, c = img.shape
        noise = np.random.normal(u, a, (h, w, c)).astype(np.float32)
        img = {0: img + noise, 1: img}[temp]
        return img