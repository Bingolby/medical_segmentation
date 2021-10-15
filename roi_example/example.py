#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 19:13:24 2020

@author: bingo
"""

import torch
from model_pytorch_parts import *
import numpy as np
#import torch.nn as nn

#generate data
x1 = torch.arange(1, 65).float().cuda()
x2 = torch.arange(65, 129).float().cuda()
x3 = torch.arange(129, 193).float().cuda()
x = torch.cat([x1.view(16, 1, 2, 2), x2.view(16, 1, 2, 2), x3.view(16, 1, 2, 2)], 1).contiguous()#B, C, H, W
x_cp = x

#merge
x = x.permute(0, 2, 3, 1).contiguous().view(4, 4, 2, 2, 3).permute(0, 2, 1, 3, 4).contiguous().view(1, 8, 8, 3).permute(0, 3, 1, 2).contiguous()

##split
w = 2
h = 2
rois = torch.FloatTensor( [[0,   0,   0,   w-1,   h-1],
                           [0,   w,   0, 2*w-1,   h-1],
                           [0, 2*w,   0, 3*w-1,   h-1],
                           [0, 3*w,   0, 4*w-1,   h-1],
                           [0,   0,   h,   w-1, 2*h-1],
                           [0,   w,   h, 2*w-1, 2*h-1],
                           [0, 2*w,   h, 3*w-1, 2*h-1],
                           [0, 3*w,   h, 4*w-1, 2*h-1],
                           [0,   0, 2*h,   w-1, 3*h-1],
                           [0,   w, 2*h, 2*w-1, 3*h-1],
                           [0, 2*w, 2*h, 3*w-1, 3*h-1],
                           [0, 3*w, 2*h, 4*w-1, 3*h-1],
                           [0,   0, 3*h,   w-1, 4*h-1],
                           [0,   w, 3*h, 2*w-1, 4*h-1],
                           [0, 2*w, 3*h, 3*w-1, 4*h-1],
                           [0, 3*w, 3*h, 4*w-1, 4*h-1]] )
rois = rois.cuda()

label_small_cpu = torch.Tensor(self.params.my_batchsize, self.params.mini_patch_size, self.params.mini_patch_size)
for i in range(4):
    for j in range(4):
        label_small_cpu[i * 4 + j, :, :] = sample['label'][0, i*self.params.mini_patch_size:(i+1)*self.params.mini_patch_size,
                                        j*self.params.mini_patch_size:(j+1)*self.params.mini_patch_size]
y = roi_pooling_2d(x, rois, (2, 2), spatial_scale=1.0).contiguous()
print(y.equal(x_cp))
                           


