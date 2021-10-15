import numpy as np
import os
import time
import cv2
import matplotlib.pyplot as plt
import pickle
import sys
from PIL import Image


t_start = time.time()
patch_root = '/home/bingo/digestpath_data/size_1600_s_800'
patch_list_path = os.path.join(patch_root, 'patch_list.txt')
patch_img_path = os.path.join(patch_root, 'training_patch')
reader_patchlist = open(patch_list_path, 'r')
patchlist = [line.strip().split('\t')[1] for line in reader_patchlist]

mean_r, mean_g, mean_b = 0, 0, 0
std_r, std_g, std_b = 0, 0, 0
for i,line in enumerate(patchlist):
    img = Image.open(os.path.join(patch_img_path, line)).convert('RGB')
    img = np.array(img)
    mean_r += img[:,:,0].mean()
    mean_g += img[:,:,1].mean()
    mean_b += img[:,:,2].mean()
    if i%100 == 0:
        print ("{}/{}".format(i, len(patchlist)))
mean_r /= len(patchlist)
mean_g /= len(patchlist)
mean_b /= len(patchlist)

for i,line in enumerate(patchlist):
    img = Image.open(os.path.join(patch_img_path, line)).convert('RGB')
    img = np.array(img)
    std_r += pow(img[:,:,0] - mean_r, 2).mean()
    std_g += pow(img[:,:,1] - mean_g, 2).mean()
    std_b += pow(img[:,:,2] - mean_b, 2).mean()
    if i%100 == 0:
        print("{}/{}".format(i, len(patchlist)))
std_r /= len(patchlist)
std_g /= len(patchlist)
std_b /= len(patchlist)
std_r = pow(std_r, 0.5)
std_g = pow(std_g, 0.5)
std_b = pow(std_b, 0.5)
print('mean of rgb:[{:.4f}, {:.4f}, {:.4f}]'.format(mean_r, mean_g, mean_b))
print('std of rgb:[{:.4f}, {:.4f}, {:.4f}]'.format(std_r, std_g, std_b))
print('Time:{:.4f}'.format(time.time() - t_start))

reader_patchlist.close()
