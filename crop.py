import numpy as np
import os
import time
import cv2
import sys

fore_th = 0.8#0.8*255
filter_th = 0.2#0.2*patchsize*patchsize
patchsize = 1600
stride = 800 #33%
TrainList, TestList = [],[]
root = "/home/bingo/digestpath_data/size_1600_s_800"
Filelist = [line.strip().split('\t') for line in open(root + '/filenames.txt','r')]
read_dir = '/home/bingo/digestpath_data/train_raw'
# write_dir = os.path.join(root, 'digestpath_patch/size_test')
write_dir = root
writer_patchlist = open(os.path.join(write_dir, 'patch_list.txt'), 'w')
writer_patchinfo = open(os.path.join(write_dir, 'patch_info.txt'), 'w')

savedir_patchimg = os.path.join(write_dir, 'training_patch')
savedir_patchmask = os.path.join(write_dir, 'training_mask')

PatchInfoArr = np.zeros((40,3)).astype(np.int64)

for line in Filelist:
    if line[0]=='train':
        TrainList.append(line[1])
    else:
        TestList.append(line[1])
print(len(TrainList), len(TestList))

for i,filename in enumerate(TrainList):
    # if (filename != "18-03912B_2019-05-07 23_00_51-lv1-24683-16377-1726-1900"):
    #     continue
    img = cv2.imread(os.path.join(read_dir, filename+'.jpg'))
    mask = cv2.imread(os.path.join(read_dir, filename+'_mask.jpg'))
    img_height, img_width, _ = img.shape
    mask_height, mask_width, _ = mask.shape
    assert img_height == mask_height
    assert img_width == mask_width
    height_padding = int(np.floor((img_height - patchsize)/stride)*stride + patchsize)
    width_padding = int(np.floor((img_width - patchsize)/stride)*stride + patchsize)
    # img_padding = np.pad(img, ((0, height_padding - img_height),(0, width_padding - img_width),(0,0)), 'constant')
    # mask_padding = np.pad(mask, ((0, height_padding - img_height),(0, width_padding - img_width),(0,0)), 'constant')
    img_padding = img[:height_padding, :width_padding, :]
    mask_padding = mask[:height_padding, :width_padding, :]
    print(img_padding.shape, img.shape)
    for h in range(0, (height_padding - patchsize + 1), stride):
        for w in range(0, (width_padding - patchsize + 1), stride):
            patch_img = img_padding[h:h+patchsize, w:w+patchsize]
            assert patch_img.shape[0] == patch_img.shape[1], patch_img.shape
            #get foregreound mask
            patch_fgmask = cv2.cvtColor(patch_img, cv2.COLOR_BGR2GRAY)
            patch_fgmask[patch_fgmask<20] = 255
            ret, patch_fgmask = cv2.threshold(patch_fgmask, fore_th*255, 1, cv2.THRESH_BINARY_INV)
            # if (filename == "18-03912B_2019-05-07 23_00_51-lv1-24683-16377-1726-1900"):
            #     print((patch_fgmask==1).sum())
            if ((patch_fgmask==1).sum() < filter_th*patchsize*patchsize):
                continue
            else:

                patch_mask = mask_padding[h:h + patchsize, w:w + patchsize]
                patch_mask = cv2.cvtColor(patch_mask, cv2.COLOR_BGR2GRAY)
                patch_mask[patch_mask > fore_th * 255] = 255
                patch_mask[patch_mask <= fore_th * 255] = 0
                ret, patch_mask = cv2.threshold(patch_mask, fore_th * 255, 1, cv2.THRESH_BINARY)

                save_patch_name = filename + '_h' + str(h) + '_w' + str(w) + '.png'
                writer_patchlist.write("{}\t{}\n".format(filename, save_patch_name))
                cv2.imwrite(os.path.join(savedir_patchmask, 'Mask_' + save_patch_name), patch_mask)
                cv2.imwrite(os.path.join(savedir_patchimg, save_patch_name), patch_img)
                PatchInfoArr[i][0] += (patch_mask == 1).sum()  # tumor
                PatchInfoArr[i][1] += (patch_mask == 0).sum()  # background
                PatchInfoArr[i][2] += 1

writer_patchinfo.write("Patch_sum:{}\nArea_sum{}\n".format(PatchInfoArr[:, 2].sum(), '-' * 30))
writer_patchinfo.write(
    "Tumor:{}\nBackground:{}\n{}\n".format(PatchInfoArr[:, 0].sum(), PatchInfoArr[:, 1].sum(), '-' * 30))
for j, filename in enumerate(TrainList):
    writer_patchinfo.write(
        "{}\tPatch:{}\tTumor:{}\tBackground:{}\n".format(filename, PatchInfoArr[j, 2], PatchInfoArr[j, 0],
                                                         PatchInfoArr[j, 1]))
writer_patchinfo.close()
writer_patchlist.close()
