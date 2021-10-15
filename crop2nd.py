import numpy as np
import os
import cv2

patchsize = 400

root = "/home/bingo/digestpath_data/size_1600_s_800_cropped"
Filelist = [line.strip().split('\t') for line in open(root + '/filenames.txt','r')]
read_dir = '/home/bingo/digestpath_data/size_1600_s_800'
write_dir = root
writer_patchlist = open(os.path.join(write_dir, 'patch_list.txt'), 'w')

savedir_patchimg = os.path.join(write_dir, 'training_patch')
savedir_patchmask = os.path.join(write_dir, 'training_mask')

TrainList = []
for line in Filelist:
    TrainList.append(line[1])

print(len(TrainList))

for i, filename in enumerate(TrainList):
    img = cv2.imread(os.path.join(read_dir, 'training_patch', filename))
    mask = cv2.imread(os.path.join(read_dir, 'training_mask', 'Mask_' + filename), cv2.IMREAD_GRAYSCALE)
    img_height, img_width, _ = img.shape
    mask_height, mask_width = mask.shape
    assert img_height == mask_height
    assert img_width == mask_width
    for h in range(0, 1201, 400):
        for w in range(0, 1201, 400):
            patch_img = img[h:h+patchsize, w:w+patchsize]
            assert patch_img.shape[0] == patch_img.shape[1], patch_img.shape
            patch_mask = mask[h:h + patchsize, w:w + patchsize]

            save_patch_name = filename[:-4] + '_h' + str(h) + '_w' + str(w) + '.png'
            writer_patchlist.write("{}\t{}\n".format(filename, save_patch_name))
            cv2.imwrite(os.path.join(savedir_patchmask, 'Mask_' + save_patch_name), patch_mask)
            cv2.imwrite(os.path.join(savedir_patchimg, save_patch_name), patch_img)

writer_patchlist.close()













