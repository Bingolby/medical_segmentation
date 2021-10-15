import numpy as np
import os
from PIL import Image

#import cv2

# my_truth_dir = '/disk1/paper_data/test'
# namelist = ['18-09926A_2019-05-08 00_06_27-lv1-21025-9557-7081-5756.jpg',
#             '18-10829B_2019-05-08 00_47_24-lv1-24450-13719-3372-5801.jpg',
#             '18-10829A_2019-05-08 00_40_37-lv1-38624-10705-5233-3289.jpg',
#             '18-09926A_2019-05-08 00_06_27-lv1-23990-14292-4181-5408.jpg',
#             '18-11879A_2019-05-08 01_03_15-lv1-10471-18155-3868-2890.jpg',
#             '18-10829A_2019-05-08 00_40_37-lv1-41168-7882-3765-3289.jpg',
#             '18-09926B_2019-05-08 00_13_33-lv1-11741-24001-4020-5474.jpg',
#             '18-11879A_2019-05-08 01_03_15-lv1-10916-21067-2928-2104.jpg',
#             '18-10829B_2019-05-08 00_47_24-lv1-20240-15499-4901-5215.jpg',
#             '18-09926B_2019-05-08 00_13_33-lv1-9910-26446-3879-4218.jpg']
# seg_dir = '/disk1/paper_data/NetSegmentation_d0_debug/result'


def area_IU(seg_img, truth_img):
    """ 为了计算平均交并比，统计交集与并集面积 """
    # inter_arr = np.zeros(2)
    # union_arr = np.zeros(2)

    seg_copy = seg_img.copy()
    truth_copy = truth_img.copy()
    # inter = np.sum((seg_copy + truth_copy)==2)
    # union = np.sum((seg_copy + truth_copy)==1)
    inter = np.sum((seg_copy + truth_copy) == 2) * 2
    union = np.sum((seg_copy) == 1) + np.sum((truth_copy) == 1)

    return inter, union

def area_mIOU(seg_img, truth_img, classes):
    """ 为了计算平均交并比，统计交集与并集面积 """
    # inter_arr = np.zeros(2)
    # union_arr = np.zeros(2)

    seg_copy = seg_img.copy()
    truth_copy = truth_img.copy()
    # inter = np.sum((seg_copy + truth_copy)==2)
    # union = np.sum((seg_copy + truth_copy)==1)
    conmatrix = np.zeros((classes, classes))
    for i in range(classes):
        for j in range(classes):
            conmatrix[i, j] += np.sum((truth_img == i) * (seg_img == j))



    # inter = np.sum((seg_copy + truth_copy) == 2) * 2
    # union = np.sum((seg_copy) == 1) + np.sum((truth_copy) == 1)
    # iu = tp / (tp + fp + fn)
    # mean_iu = np.nanmean(iu)

    return conmatrix


def precision(seg_img, truth_img):
    """ 准确率 """
    #print(truth_img.max(), seg_img.max())
    truth_pixels = np.sum(truth_img > -1)
    right_pixels = np.sum(seg_img == truth_img)
    return right_pixels, truth_pixels


def evaluation(mode='TEST', get_return=False):
    """ 评价标准计算主函数 """
    if mode == 'TEST':
        namelist = namelist
        seg_dir = seg_dir
        truth_dir = my_truth_dir
    else:
        raise Exception("error: evaluation mode!")

    # 二分类
    # nc_inter_all = np.zeros(2)
    # nc_union_all = np.zeros(2)
    nc_inter_all = 0
    nc_union_all = 0
    # 准确率
    absolute_precision_arr = np.zeros(2)

    for i, name in enumerate(namelist):
        # print("{}/{}\t{}".format(i+1, len(namelist), name))
        seg_img = np.array(Image.open(os.path.join(seg_dir, name))).astype(np.float16)
        # truth_img = np.array(Image.open(os.path.join(truth_dir, name + '_mask.jpg'))).astype(np.float16)
        truth_img = np.array(Image.open(os.path.join(truth_dir, name + '_viable.png'))).astype(np.float16)
        # 二分类
        nc_inter, nc_union = area_IU(seg_img, truth_img)
        nc_inter_all += nc_inter
        nc_union_all += nc_union
        # 准确率
        right_pixels, truth_pixels = precision(seg_img, truth_img)
        absolute_precision_arr += np.array([right_pixels, truth_pixels])

    IoU = nc_inter_all / nc_union_all
    ap = absolute_precision_arr[0] / absolute_precision_arr[1]
    print("{}:\n\tIoU:{:.4f}\n\tAP:{:.4f}".
          format(mode, IoU, ap))
    if get_return:
        return IoU, ap


def evaluation_iter(dataset, mode, result_dict, my_truth_dir, seg_dir, get_return=True, classes = 5):
    """ 评价标准计算主函数，训练过程中调用 """
    if mode == 'TEST':
        truth_dir = my_truth_dir
    else:
        raise Exception("error: evaluation mode!")

    # 二分类
    # nc_inter_all = np.zeros(2)
    # nc_union_all = np.zeros(2)
    nc_inter_all = 0
    nc_union_all = 0
    # 准确率
    absolute_precision_arr = np.zeros(2)
    conmatrix_all = np.zeros((classes, classes))
    for i, name in enumerate(result_dict.keys()):
        # print("{}/{}\t{}".format(i+1, len(namelist), name))
        seg_img = result_dict[name].astype(np.float16)  # np.array(Image.open(os.path.join(seg_dir, name)))
        if dataset == 'Digestpath':
            truth_img = np.array(Image.open(os.path.join(truth_dir, name[:-4] + '_mask.jpg'))).astype(np.float16)
        elif dataset == 'Paip':
            truth_img = np.array(Image.open(os.path.join(truth_dir, '255', name[:-4] + '_viable.png'))).astype(np.float16)
        elif dataset == 'Potsdam':
            truth_img = np.array(Image.open(os.path.join(truth_dir, name[:-8] + '_label.png'))).astype(np.float16)
        elif dataset == 'Cervix':
            truth_img = np.array(Image.open(os.path.join(truth_dir, 'Mask_' + name))).astype(np.float16)

        if dataset == 'Digestpath':
            truth_img[truth_img < 128] = 0
            truth_img[truth_img > 127] = 1

        # # 二分类
        # nc_inter, nc_union = area_IU(seg_img, truth_img)
        # nc_inter_all += nc_inter
        # nc_union_all += nc_union

        # 多分类
        conmatrix = area_mIOU(seg_img, truth_img, classes)
        conmatrix_all += conmatrix

        # nc_inter_all += nc_inter
        # nc_union_all += nc_union
        # 准确率
        # right_pixels, truth_pixels = precision(seg_img, truth_img)
        # absolute_precision_arr += np.array([right_pixels, truth_pixels])

    # 平均交并比
    M, N = conmatrix_all.shape
    tp = np.zeros(M, dtype=np.uint)
    fp = np.zeros(M, dtype=np.uint)
    fn = np.zeros(M, dtype=np.uint)

    for i in range(M):
        tp[i] = conmatrix_all[i, i]
        fp[i] = np.sum(conmatrix_all[:, i]) - tp[i]
        fn[i] = np.sum(conmatrix_all[i, :]) - tp[i]

    precision = tp / (tp + fp)  # = tp/col_sum
    recall = tp / (tp + fn)
    f1_score = 2 * recall * precision / (recall + precision)

    # ax_p = 0  # column of confusion matrix
    # ax_t = 1  # row of confusion matrix
    acc = np.diag(conmatrix_all).sum() / conmatrix_all.sum()
    # acc_cls = np.diag(conmatrix_all) / conmatrix_all.sum(axis=ax_p)
    # acc_cls = np.nanmean(acc_cls)
    iu = tp / (tp + fp + fn)
    dice = 2*tp / (2*tp + fp + fn)
    mean_iu = np.nanmean(iu)
    mean_dice = np.nanmean(dice)
    # freq = conmatrix_all.sum(axis=ax_p) / conmatrix.sum()
    # fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    # IoU = nc_inter_all / nc_union_all
    # ap = absolute_precision_arr[0] / absolute_precision_arr[1]
    print("{}:\n\tIoU:{:.4f}\n\tACC:{:.4f}\n\tDICE:{:.4f}".
          format(mode, mean_iu, acc, mean_dice))
    print(dice)
    # print(iu)
    if get_return:
        return mean_iu, acc
