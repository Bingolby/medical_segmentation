import os
import time
from PIL import Image
import numpy as np
import torch
import pickle

import matplotlib.pyplot as plt  # 用于功能测试显示图片

#from net.ocr import OCR  # 网络模块
from net.network import Novel_Strategy
from net.network import  CCNet
from utils import eval  # 评价指标

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Image.MAX_IMAGE_PIXELS = None


class Option():
    """ 可修改参数 """

    def __init__(self, model_name_):
        # 图片类别
        self.mode = 'TEST'
        # 模型路径
        model_dir = '/home/bingo/structual_seg_paper/model'
        #model_name = '20200915-deeplabv3+_Epoch10.pth'
        model_name = model_name_ + '.pth'
        self.tag = 'Two-'  # 存储文件的区别标签
        print(model_name)
        self.model_path = os.path.join(model_dir, model_name)
        self.dataset = 'Cervix'
        self.classes = 5
        self.bingo = True

        # 图片路径
        if self.mode == 'TEST':
            if self.dataset == 'Paip':
                input_img_dir = '/home/bingo/paip2019_data/png_validation'
                self.truth_dir = '/home/bingo/paip2019_data/png_validation'
            elif self.dataset == 'Digestpath':
                input_img_dir = './data/digestpath_data/test'
                self.truth_dir = './data/digestpath_data/test'
            elif self.dataset == 'Cervix':
                input_img_dir = './data/cervix_data/ImgLevel2/Test'
                self.truth_dir = './data/cervix_data/TruthLevel2/All'
            elif self.dataset == 'Deepglobe':
                input_img_dir = '/home/bingo/DeepGlobe_data/valid'
            elif self.dataset == 'Potsdam':
                input_img_dir = '/home/bingo/Potsdam_data/ToPngDir'
                self.truth_dir = '/home/bingo/Potsdam_data/ToPngDir_Label'

            self.save_seg_dir = '/home/bingo/structual_seg_paper/result'
            self.input_img_list = self.get_img_list_from_dir(input_img_dir)
            self.save_dat = os.path.join(self.save_seg_dir, model_name[:-4] + '.dat')
        else:
            raise Exception("Option init: mode error!")

        self.patch_size = 1536  # 289 # 测试时每个patch裁剪的大小
        self.max_batchsize = 1  # 25 # 最大BatchSize，测试时每行最后一个batchsize小于最大值
        self.my_batchsize = 16
        self.mini_patch_size = 384
        #digestpath
        if self.dataset == 'Digestpath':
        #digestpath
            self.mean_RGB = [199.3305, 161.1973, 201.3773]
            self.std_RGB = [46.0578, 66.9819, 38.2566]
        #cervix
        elif self.dataset == 'Cervix':
            self.mean_RGB = [123.675, 116.28, 103.53]
            self.std_RGB = [58.395, 57.12, 57.375]
        #paip2019
        elif self.dataset == 'Paip':
            self.mean_RGB = [206.1974, 162.8807, 191.9159]
            self.std_RGB = [32.1165, 50.7997, 34.9664]
        elif self.dataset == 'Deepglobe':
            self.mean_RGB = [104.0949, 96.6685, 71.8058]
            self.std_RGB = [37.4961, 29.2473, 26.7463]
        elif self.dataset == 'Potsdam':
            self.mean_RGB = [85.9192, 91.8413, 85.0438]
            self.std_RGB = [35.8101, 35.1447, 36.5198]

    def get_img_list_from_dir(self, input_img_dir):
        """ 固定文件列表防止文件缺失 """
        filelist = []
        if self.mode == 'TEST':
            if self.dataset == 'Digestpath':
                namelist = ['18-09926A_2019-05-08 00_06_27-lv1-21025-9557-7081-5756.jpg',
                            '18-10829B_2019-05-08 00_47_24-lv1-24450-13719-3372-5801.jpg',
                            '18-10829A_2019-05-08 00_40_37-lv1-38624-10705-5233-3289.jpg',
                            '18-09926A_2019-05-08 00_06_27-lv1-23990-14292-4181-5408.jpg',
                            '18-11879A_2019-05-08 01_03_15-lv1-10471-18155-3868-2890.jpg',
                            '18-10829A_2019-05-08 00_40_37-lv1-41168-7882-3765-3289.jpg',
                            '18-09926B_2019-05-08 00_13_33-lv1-11741-24001-4020-5474.jpg',
                            '18-11879A_2019-05-08 01_03_15-lv1-10916-21067-2928-2104.jpg',
                            '18-10829B_2019-05-08 00_47_24-lv1-20240-15499-4901-5215.jpg',
                            '18-09926B_2019-05-08 00_13_33-lv1-9910-26446-3879-4218.jpg']
            elif self.dataset == 'Paip':
                namelist = ['01_01_0105.png', '01_01_0142.png', '01_01_0145.png', '01_01_0161.png',
                            '01_01_0141.png', '01_01_0144.png', '01_01_0159.png', '01_01_0140.png',
                            '01_01_0143.png', '01_01_0150.png']
            elif self.dataset == 'Potsdam':
                namelist = ['top_potsdam_2_13_RGB.png', 'top_potsdam_2_14_RGB.png', 
                            'top_potsdam_3_13_RGB.png', 'top_potsdam_3_14_RGB.png', 
                            'top_potsdam_4_13_RGB.png', 'top_potsdam_4_14_RGB.png', 'top_potsdam_4_15_RGB.png', 
                            'top_potsdam_5_13_RGB.png', 'top_potsdam_5_14_RGB.png', 'top_potsdam_5_15_RGB.png', 
                            'top_potsdam_6_13_RGB.png', 'top_potsdam_6_14_RGB.png', 'top_potsdam_6_15_RGB.png', 
                            'top_potsdam_7_13_RGB.png']
            elif self.dataset == 'Cervix':
                namelist = ['Crop_B1208929-4-8D.png',  'Crop_B1213030-3-7D.png', 'Crop_B1215515-2-4D.png', 'Crop_B1215523-1-1D.png', 'Crop_B1221216-1-2D.png']
        for name in namelist:
            filelist.append(os.path.join(input_img_dir, name))
        return filelist


class GetSegImg():
    """ 获得分割结果，存储灰度图并显示彩色图 """

    def __init__(self, params):
        self.params = params
        self.mode = params.mode
        self.input_img_list = params.input_img_list  # 输入文件列表
        self.truth_dir = params.truth_dir  # truth文件夹
        self.save_seg_dir = params.save_seg_dir  # 存储结果路径
        model_path = params.model_path  # 训练好的呃模型路径
        self.patch_size = params.patch_size  # 测试时每个patch裁剪的大小
        self.max_batchsize = params.max_batchsize  # 最大BatchSize，测试时每行最后一个batchsize小于最大值
        self._model_init_(model_path)  # 模型初始化，载入参数
        self.mean_RGB = params.mean_RGB
        self.std_RGB = params.std_RGB
        self.binary_path = params.save_dat
        self.tag = params.tag

    def _model_init_(self, model_path):
        """ 模型初始化载入参数 """
        self.mynet = Novel_Strategy(backbone_name = 'resnet50', output_stride = 16, num_classes = self.params.classes, pretrain = None,
            patch_size = self.params.patch_size, mini_patch_size = self.params.mini_patch_size, my_batchsize = self.params.my_batchsize, 
            bingo = self.params.bingo).to(device)
        model_dict = {}
        pretrained_dict = torch.load(model_path)
        state_dict = self.mynet.state_dict()
        model_dict = {k: v for k, v in pretrained_dict.items() if k in state_dict}
        state_dict.update(model_dict)
        self.mynet.load_state_dict(model_dict)

    def __call__(self):

        time_all = time.time()
        result_dict = {}

        for i, input_img_path in enumerate(self.input_img_list):
            time_single = time.time()

            # 输入图像
            input_img_name = input_img_path[input_img_path.rfind('/') + 1:]
            input_img = Image.open(input_img_path).convert('RGB')
            input_img = np.array(input_img)  # 输入图像矩阵

            # 分割图像
            seg_img = self._seg_single_img_(input_img)  # 分割结果矩阵

            # 显示图像
            # self._plt_show_img_(input_img, seg_img, truth_img)

            # # 存储结果
            # seg_img_c = self._to_color_(seg_img)
            # seg_img_c = Image.fromarray(seg_img_c.astype(np.uint8))
            # seg_img_c.save(os.path.join(self.save_seg_dir, self.tag+input_img_name)) # 存储

            result_dict[input_img_name] = seg_img.astype(np.uint8)

            # print("{}/{}\t{}\t{}\t{:.4f} s".format(i + 1, len(self.input_img_list),
            #                                        input_img_name, input_img.shape, time.time() - time_single))

        print("TIme: {:.4f} s".format(time.time() - time_all))

        pickle.dump(result_dict, open(self.binary_path, 'wb'))

        # 测试
        self._eval_(self.binary_path)

    def _seg_single_img_(self, input_img):
        """
        功能：分割单张图片
        输入：numpy图像RGB h*w*3
        输出：numpy分割结果 h*w
        """
        height, width = input_img.shape[0], input_img.shape[1]

        # patch尺寸整数倍的补全尺寸
        width_ceil = int(self.patch_size * np.ceil(width / self.patch_size))
        heigth_ceil = int(self.patch_size * np.ceil(height / self.patch_size))
        # input_img_ceil = input_img
        # 填充0补全原图
        if width_ceil > width or heigth_ceil > height:
            width_pad = width_ceil - width
            height_pad = heigth_ceil - height
            input_img_ceil = np.pad(input_img, ((0, height_pad),
                                                (0, width_pad), (0, 0)), 'constant')
        # 分割结果矩阵，由于填零可能比需要分割的尺寸大
        seg_img = np.zeros((heigth_ceil, width_ceil))

        for height_index in range(0, heigth_ceil, self.patch_size):
            for width_index in range(0, width_ceil, self.max_batchsize * self.patch_size):
                batch_end = width_index + self.max_batchsize * self.patch_size
                if batch_end > width_ceil:
                    batch_end = width_ceil
                batch = input_img_ceil[height_index:(height_index +
                                                     self.patch_size), width_index:batch_end, :]
                # print(batch.shape)
                batch = np.split(batch, int(batch.shape[1] / self.patch_size), axis=1)
                batch = np.array(batch)

                # print(batch.shape)
                # exit()
                # 分割一个batch
                seg_batch = self._seg_single_batch_(batch)
                seg_batch = np.concatenate(seg_batch, axis=1)

                # 存入扩充结果矩阵
                seg_img[height_index:(height_index +
                                      self.patch_size), width_index:batch_end] = seg_batch[:, :]

        # 切片得到真正的结果矩阵
        seg_img = seg_img[:height, :width]

        # 过滤背景
        seg_img = self._erase_background_(input_img, seg_img)
        # print(seg_img.sum())
        return seg_img

    def _seg_single_batch_(self, batch):
        """
        功能：分割一个batch
        输入：numpy batchsize * size * size * 3
        输出：numpy batchsize * size * size
        """
        self.mynet.eval()
        batch = self._normalize_(batch)
        # print(batch.shape)
        mini_patch_size = self.params.mini_patch_size
        with torch.no_grad():

            output_batch = self.mynet(batch)
            if(self.params.bingo):
                my_outputs = torch.Tensor(output_batch.shape[0]//self.params.my_batchsize, 
                    output_batch.shape[1], self.params.patch_size, self.params.patch_size).to(device)
                # for k in range(output_batch):
                for i in range(4):
                    for j in range(4):
                        my_outputs[0, :, i * mini_patch_size:(i + 1) * mini_patch_size, j * mini_patch_size:(j + 1) * mini_patch_size] = output_batch[i * 4 + j, :, :, :]
            else:
                my_outputs = output_batch

        output_batch = my_outputs
        output_batch = output_batch.cpu().data.numpy()
        # print(output_batch.shape)
        output_batch = np.argmax(output_batch, axis=1)  # n*s*s
        # print(output_batch.sum())
        return output_batch

    def _erase_background_(self, input_img, seg_img):
        """ 抹掉黑色背景区域的分割结果 """
        input_img_sumRGB = np.sum(input_img, axis=2)
        input_img_sumRGB[input_img_sumRGB > 0] = 1  # 前景为1，背景为0
        seg_img = seg_img * input_img_sumRGB  # 背景置0
        return seg_img

    def _normalize_(self, batch):
        """ 分割归一化，输入 numpy n*s*s*3"""
        batch = np.array(batch).astype(np.float16)
        batch -= self.mean_RGB
        batch /= self.std_RGB
        batch = batch.transpose((0, 3, 1, 2))
        batch = torch.from_numpy(batch).float().to(device)
        return batch

    def _plt_show_img_(self, input_img, seg_img, truth_img):
        """ 显示原图、分割结果、truth的对比结果 """
        seg_img_color = self._to_color_(seg_img)
        truth_img_color = self._to_color_(truth_img)
        plt.figure()
        plt.subplot(1, 3, 1), plt.imshow(input_img), plt.axis('off')
        plt.subplot(1, 3, 2), plt.imshow(seg_img_color), plt.axis('off')
        plt.subplot(1, 3, 3), plt.imshow(truth_img_color), plt.axis('off')
        plt.show()

    def _to_color_(self, img):
        """ 预测结果和truth转成三通道彩色RGB用于显示 """
        colors = {0: (0, 0, 0),
                  1: (255, 255, 255),
                  2: (0, 255, 0),
                  3: (0, 0, 255),
                  4: (255, 0, 0), }
        height, width = img.shape
        color_img = np.zeros((height, width, 3)).astype(np.uint8)
        for h in range(height):
            for w in range(width):
                color_img[h][w] = colors[img[h][w]]
        return color_img

    def _eval_(self, binary_path):
        """ 计算评价指标 """
        region_dict = pickle.load(open(binary_path, 'rb'))
        eval.evaluation_iter(self.params.dataset, self.mode, region_dict, self.truth_dir, self.save_seg_dir, True, self.params.classes)


if __name__ == '__main__':
    # torch.backends.cudnn.benchmark = False
    model_name_ = 'Cervix_1600_Epoch'
    model_name = ''
    for i in range(10, 101, 10):
        if(os.path.isfile('./model/' + model_name_ + str(i) + '.pth')):
            model_name = model_name_ + str(i)
    opt = Option(model_name)
    obj = GetSegImg(opt)
    obj()
