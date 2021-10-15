#2020.09.02
import numpy as np
import torch
import time
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
from net.network import Deeplab
from net.network import CCNet
from net.network import Novel_Strategy
#from net.ocr import OCR#
from utils.dataset import DatasetForTraining
from utils.loss import EQLv2 as LossFunction #EQLV2
# from utils.loss import LossFunction #BCE
from utils.eval import evaluation_iter
# from roi_example.model_pytorch_parts import roi_pooling_2d
from apex import amp
opt_level = 'O1'
#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Option(object):
    def __init__(self):
        self.state = 'TRAIN'
        self.dataset = 'Cervix'
        if self.dataset == 'Digestpath':
            self.classes = 2
            self.Mean_RGB = [199.3305, 161.1973, 201.3773]
            self.Std_RGB = [46.0578, 66.9819, 38.2566]
            self.PatchRoot = './data/digestpath_data/size_1600_s_800_cropped'
            self.loss_weight = None
        elif self.dataset == 'Cervix':
            self.classes = 5
            self.Mean_RGB = [123.675, 116.28, 103.53]
            self.Std_RGB = [58.395, 57.12, 57.375]
            self.PatchRoot = './data/cervix_data/size_1600_s_400_cropped'
            self.loss_weight = None
        elif self.dataset == 'Paip':
            self.Mean_RGB = [206.1974, 162.8807, 191.9159]
            self.Std_RGB = [32.1165, 50.7997, 34.9664]
        elif self.dataset == 'Deepglobe':
            self.Mean_RGB = [104.0949, 96.6685, 71.8058]
            self.Std_RGB = [37.4961, 29.2473, 26.7463]
            self.loss_weight = [
                1.00000000,  # Unknown
                0.89455324,  # urban
                0.16654299,  # Agriculture
                1.13362805,  # Rangeland
                0.86512339,  # Forest
                2.88647058,  # Water
                1.13921060   # Barren   
            ]
        elif self.dataset == 'Potsdam':
            self.classes = 6
            self.Mean_RGB = [85.9192, 91.8413, 85.0438]
            self.Std_RGB = [35.8101, 35.1447, 36.5198]
            self.loss_weight = [
                0.67345783, #Impervious surfaces
                0.7144098,  #Building
                0.80999469, #Low vegetation
                1.30646543, #Tree
                11.32644145,#Car
                3.91366044  #Clutter/background
            ]
            self.PatchRoot = '/home/bingo/Potsdam_data/256'
        self.bingo = False
        self.amp = True
        self.lr = 1e-4
        self.weight_decay = 2e-5
        self.check_epoch = 10
        self.iter_print = 50
        self.pretrained = True
        self.model_dir = './model/'
        self.pretrain_net = 'Default_To_True'
        self.env = 'structual_seg'
        self.seed = 1632
        self.orig_img_size = 400#[1600, 400]
        self.patch_size = 384#[1536, 384]
        self.batchsize = 80#[3, 48]
        self.mini_patch_size = 384
        self.my_batchsize = 16
        self.epochs = 100
        
        self.model_tag = 'Cervix_1600_cropped'

class Structual_Segmentation(object):
    def __init__(self, params):
        self.params = params
        self.patch_size = self.params.patch_size
        self.max_batchsize = self.params.batchsize


        torch.manual_seed(self.params.seed)
        torch.cuda.manual_seed(self.params.seed)
        print("{}Initializing{}".format('*' * 10, '*' * 10))

        now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
        self.log = os.path.join(self.params.model_dir, 'log', self.params.model_tag+'.txt')
        self.log = open(self.log, 'a')

        print("{}\nState:{}\nBatchSize:{}\nPretrained:{}\nModel:{}".format(now,
            self.params.state, self.params.batchsize, self.params.pretrain_net,
            self.params.model_tag))

        self.log.write("{}\nState:{}\nBatchSize:{}\nPretrained:{}\nModel:{}\n".format(now,
            self.params.state, self.params.batchsize, self.params.pretrain_net,
            self.params.model_tag))

        if self.params.state == 'TRAIN':
            train_dataloader = DataLoader(dataset=DatasetForTraining( self.params.patch_size, self.params.orig_img_size,
                                            self.params.Mean_RGB, self.params.Std_RGB, self.params.PatchRoot),
                                            num_workers=16, batch_size=self.params.batchsize, shuffle=True)
            self.dataloader = {'TRAIN':train_dataloader}
        else:
            raise Exception ('Error: state error!(__init__)')

        self._load_mynet_()
        self.optimizer = torch.optim.Adam(params=[{'params': self.mynet.backbone.parameters(),   'lr': self.params.lr,      'weight_decay': self.params.weight_decay},
                                                  {'params': self.mynet.classifier.parameters(), 'lr': self.params.lr,   'weight_decay': self.params.weight_decay}], 
                                          amsgrad=True)
        if(self.params.loss_weight):
            mfb_weight = torch.from_numpy(np.asarray(self.params.loss_weight, dtype=np.float32)).cuda()
            self.net_loss = LossFunction(mfb_weight)
        else:
            self.net_loss = LossFunction()

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.params.epochs, 1e-5)
        if(self.params.amp):
            self.mynet, self.optimizer = amp.initialize(
                self.mynet, self.optimizer, opt_level=opt_level)

    def _load_mynet_(self, mode=None):
#         self.mynet = Deeplab(backbone = 'resnet101', output_stride = 8, num_classes = 2).to(device)
        self.mynet = Novel_Strategy(backbone_name = 'resnet50', output_stride = 16, num_classes = self.params.classes, pretrain = self.params.pretrained,
            patch_size = self.params.patch_size, mini_patch_size = self.params.mini_patch_size, my_batchsize = self.params.my_batchsize, 
            bingo = self.params.bingo).to(device)
    def _checkpoint_(self, str_, iter):
        model_out_path = os.path.join(self.params.model_dir, self.params.model_tag +'_'+ str_ + str(iter) + '.pth')
        torch.save(self.mynet.state_dict(), model_out_path)
        print("===> Checkpoint saved to {}\n".format(model_out_path))
        os.system('CUDA_VISIBLE_DEVICES=2 python test_hold.py')


    def __call__(self):
        for epoch in range(self.params.epochs):
            print('-' * 20)
            if self.params.state == 'TRAIN':
                self.Phase = ['TRAIN',]
            else:
                raise Exception("Error: state error!")

            for phase in self.Phase:
                if phase == 'TRAIN':
                    self.mynet.train()
                else:
                    raise Exception("Error: phase error!")

                epoch_loss = 0
                dataloader_size = len(self.dataloader[phase])

                t_iter = time.time()
                t_epoch = time.time()

                for iteration, sample in enumerate(self.dataloader[phase]):
                    torch.cuda.empty_cache()
                    image_cpu = sample['image']
                    label_cpu = sample['label']
                    if(self.params.bingo):
                        label_small_cpu = torch.Tensor(self.params.my_batchsize*image_cpu.shape[0], self.params.mini_patch_size, self.params.mini_patch_size)
                        M = int(self.params.my_batchsize ** 0.5)
                        for k in range(image_cpu.shape[0]):
                            for i in range(M):
                                for j in range(M):
                                    label_small_cpu[k * self.params.my_batchsize + i * M + j, :, :] = sample['label'][k, i*self.params.mini_patch_size:(i+1)*self.params.mini_patch_size,
                                                                    j*self.params.mini_patch_size:(j+1)*self.params.mini_patch_size]
                        label_small_cpu = label_small_cpu.contiguous()
                        labels = label_small_cpu.to(device).contiguous()
                    else:
                        labels = label_cpu.to(device)
                    inputs = image_cpu.to(device)
                    
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'TRAIN'):
                        my_outputs = self.mynet(inputs)
                        loss = self.net_loss(my_outputs, labels)

                        if phase == 'TRAIN':
                            if not self.params.amp:
                                loss.backward()
                            else:
                                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                                    scaled_loss.backward()
                            self.optimizer.step()

                    epoch_loss += float(loss.item())
                    if phase == 'TRAIN':
                        if(iteration + 1) % self.params.iter_print == 0:
                            print("==> Epoch[{}]({}/{}): lr: {} Loss: {:.4f} Time: {:4f}".\
                                format((epoch+1), iteration+1, dataloader_size,self.optimizer.param_groups[0]['lr'], loss.item(), time.time()-t_iter))

                            t_iter = time.time()
                #

                epoch_loss /= dataloader_size
                print("{}\n==> Epoch {} Complete: Avg. Loss: {:.4f} Time:{:.4f} ".format('_'*10,epoch+1, epoch_loss, time.time()-t_epoch))
                self.log.write("{}\n==> Epoch {} Complete: Avg. Loss: {:.4f} Time:{:.4f} \n".format('_'*10,epoch+1, epoch_loss, time.time()-t_epoch))

                if phase == 'TRAIN':
                    self.scheduler.step()
                    if ((epoch + 1) % self.params.check_epoch == 0):
                        self._checkpoint_('Epoch', epoch + 1)

                else:
                    pass


if __name__ == "__main__":
    opt = Option()
    obj = Structual_Segmentation(opt)
    obj()








































































