import torch
import torch.nn as nn
from model_pytorch_parts import *
# from roi_pooling.functions.roi_pooling import ROIPooling2d
from torch.nn import functional as F
# from roi_align import RoIAlign      # RoIAlign module
# from roi_align import CropAndResize # crop_and_resize module
import torchvision


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class model(nn.Module):
    def __init__(self, input_shape=[284, 284, 3],
                 n_classes=6,
                 hook_indexes=[3, 3],#[3, 3]->[3-3, 3]->{3, 0}->[0, 0] shang0->xia0
                 depth=4,            #[0, 3]->[3-0, 3]->{3, 3-0}->[0, 3] shang3->xia0
                 n_convs=2,
                 filter_size=3,
                 n_filters=16,
                 padding='valid',
                 batch_norm=True,
                 activation='relu',
                 learning_rate=0.000005,
                 opt_name='adam',
                 l2_lambda=0.0001 ,
                 loss_weights=[1.0, 0.0],
                 merge_type='concat'):
        super(model, self).__init__()
        self.n_classes = n_classes
        self.hook_indexes = hook_indexes
        self.depth = depth
        self.n_convs = n_convs
        self.filter_size = filter_size
        self.n_filters = n_filters
        self.padding = padding
        self.batch_norm = batch_norm
        self.activation = activation
        self.merge_type = merge_type
        self.context_branch = Context_Branch(n_filters, depth, n_convs, hook_indexes, n_classes, reshape_name = "reshape_context")
        self.target_branch = Target_Branch(n_filters, depth, n_convs, hook_indexes, n_classes, reshape_name = "reshape_target")
        self._init_weight()

    def forward(self, input1, input2):
        flatten2, context_hooks = self.context_branch(input2)

        flatten1 = self.target_branch(input1, context_hooks)

        print(flatten2.shape, flatten1.shape)

        return flatten1, flatten2

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class Context_Branch(nn.Module):
    def __init__(self, N_Filters, Depth, N_Convs, hook_indexes, n_classes, reshape_name = "reshape_context"):
        super(Context_Branch, self).__init__()
        self._n_filters = N_Filters
        self._depth = Depth
        self._hook_indexes = {(Depth - 1) - hook_indexes[0]: hook_indexes[1]}
        self._enconv_block1 = Double_Conv(in_channels=3, out_channels=N_Filters)
        self._down1 = nn.MaxPool2d(2)
        self._enconv_block2 = Double_Conv(in_channels=N_Filters, out_channels=N_Filters * 2)
        self._down2 = nn.MaxPool2d(2)
        self._enconv_block3 = Double_Conv(in_channels=N_Filters * 2, out_channels=N_Filters * 4)
        self._down3 = nn.MaxPool2d(2)
        self._enconv_block4 = Double_Conv(in_channels=N_Filters * 4, out_channels=N_Filters * 8)
        self._down4 = nn.MaxPool2d(2)

        self._mid_conv_block = Double_Conv(in_channels=N_Filters * 8, out_channels=N_Filters * 10)

        self.concat = Concat()
        self._up1 = Up(in_channels=N_Filters * 10, out_channels=N_Filters * 8)
        self._deconv_block1 = Double_Conv(in_channels=N_Filters*8*2, out_channels=N_Filters*8)
        self._up2 = Up(in_channels=N_Filters * 8, out_channels=N_Filters * 4)
        self._deconv_block2 = Double_Conv(in_channels=N_Filters*4*2, out_channels=N_Filters * 4)
        self._up3 = Up(in_channels=N_Filters * 4, out_channels=N_Filters * 2)
        self._deconv_block3 = Double_Conv(in_channels=N_Filters*2*2, out_channels=N_Filters * 2)
        self._up4 = Up(in_channels=N_Filters * 2, out_channels=N_Filters)
        self._deconv_block4 = Double_Conv(in_channels=N_Filters*2, out_channels=N_Filters)

        self.out_conv = nn.Conv2d(N_Filters, n_classes, kernel_size=1, padding=0)
        self._init_weight()

    def forward(self, input2):
        # encode and retreive residuals

        en_fea1 = self._enconv_block1(input2)               #280, 280, 16
        en_fea1_down = self._down1(en_fea1)                 #140, 140, 16

        en_fea2 = self._enconv_block2(en_fea1_down)         #136, 136, 32
        en_fea2_down = self._down2(en_fea2)                 #68, 68, 32

        en_fea3 = self._enconv_block3(en_fea2_down)         #64, 64, 64
        en_fea3_down = self._down3(en_fea3)                 #32, 32, 64

        en_fea4 = self._enconv_block4(en_fea3_down)         #28, 28, 128
        en_fea4_down = self._down4(en_fea4)                 #14, 14, 128

        mid_fea = self._mid_conv_block(en_fea4_down)        #10, 10, 160

        # decode and retreive hooks
        out_hooks = []
        de_fea1 = self._up1(mid_fea)                        #18, 18, 128
        fea1_fuse = self.concat(de_fea1, en_fea4)
        fea1_fuse = self._deconv_block1(fea1_fuse)          #14, 14, 128
        out_hooks.append(fea1_fuse)

        de_fea2 = self._up2(fea1_fuse)
        fea2_fuse = self.concat(de_fea2, en_fea3)
        fea2_fuse = self._deconv_block2(fea2_fuse)
        out_hooks.append(fea2_fuse)

        de_fea3 = self._up3(fea2_fuse)
        fea3_fuse = self.concat(de_fea3, en_fea2)
        fea3_fuse = self._deconv_block3(fea3_fuse)
        out_hooks.append(fea3_fuse)

        de_fea4 = self._up4(fea3_fuse)
        fea4_fuse = self.concat(de_fea4, en_fea1)
        fea4_fuse = self._deconv_block4(fea4_fuse)
        out_hooks.append(fea4_fuse)

        output = self.out_conv(fea4_fuse)
        output = F.softmax(output, dim=1)
        flatten = torch.reshape(output, (output.shape[0], output.shape[1], output.shape[2]*output.shape[3]))

        hooks = {}
        for shook, ehook in self._hook_indexes.items():
            hooks[ehook] = out_hooks[shook]

        return flatten, hooks

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Target_Branch(nn.Module):
    def __init__(self, N_Filters, Depth, N_Convs, hook_indexes, n_classes, reshape_name = "reshape_target"):
        super(Target_Branch, self).__init__()
        self._n_filters = N_Filters
        self._depth = Depth
        self._hook_indexes = {(Depth - 1) - hook_indexes[0]: hook_indexes[1]}

        self._enconv_block1 = Double_Conv(in_channels=3, out_channels=N_Filters)
        self._down1 = nn.MaxPool2d(2)
        self._enconv_block2 = Double_Conv(in_channels=N_Filters, out_channels=N_Filters * 2)
        self._down2 = nn.MaxPool2d(2)
        self._enconv_block3 = Double_Conv(in_channels=N_Filters * 2, out_channels=N_Filters * 4)
        self._down3 = nn.MaxPool2d(2)
        self._enconv_block4 = Double_Conv(in_channels=N_Filters * 4, out_channels=N_Filters * 8)
        self._down4 = nn.MaxPool2d(2)

        self._mid_conv_block = Double_Conv(in_channels=N_Filters * 8, out_channels=N_Filters * 10)

        self.concat = Concat()
        self._up1 = Up(in_channels=N_Filters * 10 + N_Filters * 8, out_channels=N_Filters * 8)
        self._deconv_block1 = Double_Conv(in_channels=N_Filters * 8 * 2, out_channels=N_Filters * 8)
        self._up2 = Up(in_channels=N_Filters * 8, out_channels=N_Filters * 4)
        self._deconv_block2 = Double_Conv(in_channels=N_Filters * 4 * 2, out_channels=N_Filters * 4)
        self._up3 = Up(in_channels=N_Filters * 4, out_channels=N_Filters * 2)
        self._deconv_block3 = Double_Conv(in_channels=N_Filters * 2 * 2, out_channels=N_Filters * 2)
        self._up4 = Up(in_channels=N_Filters * 2, out_channels=N_Filters)
        self._deconv_block4 = Double_Conv(in_channels=N_Filters * 2, out_channels=N_Filters)

        self.out_conv = nn.Conv2d(N_Filters, n_classes, kernel_size=1, padding=0)

        self._init_weight()

    def forward(self, input1, context_hooks):
        # encode and retreive residuals
        en_fea1 = self._enconv_block1(input1)  # 280, 280, 16
        en_fea1_down = self._down1(en_fea1)  # 140, 140, 16

        en_fea2 = self._enconv_block2(en_fea1_down)  # 136, 136, 32
        en_fea2_down = self._down2(en_fea2)  # 68, 68, 32

        en_fea3 = self._enconv_block3(en_fea2_down)  # 64, 64, 64
        en_fea3_down = self._down3(en_fea3)  # 32, 32, 64

        en_fea4 = self._enconv_block4(en_fea3_down)  # 28, 28, 128
        en_fea4_down = self._down4(en_fea4)  # 14, 14, 128

        mid_fea = self._mid_conv_block(en_fea4_down)  # 10, 10, 160
        mid_fea_fuse = self.concat(mid_fea, context_hooks[3])# 10, 10, 288

        de_fea1 = self._up1(mid_fea_fuse)  # 18, 18, 128
        fea1_fuse = self.concat(de_fea1, en_fea4)
        fea1_fuse = self._deconv_block1(fea1_fuse)

        de_fea2 = self._up2(fea1_fuse)
        fea2_fuse = self.concat(de_fea2, en_fea3)
        fea2_fuse = self._deconv_block2(fea2_fuse)

        de_fea3 = self._up3(fea2_fuse)
        fea3_fuse = self.concat(de_fea3, en_fea2)
        fea3_fuse = self._deconv_block3(fea3_fuse)

        de_fea4 = self._up4(fea3_fuse)
        fea4_fuse = self.concat(de_fea4, en_fea1)
        fea4_fuse = self._deconv_block4(fea4_fuse)

        output = self.out_conv(fea4_fuse)
        output = F.softmax(output, dim=1)
        flatten = torch.reshape(output, (output.shape[0], output.shape[1], output.shape[2] * output.shape[3]))

        return flatten
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



if __name__ == '__main__':
    model = model()
    # model = Context_Branch(16, 3, 2, hook_indexes=[3, 3], n_classes=6)
    model = model.cuda()
    x1 = torch.randn(16, 3, 284, 284).cuda()
    x2 = torch.randn(16, 3, 284, 284).cuda()
    out1, out2 = model(x1, x2)
