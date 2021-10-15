import torch
import torch.nn as nn
import torch.nn.functional as F

from net.aspp import build_aspp
from net.decoder import build_decoder
# from net.backbone import build_backbone
from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
from .backbone import resnet
# class Deeplab(nn.Module):
# def Deeplab(self, backbone_name='resnet50', output_stride=16, num_classes=2, pretrain=True):
def Deeplab(backbone_name = 'resnet50', output_stride = 8, num_classes = 2, pretrain = True):

    # super(Deeplab, self).__init__()
    BachNorm = nn.BatchNorm2d
    # A = torch.randn((192, 192), requires_grad=True)
    # self.A = torch.nn.Parameter(A)
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name]( pretrained=pretrain, replace_stride_with_dilation=replace_stride_with_dilation )
    # self.backbone = build_backbone(backbone, output_stride)

    inplanes = 2048
    low_level_planes = 256
    name = 'deeplabv3plus'
    if name == 'deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name == 'deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = DeepLabV3(backbone, classifier)
    return model
        # self.aspp = build_aspp(backbone, output_stride, BachNorm)
        # self.decoder = build_decoder(num_classes, backbone, BachNorm)
        # self.register_parameter("Bingo", self.A)

    # def forward(self, input):
    #
    #     # x = input
    #     # print('input: ', input.shape)
    #     # exit()
    #     x, low_level_feat, mid_level_feat = self.backbone(input)                            #([16, 2048, 24, 24]) ([16, 256, 96, 96]) ([16, 512, 48, 48])
    #
    #     x = self.aspp(x)                                                                    #([16,  256, 24, 24])
    #     # print('x: ', x.shape)
    #     # exit()
    #     x = self.decoder(x, low_level_feat, mid_level_feat)                                 #([16,   2 , 96, 96])
    #     x = F.interpolate(x, size=input.size()[2:], mode = 'bilinear', align_corners=True)  #([16,   2 ,384,384])
    #     # x.mul(self.A)
    #
    #     return x
    #
    # def get_1x_lr_params(self):
    #     modules = [self.backbone]
    #     for i in range(len(modules)):
    #         for m in modules[i].named_modules():
    #             if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) \
    #                     or isinstance(m[1], nn.BatchNorm2d):
    #                 for p in m[1].parameters():
    #                     if p.requires_grad:
    #                         yield p
    #
    # def get_10x_lr_params(self):
    #     modules = [self.aspp, self.decoder]
    #     for i in range(len(modules)):
    #         for m in modules[i].named_modules():
    #             if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) \
    #                     or isinstance(m[1], nn.BatchNorm2d):
    #                 for p in m[1].parameters():
    #                     if p.requires_grad:
    #                         yield  p



























































