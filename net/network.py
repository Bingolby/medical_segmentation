import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus
from ._ccnet import RCCAModule
from ._bingo import bingo_cc
from ._bingo import bingo_deeplab
from ._bingo import bingo_deeplabplus
from .utils import _SimpleSegmentationModel
from .utils import bingo_SimpleSegmentationModel
from .utils import bingo_DoubleSegmentationModel
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
    # model = DeepLabV3(backbone, classifier)
    model = _SimpleSegmentationModel(backbone, classifier)
    print(name)
    return model

def CCNet(backbone_name='resnet50', output_stride=8, num_classes=2, pretrain=True):
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name]( pretrained=pretrain, replace_stride_with_dilation=replace_stride_with_dilation )
    inplanes = 2048
    low_level_planes = 256
    out_channels = 512
    recurrence = 2
    name = 'CCNet'
    if name == 'CCNetplus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name == 'CCNet':
        return_layers = {'layer4': 'out'}
        classifier = RCCAModule(inplanes, out_channels, num_classes, recurrence)
        # classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = _SimpleSegmentationModel(backbone, classifier)
    # model = CCNet(backbone, classifier)
    print(name)

    return model



def Novel_Strategy(backbone_name='resnet50', output_stride=8, num_classes=2, pretrain=True, patch_size=1536,
              mini_patch_size=384, my_batchsize=16, bingo=True):
        if output_stride == 8:
            replace_stride_with_dilation = [False, True, True]
            aspp_dilate = [12, 24, 36]
        else:
            replace_stride_with_dilation = [False, False, True]
            aspp_dilate = [6, 12, 18]

        backbone = resnet.__dict__[backbone_name](pretrained=pretrain,
                                                  replace_stride_with_dilation=replace_stride_with_dilation)
        inplanes = 2048
        low_level_planes = 256
        out_channels = 512
        recurrence = 2
        name = 'bingo_deeplabplus'
        if name == 'bingo_deeplabplus':
            output_stride = 4
            return_layers = {'layer4': 'out', 'layer1': 'low_level'}
            classifier = bingo_deeplabplus(inplanes, low_level_planes, num_classes, aspp_dilate, 
                patch_size, mini_patch_size, output_stride, bingo)
        elif name == 'bingo_cc':
            return_layers = {'layer4': 'out'}
            classifier = bingo_cc(inplanes, out_channels, num_classes, recurrence, 
                patch_size, mini_patch_size, output_stride, bingo)
        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        model = bingo_SimpleSegmentationModel(backbone, classifier, mini_patch_size, my_batchsize, bingo)
        print(name)

        return model



























































