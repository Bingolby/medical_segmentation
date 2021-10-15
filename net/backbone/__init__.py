from net.backbone import resnet_bk
from . import resnet

def build_backbone(backbone, output_stride):
    if backbone == 'resnet101':
        return resnet_bk.ResNet101(output_stride)
    elif backbone == 'resnet50':
        return resnet_bk.ResNet50(output_stride)
    else:
        raise NotImplementedError