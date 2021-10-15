import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
import sys
sys.path.append("..")
from roi_example.model_pytorch_parts import roi_pooling_2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x


class bingo_SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier, mini_patch_size, my_batchsize, bingo):
        super(bingo_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.mini_patch_size = mini_patch_size
        self.my_batchsize = my_batchsize
        self.bingo = bingo

    def forward(self, x):
        if(self.bingo):
            mini_patch_size = self.mini_patch_size
            my_batchsize = self.my_batchsize
            M = int(my_batchsize ** 0.5)
            x = x.contiguous()
            batchsize = x.shape[0]

            x_post = torch.Tensor(my_batchsize * batchsize, 3, mini_patch_size, mini_patch_size).to(device)
            for k in range(batchsize):
                for i in range(M):
                    for j in range(M):
                        x_post[k * my_batchsize + i * M + j, :, :, :] = x[k, :, i * mini_patch_size:(i + 1) * mini_patch_size, j * mini_patch_size:(j + 1) * mini_patch_size]
            x_post = x_post.contiguous()
        else:
            x_post = x
        # w = mini_patch_size
        # h = mini_patch_size
        # rois_ = torch.FloatTensor([[0, 0, w - 1, h - 1],
        #                            [w, 0, 2 * w - 1, h - 1],
        #                            [2 * w, 0, 3 * w - 1, h - 1],
        #                            [3 * w, 0, 4 * w - 1, h - 1],
        #                            [0, h, w - 1, 2 * h - 1],
        #                            [w, h, 2 * w - 1, 2 * h - 1],
        #                            [2 * w, h, 3 * w - 1, 2 * h - 1],
        #                            [3 * w, h, 4 * w - 1, 2 * h - 1],
        #                            [0, 2 * h, w - 1, 3 * h - 1],
        #                            [w, 2 * h, 2 * w - 1, 3 * h - 1],
        #                            [2 * w, 2 * h, 3 * w - 1, 3 * h - 1],
        #                            [3 * w, 2 * h, 4 * w - 1, 3 * h - 1],
        #                            [0, 3 * h, w - 1, 4 * h - 1],
        #                            [w, 3 * h, 2 * w - 1, 4 * h - 1],
        #                            [2 * w, 3 * h, 3 * w - 1, 4 * h - 1],
        #                            [3 * w, 3 * h, 4 * w - 1, 4 * h - 1]])
        # rois = torch.FloatTensor([0])
        # for i in range(x.shape[0]):
        #     index = torch.FloatTensor([i]).repeat(16, 1)
        #     temp = torch.cat([index, rois_], dim=1)
        #     if (i == 0):
        #         rois = temp
        #     else:
        #         rois = torch.cat([rois, temp], dim=0)
        # rois = rois.cuda()
        # x_post = roi_pooling_2d(x, rois, (w, h), spatial_scale=1.0).contiguous()

        input_shape = x_post.shape[-2:]

        features = self.backbone(x_post)#[32, 3, 384, 384]->[32, 2048, 48, 48]

        x = self.classifier(features)

        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        return x

class bingo_DoubleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier, mini_patch_size, my_batchsize):
        super(bingo_DoubleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.mini_patch_size = mini_patch_size
        self.my_batchsize = my_batchsize

    def forward(self, x):
        input_shape_big = x.shape[-2:]
        mini_patch_size = self.mini_patch_size
        my_batchsize = self.my_batchsize
        x_cpu = torch.Tensor(my_batchsize, 3, mini_patch_size, mini_patch_size)
        for i in range(4):
            for j in range(4):
                x_cpu[i * 4 + j, :, :, :] = x[0, :, i * mini_patch_size:(i + 1) * mini_patch_size, j * mini_patch_size:(j + 1) * mini_patch_size]
        x = x_cpu.to(device)
        # print(x.shape)
        input_shape_small = x.shape[-2:]
        features = self.backbone(x)
        # print(features['out'].shape)
        x_big, x_small = self.classifier(features)
        # print(x.shape)
        # exit()
        x_big = F.interpolate(x_big, size=input_shape_big, mode='bilinear', align_corners=False)
        x_small = F.interpolate(x_small, size=input_shape_small, mode='bilinear', align_corners=False)

        return x_big, x_small




class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out